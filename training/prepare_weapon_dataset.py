from __future__ import annotations

import json
import os
import shutil
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parent
SOURCE = WORKSPACE / "data" / "dataset_merged"
OUTPUT = WORKSPACE / "data" / "dataset_cleaned"
SPLITS = ("train", "val", "test")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
KEEP_EMPTY_PREFIXES = {"neg"}
CLASS_NAMES = ["Weapon"]


def is_image_file(path: Path) -> bool:
    return path.suffix.lower() in IMAGE_EXTENSIONS


def parse_label_file(label_path: Path) -> tuple[list[str], list[str]]:
    valid_lines: list[str] = []
    issues: list[str] = []

    for line_number, raw_line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue

        parts = line.split()
        if len(parts) != 5:
            issues.append(f"invalid_field_count:{line_number}")
            continue

        try:
            cls = int(parts[0])
            coords = [float(value) for value in parts[1:]]
        except ValueError:
            issues.append(f"non_numeric:{line_number}")
            continue

        if cls < 0 or cls >= len(CLASS_NAMES):
            issues.append(f"invalid_class:{line_number}")
            continue

        x_center, y_center, width, height = coords
        if not (0.0 <= x_center <= 1.0 and 0.0 <= y_center <= 1.0):
            issues.append(f"center_out_of_range:{line_number}")
            continue
        if not (0.0 < width <= 1.0 and 0.0 < height <= 1.0):
            issues.append(f"box_out_of_range:{line_number}")
            continue

        valid_lines.append(f"{cls} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")

    return valid_lines, issues


def recreate_output_dirs() -> None:
    if OUTPUT.exists():
        for path in sorted(OUTPUT.rglob("*"), key=lambda item: len(item.parts), reverse=True):
            if path.is_file() or path.is_symlink():
                path.unlink(missing_ok=True)
            elif path.is_dir():
                try:
                    path.rmdir()
                except OSError:
                    for child in path.iterdir():
                        if child.is_file() or child.is_symlink():
                            child.unlink(missing_ok=True)
                    path.rmdir()
        OUTPUT.rmdir()

    for split in SPLITS:
        (OUTPUT / split / "images").mkdir(parents=True, exist_ok=True)
        (OUTPUT / split / "labels").mkdir(parents=True, exist_ok=True)


def copy_dataset() -> dict[str, dict[str, object]]:
    report: dict[str, dict[str, object]] = {}

    for split in SPLITS:
        split_report: dict[str, object] = {
            "kept_images": 0,
            "kept_negative_images": 0,
            "dropped_unintended_empty_labels": 0,
            "dropped_missing_label": 0,
            "dropped_invalid_labels": 0,
            "invalid_label_issue_counts": {},
            "kept_by_prefix": {},
            "dropped_empty_by_prefix": {},
            "dropped_invalid_examples": [],
        }

        kept_by_prefix: Counter[str] = Counter()
        dropped_empty_by_prefix: Counter[str] = Counter()
        invalid_issue_counts: Counter[str] = Counter()
        dropped_invalid_examples: list[str] = []

        image_dir = SOURCE / split / "images"
        label_dir = SOURCE / split / "labels"

        for image_path in sorted(image_dir.iterdir()):
            if not image_path.is_file() or not is_image_file(image_path):
                continue

            label_path = label_dir / f"{image_path.stem}.txt"
            prefix = image_path.stem.split("_")[0]

            if not label_path.exists():
                split_report["dropped_missing_label"] += 1
                continue

            valid_lines, issues = parse_label_file(label_path)

            if issues:
                split_report["dropped_invalid_labels"] += 1
                invalid_issue_counts.update(issues)
                if len(dropped_invalid_examples) < 20:
                    dropped_invalid_examples.append(image_path.name)
                continue

            if not valid_lines and prefix not in KEEP_EMPTY_PREFIXES:
                split_report["dropped_unintended_empty_labels"] += 1
                dropped_empty_by_prefix[prefix] += 1
                continue

            destination_image = OUTPUT / split / "images" / image_path.name
            destination_label = OUTPUT / split / "labels" / label_path.name
            shutil.copy2(image_path, destination_image)
            destination_label.write_text("\n".join(valid_lines), encoding="utf-8")

            split_report["kept_images"] += 1
            kept_by_prefix[prefix] += 1
            if not valid_lines:
                split_report["kept_negative_images"] += 1

        split_report["kept_by_prefix"] = dict(sorted(kept_by_prefix.items()))
        split_report["dropped_empty_by_prefix"] = dict(sorted(dropped_empty_by_prefix.items()))
        split_report["invalid_label_issue_counts"] = dict(sorted(invalid_issue_counts.items()))
        split_report["dropped_invalid_examples"] = dropped_invalid_examples
        report[split] = split_report

    return report


def write_data_yaml() -> None:
    yaml_text = "\n".join(
        [
            "path: .",
            "train: train/images",
            "val: val/images",
            "test: test/images",
            f"nc: {len(CLASS_NAMES)}",
            "names: [" + ", ".join(CLASS_NAMES) + "]",
            "",
        ]
    )
    (OUTPUT / "data.yaml").write_text(yaml_text, encoding="utf-8")


def write_report(report: dict[str, dict[str, object]]) -> None:
    totals = defaultdict(int)
    for split_report in report.values():
        for key in (
            "kept_images",
            "kept_negative_images",
            "dropped_unintended_empty_labels",
            "dropped_missing_label",
            "dropped_invalid_labels",
        ):
            totals[key] += int(split_report[key])

    payload = {
        "source_dataset": str(SOURCE),
        "output_dataset": str(OUTPUT),
        "policy": {
            "keep_all_valid_positive_labels": True,
            "keep_empty_labels_for_prefixes": sorted(KEEP_EMPTY_PREFIXES),
            "drop_empty_labels_for_other_prefixes": True,
            "drop_missing_or_invalid_labels": True,
        },
        "class_names": CLASS_NAMES,
        "splits": report,
        "totals": dict(sorted(totals.items())),
    }

    (OUTPUT / "cleaning_report.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    if not SOURCE.exists():
        raise SystemExit(f"Source dataset not found: {SOURCE}")

    recreate_output_dirs()
    report = copy_dataset()
    write_data_yaml()
    write_report(report)

    print(f"Created cleaned dataset at: {OUTPUT}")
    for split in SPLITS:
        split_report = report[split]
        print(
            f"{split}: kept={split_report['kept_images']} "
            f"kept_neg={split_report['kept_negative_images']} "
            f"dropped_empty={split_report['dropped_unintended_empty_labels']} "
            f"dropped_missing={split_report['dropped_missing_label']} "
            f"dropped_invalid={split_report['dropped_invalid_labels']}"
        )


if __name__ == "__main__":
    main()
