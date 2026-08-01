from __future__ import annotations

import json
import shutil
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parent
PRIMARY = WORKSPACE / "data" / "dataset_merged"
SECONDARY = WORKSPACE / "data" / "roboflow_cctv_weapon"
OUTPUT = WORKSPACE / "data" / "dataset_train_ready"

PRIMARY_SPLITS = {"train": "train", "val": "val", "test": "test"}
SECONDARY_SPLITS = {"train": "train", "valid": "val", "test": "test"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def recreate_output() -> None:
    if OUTPUT.exists():
        shutil.rmtree(OUTPUT)
    for split in ("train", "val", "test"):
        (OUTPUT / split / "images").mkdir(parents=True, exist_ok=True)
        (OUTPUT / split / "labels").mkdir(parents=True, exist_ok=True)


def copy_primary(report: dict[str, Counter]) -> None:
    for src_split, dst_split in PRIMARY_SPLITS.items():
        image_dir = PRIMARY / src_split / "images"
        label_dir = PRIMARY / src_split / "labels"
        for image_path in sorted(image_dir.iterdir()):
            if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            label_path = label_dir / f"{image_path.stem}.txt"
            if not label_path.exists():
                continue
            shutil.copy2(image_path, OUTPUT / dst_split / "images" / image_path.name)
            shutil.copy2(label_path, OUTPUT / dst_split / "labels" / label_path.name)
            report[dst_split]["primary_images"] += 1


def convert_secondary_label(label_path: Path) -> tuple[list[str], bool]:
    converted: list[str] = []
    had_person = False
    for raw_line in label_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 5:
            continue
        cls = parts[0]
        if cls == "1":
            converted.append("0 " + " ".join(parts[1:]))
        elif cls == "0":
            had_person = True
    return converted, had_person


def copy_secondary(report: dict[str, Counter]) -> None:
    for src_split, dst_split in SECONDARY_SPLITS.items():
        image_dir = SECONDARY / src_split / "images"
        label_dir = SECONDARY / src_split / "labels"
        for image_path in sorted(image_dir.iterdir()):
            if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            label_path = label_dir / f"{image_path.stem}.txt"
            if not label_path.exists():
                continue
            converted, had_person = convert_secondary_label(label_path)
            new_stem = f"rf_{src_split}_{image_path.stem}"
            dst_image = OUTPUT / dst_split / "images" / f"{new_stem}{image_path.suffix.lower()}"
            dst_label = OUTPUT / dst_split / "labels" / f"{new_stem}.txt"
            shutil.copy2(image_path, dst_image)
            dst_label.write_text("\n".join(converted), encoding="utf-8")
            report[dst_split]["secondary_images"] += 1
            if converted:
                report[dst_split]["secondary_weapon_images"] += 1
            else:
                report[dst_split]["secondary_background_images"] += 1
            if had_person:
                report[dst_split]["secondary_person_labels_removed"] += 1


def write_yaml() -> None:
    text = "\n".join(
        [
            f"path: {OUTPUT.as_posix()}",
            "train: train/images",
            "val: val/images",
            "test: test/images",
            "nc: 1",
            "names: [Weapon]",
            "",
        ]
    )
    (OUTPUT / "data.yaml").write_text(text, encoding="utf-8")


def write_report(report: dict[str, Counter]) -> None:
    payload = {
        split: dict(counter)
        for split, counter in report.items()
    }
    (OUTPUT / "merge_report.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def print_summary(report: dict[str, Counter]) -> None:
    print(f"Created merged dataset at: {OUTPUT}")
    for split, counter in report.items():
        total = counter["primary_images"] + counter["secondary_images"]
        print(
            f"{split}: total={total} "
            f"primary={counter['primary_images']} "
            f"secondary={counter['secondary_images']} "
            f"secondary_weapon={counter['secondary_weapon_images']} "
            f"secondary_background={counter['secondary_background_images']}"
        )


def main() -> None:
    report = {split: Counter() for split in ("train", "val", "test")}
    recreate_output()
    copy_primary(report)
    copy_secondary(report)
    write_yaml()
    write_report(report)
    print_summary(report)


if __name__ == "__main__":
    main()
