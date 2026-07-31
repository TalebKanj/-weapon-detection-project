from __future__ import annotations

from pathlib import Path

from ultralytics import YOLO


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "dataset_train_ready" / "data.yaml"
PROJECT = ROOT / "training_runs"
RUN_NAME = "yolo26s_weapon_merged_v1"
MODEL = "yolo26s.pt"


def main() -> None:
    print(f"Data: {DATA}")
    print(f"Project: {PROJECT}")
    print(f"Run name: {RUN_NAME}")
    model = YOLO(MODEL)
    model.train(
        data=str(DATA),
        epochs=40,
        imgsz=640,
        batch=8,
        workers=2,
        device=0,
        project=str(PROJECT),
        name=RUN_NAME,
        pretrained=True,
        cache=False,
        patience=15,
    )


if __name__ == "__main__":
    main()
