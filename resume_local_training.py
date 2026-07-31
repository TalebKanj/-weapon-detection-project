from pathlib import Path

from ultralytics import YOLO


ROOT = Path(__file__).resolve().parent
CHECKPOINT = ROOT / "weapon_training_results_from_colab" / "weapon_training_results" / "yolo26s_weapon_run1-3" / "weights" / "last.pt"
DATA = ROOT / "dataset_merged" / "data.yaml"
PROJECT = ROOT / "local_training_results"
RUN_NAME = "yolo26s_weapon_continue_from32"


def main() -> None:
    print(f"Checkpoint: {CHECKPOINT}")
    print(f"Dataset: {DATA}")
    print(f"Project: {PROJECT}")
    model = YOLO(str(CHECKPOINT))
    model.train(
        data=str(DATA),
        epochs=18,
        imgsz=640,
        batch=16,
        workers=2,
        device=0,
        project=str(PROJECT),
        name=RUN_NAME,
        pretrained=True,
        cache=False,
    )


if __name__ == "__main__":
    main()

