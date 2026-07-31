# Weapon Detection Project

This repository contains the training and inference scripts for a real-time weapon detection project based on Ultralytics YOLO, along with selected training result folders and exported model weights.

The datasets are intentionally excluded from Git so the repository stays lighter and safer to publish.

## Included

- Training scripts
- Dataset preparation and merge scripts
- IP camera real-time detection scripts
- Saved training results
- Best model checkpoints already produced during training

## Excluded From Git

The following dataset folders are ignored and will not be uploaded:

- `dataset_cleaned/`
- `dataset_merged/`
- `dataset_train_ready/`
- `roboflow_cctv_weapon/`

The dataset archive `dataset_merged.zip` is also ignored.

## Main Files

- `prepare_weapon_dataset.py`
- `merge_weapon_datasets.py`
- `train_weapon_merged.py`
- `resume_local_training.py`
- `run_ip_camera_detection.py`
- `run_ip_camera_web.py`

## Main Result Folders

- `training_runs/`
- `local_training_results/`
- `weapon_training_results_from_colab/`
- `runs/`

## Install

```bash
pip install -r requirements.txt
```

## Real-Time Detection From IP Camera

Terminal mode:

```bash
python run_ip_camera_detection.py --source "rtsp://USERNAME:PASSWORD@IP:554/cam/realmonitor?channel=1&subtype=1"
```

Browser mode:

```bash
python run_ip_camera_web.py --source "rtsp://USERNAME:PASSWORD@IP:554/cam/realmonitor?channel=1&subtype=1" --imgsz 416 --frame-skip 1 --host 127.0.0.1 --port 8008
```

Then open:

```text
http://127.0.0.1:8008
```

## Notes

- The default confidence threshold in the inference scripts is `0.60`.
- The web viewer is useful when OpenCV GUI support is unavailable on Windows.
- Some scripts expect dataset paths to exist locally if you want to retrain.

## Suggested GitHub Upload Steps

```bash
git init
git add .
git commit -m "Initial weapon detection project"
```

Then create a GitHub repository and connect it:

```bash
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO.git
git branch -M main
git push -u origin main
```

