from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL = ROOT / "local_training_results" / "yolo26s_weapon_continue_from32" / "weights" / "best.pt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run real-time weapon detection on an IP camera stream.")
    parser.add_argument("--source", required=True, help="IP camera stream URL, e.g. rtsp://... or http://...")
    parser.add_argument("--model", default=str(DEFAULT_MODEL), help="Path to a YOLO model file.")
    parser.add_argument("--conf", type=float, default=0.60, help="Confidence threshold.")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size.")
    parser.add_argument("--device", default="0", help="Inference device. Use 0 for GPU or cpu.")
    parser.add_argument("--save", default="", help="Optional output video path, e.g. output.mp4")
    parser.add_argument("--window", default="Weapon Detection", help="OpenCV window title.")
    parser.add_argument("--status-every", type=int, default=10, help="Print terminal status every N processed frames.")
    parser.add_argument("--frame-skip", type=int, default=0, help="Skip N frames between processed frames to reduce latency.")
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Disable OpenCV window display. Useful when OpenCV was built without GUI support.",
    )
    return parser.parse_args()


def open_stream(source: str) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        # Fallback for sources that do not work with FFMPEG backend.
        cap = cv2.VideoCapture(source)
    try:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    except Exception:
        pass
    return cap


def create_writer(save_path: str, fps: float, width: int, height: int) -> cv2.VideoWriter:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    return cv2.VideoWriter(save_path, fourcc, fps, (width, height))


def can_use_imshow() -> bool:
    try:
        cv2.namedWindow("__opencv_test__", cv2.WINDOW_NORMAL)
        cv2.destroyWindow("__opencv_test__")
        return True
    except cv2.error:
        return False


def main() -> int:
    args = parse_args()
    model_path = Path(args.model)

    if not model_path.exists():
        print(f"Model not found: {model_path}")
        return 1

    model = YOLO(str(model_path))
    cap = open_stream(args.source)

    if not cap.isOpened():
        print(f"Could not open stream: {args.source}")
        return 1

    writer: cv2.VideoWriter | None = None
    frame_count = 0
    raw_frame_count = 0
    start_time = time.time()
    display_enabled = not args.no_display and can_use_imshow()

    print("Stream opened successfully.")
    if display_enabled:
        print("Press 'q' to quit.")
    else:
        print("OpenCV GUI is unavailable, so live window display is disabled.")
        print("Processing will continue without a window.")

    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                print("Stream frame read failed. Stopping.")
                break

            raw_frame_count += 1
            if args.frame_skip > 0 and raw_frame_count % (args.frame_skip + 1) != 1:
                continue

            results = model.predict(
                source=frame,
                conf=args.conf,
                imgsz=args.imgsz,
                device=args.device,
                verbose=False,
            )

            plotted = results[0].plot()
            frame_count += 1
            elapsed = max(time.time() - start_time, 1e-6)
            fps = frame_count / elapsed
            detections = 0 if results[0].boxes is None else len(results[0].boxes)

            if args.status_every > 0 and frame_count % args.status_every == 0:
                print(
                    f"processed={frame_count} raw_frames={raw_frame_count} "
                    f"fps={fps:.2f} detections={detections}",
                    flush=True,
                )

            cv2.putText(
                plotted,
                f"FPS: {fps:.2f} Dets: {detections}",
                (12, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

            if args.save and writer is None:
                height, width = plotted.shape[:2]
                source_fps = cap.get(cv2.CAP_PROP_FPS)
                writer = create_writer(args.save, source_fps if source_fps > 0 else 20.0, width, height)

            if writer is not None:
                writer.write(plotted)

            if display_enabled:
                try:
                    cv2.imshow(args.window, plotted)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q"):
                        break
                except cv2.error:
                    print("OpenCV GUI display failed during runtime. Continuing without a window.")
                    display_enabled = False
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        if display_enabled:
            try:
                cv2.destroyAllWindows()
            except cv2.error:
                pass

    print("Detection stopped cleanly.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
