from __future__ import annotations

import argparse
import base64
import threading
import time
from pathlib import Path

import cv2
from flask import Flask, Response, jsonify, render_template_string
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parent
DEFAULT_MODEL = WORKSPACE / "models" / "approved" / "weapon_detector_best.pt"

HTML = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Weapon Detection Stream</title>
  <style>
    body { font-family: Arial, sans-serif; background: #111; color: #eee; margin: 24px; }
    h1 { margin-bottom: 8px; }
    p { color: #bbb; }
    img { max-width: 100%; border: 2px solid #333; border-radius: 8px; min-height: 240px; background: #000; }
    .meta { margin: 12px 0 18px; font-size: 14px; }
    .status { margin: 12px 0; padding: 10px 12px; background: #1b1b1b; border-radius: 8px; }
  </style>
</head>
<body>
  <h1>Weapon Detection Stream</h1>
  <div class="meta">Open this page in any browser on this machine. The image auto-refreshes.</div>
  <div class="status" id="status">Connecting...</div>
  <img id="stream" src="/snapshot?t=0" alt="Live stream">
  <script>
    const statusEl = document.getElementById("status");
    const imgEl = document.getElementById("stream");

    async function refreshStatus() {
      try {
        const res = await fetch("/status");
        const data = await res.json();
        statusEl.textContent = data.message;
      } catch (err) {
        statusEl.textContent = "Status request failed.";
      }
    }

    function refreshImage() {
      imgEl.src = "/snapshot?t=" + Date.now();
    }

    setInterval(refreshStatus, 1000);
    setInterval(refreshImage, 150);
    refreshStatus();
  </script>
</body>
</html>
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve real-time weapon detection from an IP camera in the browser.")
    parser.add_argument("--source", required=True, help="RTSP/HTTP camera stream URL.")
    parser.add_argument("--model", default=str(DEFAULT_MODEL), help="Path to the YOLO model.")
    parser.add_argument("--conf", type=float, default=0.60, help="Confidence threshold.")
    parser.add_argument("--imgsz", type=int, default=512, help="Inference image size.")
    parser.add_argument("--device", default="0", help="Inference device, e.g. 0 or cpu.")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind the web server.")
    parser.add_argument("--port", type=int, default=5000, help="Port to bind the web server.")
    parser.add_argument("--frame-skip", type=int, default=0, help="Skip N frames between processed frames.")
    return parser.parse_args()


class StreamState:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.latest_jpeg: bytes | None = None
        self.running = True
        self.message = "Starting stream..."


def open_stream(source: str) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        cap = cv2.VideoCapture(source)
    try:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    except Exception:
        pass
    return cap


def detector_loop(args: argparse.Namespace, state: StreamState) -> None:
    model = YOLO(args.model)
    cap = open_stream(args.source)
    frame_idx = 0
    processed = 0
    start = time.time()

    if not cap.isOpened():
        with state.lock:
            state.message = f"Could not open stream: {args.source}"
            state.running = False
        return

    while state.running:
        ok, frame = cap.read()
        if not ok or frame is None:
            with state.lock:
                state.message = "Failed to read frame from stream."
            time.sleep(0.25)
            continue

        frame_idx += 1
        if args.frame_skip > 0 and frame_idx % (args.frame_skip + 1) != 1:
            continue

        with state.lock:
            state.message = f"Processing frame {processed + 1}..."

        results = model.predict(
            source=frame,
            conf=args.conf,
            imgsz=args.imgsz,
            device=args.device,
            verbose=False,
        )

        processed += 1
        elapsed = max(time.time() - start, 1e-6)
        fps = processed / elapsed
        detections = 0 if results[0].boxes is None else len(results[0].boxes)
        plotted = results[0].plot()

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

        ok, buffer = cv2.imencode(".jpg", plotted, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        if not ok:
            continue

        with state.lock:
            state.latest_jpeg = buffer.tobytes()
            state.message = f"processed={processed} fps={fps:.2f} detections={detections}"

    cap.release()


def create_app(state: StreamState) -> Flask:
    app = Flask(__name__)

    @app.get("/")
    def index() -> str:
        return render_template_string(HTML)

    @app.get("/video_feed")
    def video_feed() -> Response:
        def generate():
            while True:
                with state.lock:
                    frame = state.latest_jpeg
                    running = state.running
                if frame is not None:
                    yield (
                        b"--frame\r\n"
                        b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
                    )
                elif not running:
                    break
                time.sleep(0.03)

        return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")

    @app.get("/snapshot")
    def snapshot() -> Response:
        with state.lock:
            frame = state.latest_jpeg
        if frame is None:
            placeholder = (
                "<svg xmlns='http://www.w3.org/2000/svg' width='640' height='360'>"
                "<rect width='100%' height='100%' fill='black'/>"
                "<text x='50%' y='50%' dominant-baseline='middle' text-anchor='middle' "
                "fill='white' font-size='24'>Waiting for first processed frame...</text>"
                "</svg>"
            ).encode("utf-8")
            return Response(placeholder, mimetype="image/svg+xml")
        return Response(frame, mimetype="image/jpeg")

    @app.get("/status")
    def status() -> Response:
        with state.lock:
            return jsonify({"message": state.message, "running": state.running})

    return app


def main() -> int:
    args = parse_args()
    state = StreamState()
    thread = threading.Thread(target=detector_loop, args=(args, state), daemon=True)
    thread.start()

    print(f"Open in browser: http://{args.host}:{args.port}")
    print("Press Ctrl+C in this terminal to stop.")

    app = create_app(state)
    try:
        app.run(host=args.host, port=args.port, debug=False, threaded=True)
    finally:
        state.running = False
        thread.join(timeout=2.0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
