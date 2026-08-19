"""
Live object tracker for an ESP32-CAM running the CameraWebServer sketch.

Connects to the board's MJPEG stream (served on port 81 at /stream) and runs
YOLOv8 object detection + ByteTrack on every frame to follow a target class
(person, by default). Unlike a classic single-object CV tracker (CSRT etc.),
the detector re-finds the target from scratch each frame, so there's no
drift, no manual re-selection after occlusion, and no need for a hand-picked
ROI - it just keeps detecting.

When multiple instances of the target class are in view, the tracker sticks
to whichever ByteTrack ID it last locked onto (so it doesn't hop between
people frame to frame). If that ID disappears, it re-locks onto whichever
detection is currently closest to the frame center.

Usage:
    python tracker.py --host 192.168.1.50
    (find the IP from the ESP32's serial monitor output on boot)

    python tracker.py --host 192.168.1.50 --class-id 32   # track a sports ball instead
    python tracker.py --host 192.168.1.50 --select-roi     # drag a box around the object to track
    python tracker.py --list-classes                       # print COCO class id -> name

Controls:
    r - freeze the frame and drag a box around a new target to lock onto
    q - quit

If the feed is too dark, try (AI-Thinker board has an onboard flash LED):
    python tracker.py --host 192.168.1.50 --led 80
    python tracker.py --host 192.168.1.50 --ae-level 2 --brightness 2

For more vivid color:
    python tracker.py --host 192.168.1.50 --saturation 2 --contrast 1
"""

import argparse
import threading
import time

import cv2
import numpy as np
import requests
import torch
from ultralytics import YOLO


def frame_generator(stream_url, timeout=10):
    stream = requests.get(stream_url, stream=True, timeout=timeout)
    buffer = bytearray()
    for chunk in stream.iter_content(chunk_size=4096):
        buffer += chunk
        while True:
            start = buffer.find(b"\xff\xd8")
            if start == -1:
                buffer.clear()
                break
            end = buffer.find(b"\xff\xd9", start + 2)
            if end == -1:
                if start > 0:
                    del buffer[:start]
                break
            jpg = bytes(buffer[start : end + 2])
            del buffer[: end + 2]
            frame = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
            if frame is not None:
                yield frame


class StreamReader:
    """Continuously drains the MJPEG stream on a background thread so a
    blocking UI call on the main thread never stalls the network read and
    breaks the connection. Reconnects automatically on transient errors."""

    def __init__(self, stream_url):
        self.stream_url = stream_url
        self._frame = None
        self._lock = threading.Lock()
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self._thread.start()
        return self

    def _run(self):
        while self._running:
            try:
                for frame in frame_generator(self.stream_url):
                    with self._lock:
                        self._frame = frame
                    if not self._running:
                        return
            except requests.exceptions.RequestException as exc:
                print(f"Stream error ({exc}); reconnecting...")
                time.sleep(1)

    def read(self):
        with self._lock:
            return None if self._frame is None else self._frame.copy()

    def stop(self):
        self._running = False


def set_control(host, var, val):
    url = f"http://{host}/control?var={var}&val={val}"
    try:
        requests.get(url, timeout=5)
    except requests.exceptions.RequestException as exc:
        print(f"Warning: failed to set {var}={val} ({exc})")


def pick_device(requested):
    if requested is not None:
        return requested
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def parse_detections(results):
    """Extracts (box, track_id) pairs from a single model.track() result.
    Detections without a track ID (tracker hasn't confirmed them yet) are
    dropped since locking relies on stable IDs."""
    boxes = results[0].boxes
    if boxes is None or boxes.id is None:
        return []
    xyxy = boxes.xyxy.cpu().numpy()
    ids = boxes.id.int().cpu().numpy()
    return list(zip(xyxy, ids))


def iou(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / union if union > 0 else 0.0


def select_target_class(detector, frame, conf, device):
    """Lets the user drag a box around the object to track on the given frame,
    then matches it against the detector's detections on that same frame to
    infer which COCO class it belongs to. `detector` must be a YOLO instance
    that has never had .track() called on it before this - calling .track()
    permanently attaches a tracking callback to a model that then filters all
    later results (even from .predict()) down to already-activated ByteTrack
    tracks, hiding any object outside the class currently being tracked.
    persist=False here so every call is a fresh single-frame tracking session
    (frame_id resets to 1), which ByteTrack always activates immediately.
    Returns the matched class id, or None if nothing was selected or matched
    closely enough."""
    window = "Drag a box around the object to track, then press ENTER (c to cancel)"
    rx, ry, rw, rh = cv2.selectROI(window, frame, showCrosshair=True)
    cv2.destroyWindow(window)
    if rw == 0 or rh == 0:
        return None
    roi_box = (rx, ry, rx + rw, ry + rh)

    results = detector.track(frame, persist=False, tracker="bytetrack.yaml", conf=conf, device=device, verbose=False)
    boxes = results[0].boxes
    if boxes is None or len(boxes) == 0:
        print(f"Detector found no objects at all at conf>={conf} (try lowering --conf).")
        return None
    xyxy = boxes.xyxy.cpu().numpy()
    class_ids = boxes.cls.int().cpu().numpy()

    best_iou, best_class = 0.0, None
    for box, class_id in zip(xyxy, class_ids):
        score = iou(roi_box, box)
        if score > best_iou:
            best_iou, best_class = score, int(class_id)

    if best_class is None or best_iou < 0.1:
        names = [detector.names[int(c)] for c in class_ids]
        print(f"Found {len(class_ids)} object(s) ({names}) but none overlapped the box (best IoU={best_iou:.2f}).")
        return None
    return best_class


def select_target(detections, locked_id, frame_center):
    """Sticks to the currently locked track ID if it's still detected;
    otherwise (re-)locks onto whichever detection is closest to the frame
    center. Returns (box, track_id) or (None, None) if nothing is detected."""
    if locked_id is not None:
        for box, track_id in detections:
            if track_id == locked_id:
                return box, track_id

    if not detections:
        return None, None

    def dist_to_center(item):
        x1, y1, x2, y2 = item[0]
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        return (cx - frame_center[0]) ** 2 + (cy - frame_center[1]) ** 2

    box, track_id = min(detections, key=dist_to_center)
    return box, track_id


def draw_overlay(frame, box, track_id):
    x1, y1, x2, y2 = [int(v) for v in box]
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)
    cv2.line(frame, (cx - 15, cy), (cx + 15, cy), (0, 255, 255), 1)
    cv2.line(frame, (cx, cy - 15), (cx, cy + 15), (0, 255, 255), 1)

    frame_h, frame_w = frame.shape[:2]
    offset_x, offset_y = cx - frame_w // 2, cy - frame_h // 2
    cv2.putText(
        frame,
        f"ID:{track_id} dx={offset_x} dy={offset_y}",
        (x1, max(y1 - 10, 15)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 0),
        2,
    )


def draw_status(frame, text, color=(0, 255, 255)):
    cv2.putText(frame, text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)


def main():
    parser = argparse.ArgumentParser(description="ESP32-CAM live object tracker (YOLOv8 + ByteTrack)")
    parser.add_argument("--host", help="ESP32-CAM IP address, e.g. 192.168.1.50")
    parser.add_argument("--port", type=int, default=81, help="Stream server port (default 81)")
    parser.add_argument("--led", type=int, help="Onboard flash LED intensity, 0-255 (AI-Thinker board only)")
    parser.add_argument("--ae-level", type=int, choices=range(-2, 3), help="Auto exposure target level, -2 to 2")
    parser.add_argument("--brightness", type=int, choices=range(-2, 3), help="Brightness offset, -2 to 2")
    parser.add_argument("--saturation", type=int, choices=range(-2, 3), help="Color saturation, -2 to 2")
    parser.add_argument("--contrast", type=int, choices=range(-2, 3), help="Contrast, -2 to 2")
    parser.add_argument("--model", default="yolov8n.pt", help="Ultralytics model weights (default yolov8n.pt)")
    parser.add_argument("--class-id", type=int, default=75, help="COCO class id to track (default 0 = person)")
    parser.add_argument("--conf", type=float, default=0.2, help="Detection confidence threshold (default 0.2)")
    parser.add_argument("--device", help="Inference device, e.g. mps/cuda/cpu (default: auto-detect)")
    parser.add_argument("--list-classes", action="store_true", help="Print COCO class id -> name and exit")
    parser.add_argument(
        "--select-roi",
        action="store_true",
        help="Drag a box around the object to track on the first frame instead of passing --class-id",
    )
    args = parser.parse_args()

    if args.list_classes:
        model = YOLO(args.model)
        for class_id, name in sorted(model.names.items()):
            print(f"{class_id}: {name}")
        return

    if not args.host:
        parser.error("--host is required")

    if args.led is not None:
        set_control(args.host, "led_intensity", args.led)
    if args.ae_level is not None:
        set_control(args.host, "ae_level", args.ae_level)
    if args.brightness is not None:
        set_control(args.host, "brightness", args.brightness)
    if args.saturation is not None:
        set_control(args.host, "saturation", args.saturation)
    if args.contrast is not None:
        set_control(args.host, "contrast", args.contrast)

    stream_url = f"http://{args.host}:{args.port}/stream"
    print(f"Connecting to {stream_url} ...")

    device = pick_device(args.device)
    print(f"Loading {args.model} on device={device} ...")
    model = YOLO(args.model)
    # Separate instance for ROI class-picking: model.track() permanently attaches a
    # tracking callback to whichever model it's called on, which then keeps filtering
    # results down to activated tracks even on later plain .predict() calls. A model
    # that never has .track() called on it stays a clean, stateless detector.
    detector = YOLO(args.model)

    window = "ESP32-CAM Object Tracker"
    locked_id = None

    reader = StreamReader(stream_url).start()

    print("Waiting for first frame...")
    while reader.read() is None:
        time.sleep(0.05)

    if args.select_roi:
        class_id = select_target_class(detector, reader.read(), args.conf, device)
        if class_id is None:
            print("No object detected in the selected box; falling back to --class-id.")
        else:
            args.class_id = class_id
            print(f"Tracking class {args.class_id} ({model.names[args.class_id]}); acquiring nearest match...")

    while True:
        frame = reader.read()
        if frame is None:
            continue

        frame_h, frame_w = frame.shape[:2]
        results = model.track(
            frame,
            persist=True,
            tracker="bytetrack.yaml",
            classes=[args.class_id],
            conf=args.conf,
            device=device,
            verbose=False,
        )
        detections = parse_detections(results)
        box, track_id = select_target(detections, locked_id, (frame_w / 2, frame_h / 2))
        locked_id = track_id

        if box is not None:
            draw_overlay(frame, box, track_id)
        else:
            draw_status(frame, "TARGET LOST - searching...", (0, 0, 255))

        cv2.imshow(window, frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break
        elif key == ord("r"):
            class_id = select_target_class(detector, frame, args.conf, device)
            if class_id is None:
                print("No object detected in the selected box; keeping current target.")
            else:
                args.class_id, locked_id = class_id, None
                print(f"Tracking class {args.class_id} ({model.names[args.class_id]}); re-acquiring nearest match...")

    reader.stop()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
