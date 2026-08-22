"""
Live object tracker for an ESP32-CAM running the CameraWebServer sketch,
with optional closed-loop driving of the motor car (see motor_car/motor_car.ino)
to keep the tracked object centered in frame.

Connects to the camera board's MJPEG stream (served on port 81 at /stream)
and runs YOLOv8 object detection + ByteTrack on every frame (mirroring
object_tracker/tracker.py) instead of a classic single-object CV tracker.
The detector re-finds objects from scratch each frame, so there's no drift
and no manual ROI to draw.

Selection logic: at startup no object class is fixed, so nothing is
detected or tracked yet. Press 'r' to drag a box around the object you
want; a one-off, class-unfiltered detection pass matches (by IoU) whatever
overlaps that box to figure out its COCO class, and detection is then
narrowed to just that class from then on (e.g. selecting a person
restricts every later frame to detecting people). Once a class is set, if
nothing is locked yet the tracker locks onto whichever detection of that
class currently has the largest bounding-box area; once locked, it sticks
to that ByteTrack ID frame to frame (so it doesn't hop between objects)
until that ID disappears, at which point it falls back to re-locking onto
the current biggest detection of the same class. Press 'r' again at any
time to pick a different object (and class) to track.

When --car-host is given, it also drives the car via the motor board's
/move?forward=&left= endpoint to keep the locked object centered (rotating
in place to correct left/right offset, and optionally driving forward/back
to hold the object's on-screen size).

Usage:
    python car_controller.py --host 192.168.1.50
    (find the camera's IP from the ESP32's serial monitor output on boot)

    python car_controller.py --host 192.168.1.50 --car-host 192.168.1.60
    (also drive the car to keep the locked object centered)

    python car_controller.py --host 192.168.1.50 --classes 0
    (skip the 'r' selection step and start tracking COCO class 0 (person)
    directly; pressing 'r' still lets you switch to a different object/class)

Controls:
    r - drag a box around the object to track; a one-off detection pass
        figures out its class, and tracking narrows to that class from then
        on. Cancel with ENTER/c on an empty box to keep the current target.
    c - toggle auto-drive on/off without dropping the lock
    s - emergency stop (halts the car, keeps tracking)
    q - quit (stops the car first)

If the feed is too dark, try (AI-Thinker board has an onboard flash LED):
    python car_controller.py --host 192.168.1.50 --led 80
    python car_controller.py --host 192.168.1.50 --ae-level 2 --brightness 2

For more vivid color:
    python car_controller.py --host 192.168.1.50 --saturation 2 --contrast 1
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


def clamp(value, low, high):
    return max(low, min(high, value))


class CarController:
    """Drives the motor car via its /move?forward=&left= endpoint (see
    motor_car/motor_car.ino) to keep a tracked object centered.

    Commands are sent from a background thread at a fixed rate rather than
    inline with the video loop, so sending synchronously from the frame
    loop can't stall the UI waiting on a request.

    The car's firmware auto-stops any movement 0.2s after the command that
    started it, so a nonzero target is resent every send_interval (which
    must stay below that 0.2s window) to keep the car moving continuously;
    a stale/delayed decision can't leave it running unchecked, since the
    firmware itself is what enforces the timeout.
    """

    def __init__(self, host, port=80, send_interval=0.15, timeout=3):
        self.base_url = f"http://{host}:{port}"
        self.send_interval = send_interval
        self.timeout = timeout
        self._target = (0.0, 0.0, 0.0)
        self._sent = (None, None, None)
        self._lock = threading.Lock()
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self._thread.start()
        return self

    def set_target(self, forward, left, shift_left):
        with self._lock:
            self._target = (forward, left, shift_left)

    def stop_moving(self):
        self.set_target(0.0, 0.0, 0.0)

    def _run(self):
        while self._running:
            with self._lock:
                target = self._target

            print(
                f"Car command: forward={target[0]:+.2f} left={target[1]:+.2f} "
                f"shift_left={target[2]:+.2f}"
            )
            if target != (0.0, 0.0, 0.0) or target != self._sent:
                try:
                    requests.get(
                        f"{self.base_url}/move",
                        params={"forward": target[0], "left": target[1], "shift_left": target[2]},
                        timeout=self.timeout,
                    )
                    self._sent = target
                except requests.exceptions.RequestException as exc:
                    print(f"Car command failed ({exc})")
            time.sleep(self.send_interval)

    def stop(self):
        """Zero the car's speed and shut down the sender thread."""
        self.stop_moving()
        time.sleep(self.send_interval + 0.2)
        self._running = False
        self._thread.join(timeout=1)

def sign(x):
    return 1 if x > 0 else -1 if x < 0 else 0

def compute_drive_command(
    bbox,
    frame_shape,
    target_area_frac_lower,
    target_area_frac_upper,
    kp_forward,
    kp_left,
    kp_shift,
    deadzone_x=0.08,
    quarter_threshold=0.5,
    max_speed=0.4,
    max_left=0.2,
    max_shift=0.2,
    min_move=0.08,
):
    """Proportional controller that keeps a tracked bbox centered.

    Horizontal offset drives one of two corrections, never both at once:
    when the object is out in the left or right quarter of the frame
    (|dx_norm| > quarter_threshold, since dx_norm spans -1..1 across the
    full width so each quarter is 0.5 wide), the car rotates in place
    ("left") toward that side, which pans the camera that way and pulls the
    object back toward center. Once the object is closer in (inside the two
    middle quarters but still outside the dead zone), rotation overshoots
    for how little correction is needed, so the car instead shifts
    laterally ("shift_left") for finer, more precise centering.

    Bounding-box area, as a fraction of the frame area, drives forward/back:
    below target_area_frac_lower the object is too small/far, so the car
    moves forward, proportionally to how far below the threshold it is;
    above target_area_frac_upper the object is too close, so the car backs
    up, proportionally to how far above; in between (the "hold" band) the
    car does neither. Pass both as None to disable forward/back driving
    entirely. This runs independently of horizontal centering, so the car
    keeps closing/holding distance even while dx is already zero.

    Horizontal correction only kicks in once the object drifts outside the
    horizontal dead zone; within it, both left and shift_left are zero.
    Forward/back speed is capped at max_speed, rotation ("left") at
    max_left, and lateral shift ("shift_left") at max_shift.

    Below min_move the motor can't overcome static friction, so it just
    strains without actually moving the car; all axes are snapped to 0
    below that magnitude so the motor stops trying.
    """
    frame_h, frame_w = frame_shape[:2]
    x, y, w, h = bbox
    cx = x + w / 2
    dx_norm = (cx - frame_w / 2) / (frame_w / 2)

    left = 0.0
    shift_left = 0.0
    if abs(dx_norm) > deadzone_x:
        if abs(dx_norm) > quarter_threshold:
            left = clamp(-kp_left * dx_norm, -max_left, max_left)
        else:
            shift_left = clamp(-kp_shift * dx_norm, -max_shift, max_shift)

    forward = 0.0
    real_frac = (w * h) / (frame_w * frame_h)
    if target_area_frac_lower is not None and real_frac < target_area_frac_lower:
        error = (target_area_frac_lower - real_frac) / target_area_frac_lower
        forward = clamp(kp_forward * error, 0.0, max_speed)
    elif target_area_frac_upper is not None and real_frac > target_area_frac_upper:
        error = (real_frac - target_area_frac_upper) / target_area_frac_upper
        forward = clamp(-kp_forward * error, -max_speed, 0.0)

    if abs(forward) < min_move:
        forward = 0.0
    if abs(left) < min_move:
        left = 0.0
    if abs(shift_left) < min_move:
        shift_left = 0.0

    return forward, left, shift_left


def pick_device(requested):
    if requested is not None:
        return requested
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def parse_detections(results):
    """Extracts (box, track_id, class_id) triples from a single
    model.track() result. Detections without a track ID (tracker hasn't
    confirmed them yet) are dropped since locking relies on stable IDs."""
    boxes = results[0].boxes
    if boxes is None or boxes.id is None:
        return []
    xyxy = boxes.xyxy.cpu().numpy()
    ids = boxes.id.int().cpu().numpy()
    classes = boxes.cls.int().cpu().numpy()
    return list(zip(xyxy, ids, classes))


def box_area(box):
    x1, y1, x2, y2 = box
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def select_target(detections, locked_id):
    """Sticks to the locked track ID if it's still detected; otherwise (re-)
    locks onto whichever detection currently has the largest bounding-box
    area. Returns (box, track_id) or (None, None) if nothing is detected."""
    if locked_id is not None:
        for box, track_id, _ in detections:
            if track_id == locked_id:
                return box, track_id

    if not detections:
        return None, None

    box, track_id, _ = max(detections, key=lambda item: box_area(item[0]))
    return box, track_id


def iou(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / union if union > 0 else 0.0


def select_target_class(detector, frame, conf, device, min_iou=0.1):
    """Lets the user drag a box around the object to track, then runs a
    fresh, class-unfiltered detection pass on that same frame to figure out
    which COCO class it belongs to - matching by IoU against the box's
    position, since the continuously-tracked `detections` can't be used
    here (before a class is selected, args.classes is empty, so nothing is
    being detected yet to match against).

    `detector` must be a separate YOLO instance from the one driving the
    persistent tracking loop: calling .track() on a model permanently
    attaches a tracking callback that keeps filtering later results down to
    already-activated tracks, so running this unfiltered scan on the same
    model would corrupt its persist=True tracking state.

    Returns the matched class id, or None if nothing was selected or
    nothing overlapped closely enough (in which case the current lock and
    class filter should be left alone)."""
    picker_window = "Drag a box around the object to track, then press ENTER (c to cancel)"
    rx, ry, rw, rh = cv2.selectROI(picker_window, frame, fromCenter=False, showCrosshair=True)
    cv2.destroyWindow(picker_window)
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

    if best_class is None or best_iou < min_iou:
        names = sorted({detector.names[int(c)] for c in class_ids})
        print(f"Found {len(class_ids)} object(s) ({names}) but none overlapped the box closely enough (best IoU={best_iou:.2f}).")
        return None
    return best_class


def xyxy_to_xywh(box):
    x1, y1, x2, y2 = box
    return (float(x1), float(y1), float(x2 - x1), float(y2 - y1))


def draw_overlay(frame, bbox, track_id, drive_command=None):
    x, y, w, h = [int(v) for v in bbox]
    cx, cy = x + w // 2, y + h // 2
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)
    cv2.line(frame, (cx - 15, cy), (cx + 15, cy), (0, 255, 255), 1)
    cv2.line(frame, (cx, cy - 15), (cx, cy + 15), (0, 255, 255), 1)

    frame_h, frame_w = frame.shape[:2]
    offset_x, offset_y = cx - frame_w // 2, cy - frame_h // 2
    cv2.putText(
        frame,
        f"ID:{track_id} dx={offset_x} dy={offset_y}",
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 255),
        2,
    )

    if drive_command is not None:
        forward, left, shift_left = drive_command
        cv2.putText(
            frame,
            f"drive: forward={forward:+.2f} left={left:+.2f} shift_left={shift_left:+.2f}",
            (10, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 200, 0),
            2,
        )


def draw_status(frame, text, color=(0, 255, 255)):
    cv2.putText(frame, text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)


def apply_bbox(frame, bbox, track_id, args, car, driving_enabled):
    """Drives the car (if enabled) to keep bbox centered and draws the
    overlay."""
    _, _, bw, bh = bbox
    frame_h, frame_w = frame.shape[:2]
    area_pct = 100 * (bw * bh) / (frame_w * frame_h)
    print(f"Object area: {area_pct:.1f}% of frame (ID:{track_id})")

    drive_command = None
    if car is not None:
        if driving_enabled:
            if args.no_follow_distance:
                area_lower, area_upper = None, None
            else:
                area_lower = args.target_area_frac_lower
                area_upper = args.target_area_frac_upper
            drive_command = compute_drive_command(
                bbox,
                frame.shape,
                area_lower,
                area_upper,
                args.kp_forward,
                args.kp_left,
                args.kp_shift,
                args.deadzone_x,
                args.quarter_threshold,
                args.max_speed,
                args.max_left,
                args.max_shift,
                args.min_move,
            )
            car.set_target(*drive_command)
        else:
            car.stop_moving()
    draw_overlay(frame, bbox, track_id, drive_command)


def main():
    parser = argparse.ArgumentParser(description="ESP32-CAM live object tracker (YOLOv8 + ByteTrack)")
    parser.add_argument("--host", required=True, help="ESP32-CAM IP address, e.g. 192.168.1.50")
    parser.add_argument("--port", type=int, default=81, help="Stream server port (default 81)")
    parser.add_argument("--led", type=int, help="Onboard flash LED intensity, 0-255 (AI-Thinker board only)")
    parser.add_argument("--ae-level", type=int, choices=range(-2, 3), help="Auto exposure target level, -2 to 2")
    parser.add_argument("--brightness", type=int, choices=range(-2, 3), help="Brightness offset, -2 to 2")
    parser.add_argument("--saturation", type=int, choices=range(-2, 3), help="Color saturation, -2 to 2")
    parser.add_argument("--contrast", type=int, choices=range(-2, 3), help="Contrast, -2 to 2")
    parser.add_argument("--model", default="yolov8n.pt", help="Ultralytics model weights (default yolov8n.pt)")
    parser.add_argument(
        "--classes",
        type=int,
        nargs="+",
        default=[],
        help="Start tracking these COCO class ids immediately, skipping the 'r' selection step "
        "(default: none - nothing is detected until you press 'r'). Pressing 'r' overrides this "
        "to whichever single class you select.",
    )
    parser.add_argument("--conf", type=float, default=0.2, help="Detection confidence threshold (default 0.2)")
    parser.add_argument("--device", help="Inference device, e.g. mps/cuda/cpu (default: auto-detect)")
    parser.add_argument(
        "--car-host",
        help="Motor car IP address (the board running motor_car.ino). "
        "If given, the car auto-drives to keep the tracked object centered.",
    )
    parser.add_argument("--car-port", type=int, default=80, help="Motor car HTTP port (default 80)")
    parser.add_argument("--kp-left", type=float, default=0.15, help="Proportional gain for left/right rotation correction")
    parser.add_argument("--kp-shift", type=float, default=0.6, help="Proportional gain for lateral shift correction")
    parser.add_argument("--kp-forward", type=float, default=0.6, help="Proportional gain for forward/back distance correction")
    parser.add_argument("--deadzone-x", type=float, default=0.15, help="Horizontal dead zone as a fraction of half frame width")
    parser.add_argument(
        "--quarter-threshold",
        type=float,
        default=0.5,
        help="Horizontal offset (as a fraction of half frame width, 0-1) beyond which the object "
        "is considered to be in the left/right quarter of the frame and the car rotates to "
        "recenter it; below this (but still outside the dead zone) it shifts laterally instead "
        "for finer correction (default 0.5)",
    )
    parser.add_argument("--max-speed", type=float, default=0.4, help="Cap on forward/back drive magnitude, 0-1 (default 0.4)")
    parser.add_argument("--max-left", type=float, default=0.2, help="Cap on rotation (left/right) drive magnitude, 0-1 (default 0.2)")
    parser.add_argument("--max-shift", type=float, default=0.2, help="Cap on lateral shift drive magnitude, 0-1 (default 0.2)")
    parser.add_argument(
        "--min-move",
        type=float,
        default=0.1,
        help="Forward/left values below this magnitude are snapped to 0 (motor can't overcome "
        "static friction below it, so it just strains without moving the car) (default 0.08)",
    )
    parser.add_argument(
        "--target-area-frac-lower",
        type=float,
        default=0.05,
        help="Bbox area below this fraction of the frame is too small/far; the car drives forward (default 0.05)",
    )
    parser.add_argument(
        "--target-area-frac-upper",
        type=float,
        default=0.15,
        help="Bbox area above this fraction of the frame is too close; the car backs up (default 0.15)",
    )
    parser.add_argument(
        "--no-follow-distance",
        action="store_true",
        help="Only rotate to correct left/right; don't drive forward/back to hold distance",
    )
    args = parser.parse_args()

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
    # Separate instance for ROI class-picking (see select_target_class) so the
    # persistent tracking loop's ByteTrack state on `model` isn't disturbed.
    detector = YOLO(args.model)

    window = "ESP32-CAM Object Tracker"
    locked_id = None
    driving_enabled = True

    car = None
    if args.car_host:
        car = CarController(args.car_host, args.car_port).start()
        print(f"Car control enabled -> http://{args.car_host}:{args.car_port}/move")
    else:
        print("No --car-host given; running in tracking-only mode (no driving).")

    reader = StreamReader(stream_url).start()

    print("Waiting for first frame...")
    while reader.read() is None:
        time.sleep(0.05)

    try:
        while True:
            frame = reader.read()
            if frame is None:
                continue

            results = model.track(
                frame,
                persist=True,
                tracker="bytetrack.yaml",
                classes=args.classes,
                conf=args.conf,
                device=device,
                verbose=False,
            )
            detections = parse_detections(results)
            box, track_id = select_target(detections, locked_id)
            locked_id = track_id

            if box is not None:
                bbox = xyxy_to_xywh(box)
                apply_bbox(frame, bbox, track_id, args, car, driving_enabled)
            else:
                if car is not None:
                    car.stop_moving()
                if not args.classes:
                    draw_status(frame, "Press 'r' to select an object to track")
                else:
                    draw_status(frame, "TARGET LOST - searching...", (0, 165, 255))

            cv2.imshow(window, frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break
            elif key == ord("r"):
                if car is not None:
                    car.stop_moving()
                new_class = select_target_class(detector, frame, args.conf, device)
                if new_class is None:
                    print("Selection cancelled or no match; keeping current target.")
                else:
                    args.classes = [new_class]
                    locked_id = None
                    driving_enabled = True
                    print(f"Tracking class {new_class} ({model.names[new_class]}); acquiring biggest match...")
            elif key == ord("c"):
                driving_enabled = not driving_enabled
                print(f"Auto-drive {'enabled' if driving_enabled else 'disabled'}")
                if not driving_enabled and car is not None:
                    car.stop_moving()
            elif key == ord("s"):
                driving_enabled = False
                if car is not None:
                    car.stop_moving()
                print("Emergency stop")
    finally:
        if car is not None:
            car.stop()
        reader.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
