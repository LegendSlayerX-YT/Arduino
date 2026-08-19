"""
Live object tracker for an ESP32-CAM running the CameraWebServer sketch,
with optional closed-loop driving of the motor car (see motor_car/motor_car.ino)
to keep the tracked object centered in frame.

Connects to the camera board's MJPEG stream (served on port 81 at /stream),
lets you draw a box around anything in the video, and tracks it frame-to-frame
with OpenCV while overlaying a bounding box, center marker, and pixel offset
from frame center. When --car-host is given, it also drives the car via the
motor board's /move?forward=&left= endpoint to keep the object centered
(rotating in place to correct left/right offset, and optionally driving
forward/back to hold the object's on-screen size).

If the tracked object leaves the frame (or the tracker otherwise loses it),
the tracker doesn't just give up: it remembers the object's appearance (ORB
features + a color histogram, captured when you selected it) and keeps
scanning each incoming frame for a confident match. As soon as the object
reappears anywhere in view, tracking (and driving) resumes automatically -
no need to press 'r' again.

Usage:
    python car_controller.py --host 192.168.1.50
    (find the camera's IP from the ESP32's serial monitor output on boot)

    python car_controller.py --host 192.168.1.50 --car-host 192.168.1.60
    (also drive the car to keep the selected object centered)

Controls:
    r - select (or reselect) the object to track (re-enables driving)
    c - toggle auto-drive on/off without losing the tracked object
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
    blocking UI call on the main thread (e.g. ROI selection) never stalls
    the network read and breaks the connection. Reconnects automatically
    on transient errors."""

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
        self._target = (0.0, 0.0)
        self._sent = (None, None)
        self._lock = threading.Lock()
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self._thread.start()
        return self

    def set_target(self, forward, left):
        with self._lock:
            self._target = (forward, left)

    def stop_moving(self):
        self.set_target(0.0, 0.0)

    def _run(self):
        while self._running:
            with self._lock:
                target = self._target

            if target != (0.0, 0.0) or target != self._sent:
                try:
                    print(f"Car command: forward={target[0]:+.2f} left={target[1]:+.2f}")
                    requests.get(
                        f"{self.base_url}/move",
                        params={"forward": target[0], "left": target[1]},
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


def compute_drive_command(
    bbox,
    frame_shape,
    target_area_frac_lower,
    target_area_frac_upper,
    kp_forward,
    kp_left,
    deadzone_x=0.08,
    max_speed=0.4,
    max_left=0.2,
):
    """Proportional controller that keeps a tracked bbox centered.

    Horizontal offset drives rotation ("left"): the car rotates in place
    toward the side the object drifted to, which pans the camera that way
    and pulls the object back toward center.

    Bounding-box area, as a fraction of the frame area, drives forward/back:
    below target_area_frac_lower the object is too small/far, so the car
    moves forward, proportionally to how far below the threshold it is;
    above target_area_frac_upper the object is too close, so the car backs
    up, proportionally to how far above; in between (the "hold" band) the
    car does neither. Pass both as None to disable forward/back driving
    entirely. This runs independently of horizontal centering, so the car
    keeps closing/holding distance even while dx is already zero.

    Rotation only kicks in once the object drifts outside the horizontal
    dead zone; within it, rotation is zero. Forward/back speed is capped at
    max_speed and rotation ("left") is capped at max_left.
    """
    frame_h, frame_w = frame_shape[:2]
    x, y, w, h = bbox
    cx = x + w / 2
    dx_norm = (cx - frame_w / 2) / (frame_w / 2)

    left = 0.0
    if abs(dx_norm) > deadzone_x:
        left = clamp(-kp_left * dx_norm, -max_left, max_left)

    forward = 0.0
    real_frac = (w * h) / (frame_w * frame_h)
    if target_area_frac_lower is not None and real_frac < target_area_frac_lower:
        error = (target_area_frac_lower - real_frac) / target_area_frac_lower
        forward = clamp(kp_forward * error, 0.0, max_speed)
    elif target_area_frac_upper is not None and real_frac > target_area_frac_upper:
        error = (real_frac - target_area_frac_upper) / target_area_frac_upper
        forward = clamp(-kp_forward * error, -max_speed, 0.0)

    return forward, left


def create_tracker():
    if hasattr(cv2, "TrackerCSRT_create"):
        return cv2.TrackerCSRT_create()
    return cv2.legacy.TrackerCSRT_create()


# Re-detection tuning: how many good ORB matches (after Lowe's ratio test)
# are needed before a candidate region is trusted, and how well its color
# histogram must correlate with the original selection's.
ORB_FEATURES = 500
MIN_GOOD_MATCHES = 10
MIN_INLIER_MATCHES = 8
LOWE_RATIO = 0.75
HIST_CORR_THRESHOLD = 0.5
# While CSRT is actively tracking, its bbox is checked against the stored
# histogram each frame. This is more lenient than HIST_CORR_THRESHOLD since
# minor motion blur/lighting shifts shouldn't be treated as a lost object,
# but it still catches CSRT locking onto background after the object left.
TRACK_VERIFY_HIST_THRESHOLD = 0.35
# Consecutive failed appearance checks allowed before a CSRT-reported box is
# actually discarded. A single bad frame is normal when the object is only
# partially out of view (a smaller/edge-clipped crop naturally correlates
# worse) - only a sustained mismatch means CSRT has locked onto background.
TRACK_LOSS_GRACE_FRAMES = 5
# Fallback re-detection by color alone (used when the object doesn't have
# enough distinctive texture for ORB matching to reliably reacquire it).
COLOR_BACKPROJ_THRESHOLD = 50
MIN_COLOR_BLOB_AREA_RATIO = 0.15


def compute_hsv_hist(frame, bbox):
    x, y, w, h = [int(v) for v in bbox]
    patch = frame[y : y + h, x : x + w]
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [30, 32], [0, 180, 0, 256])
    cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
    return hist


def compute_signature(orb, frame, bbox):
    """Captures what the selected object looks like, so it can be found
    again later even after leaving the frame entirely. Keypoints are kept
    in patch-local coordinates so a homography can later map them onto
    wherever the object reappears in the full frame."""
    x, y, w, h = [int(v) for v in bbox]
    patch_gray = cv2.cvtColor(frame[y : y + h, x : x + w], cv2.COLOR_BGR2GRAY)
    keypoints, descriptors = orb.detectAndCompute(patch_gray, None)
    return {
        "keypoints": keypoints,
        "descriptors": descriptors,
        "hist": compute_hsv_hist(frame, bbox),
        "size": (w, h),
    }


def _find_object_by_features(orb, matcher, frame, signature):
    """Scans the full frame for the object using RANSAC-filtered ORB feature
    matches to compute a homography from the original patch to its new
    location. This rejects outlier/spurious matches instead of letting them
    skew a naive bounding box, giving a precise bbox - but it needs the
    object to have enough distinctive texture to match on."""
    keypoints, descriptors = signature["keypoints"], signature["descriptors"]
    if descriptors is None or len(descriptors) < 4:
        return None

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_kp, frame_desc = orb.detectAndCompute(gray, None)
    if frame_desc is None or len(frame_kp) < 2:
        return None

    matches = matcher.knnMatch(descriptors, frame_desc, k=2)
    good = [m for m, n in ((p[0], p[1]) for p in matches if len(p) == 2) if m.distance < LOWE_RATIO * n.distance]
    if len(good) < MIN_GOOD_MATCHES:
        return None

    src_pts = np.float32([keypoints[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([frame_kp[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    homography, inlier_mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    if homography is None or inlier_mask is None or int(inlier_mask.sum()) < MIN_INLIER_MATCHES:
        return None

    ref_w, ref_h = signature["size"]
    corners = np.float32([[0, 0], [0, ref_h], [ref_w, ref_h], [ref_w, 0]]).reshape(-1, 1, 2)
    projected = cv2.perspectiveTransform(corners, homography)
    x, y, w, h = cv2.boundingRect(projected)

    # A poorly-conditioned homography (few/clustered inlier matches) can
    # still pass the checks above while producing a plausible-but-wrong box
    # size (e.g. half the object). The camera position is fixed, so the
    # object's apparent size shouldn't swing far from its original scale.
    if w < ref_w * 0.6 or h < ref_h * 0.6 or w > ref_w * 1.8 or h > ref_h * 1.8:
        return None

    return _clip_and_verify(frame, (x, y, w, h), signature)


def _find_object_by_color(frame, signature):
    """Fallback re-detection using only the stored color histogram, via
    backprojection. Doesn't need any texture on the object, so it catches
    cases (plain-colored objects) where ORB feature matching can't find
    enough keypoints to reacquire the object."""
    ref_w, ref_h = signature["size"]
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # signature["hist"] is normalized to 0-1 (for compareHist elsewhere), but
    # calcBackProject uses histogram values directly as pixel weights, so it
    # needs its own copy rescaled to 0-255 or COLOR_BACKPROJ_THRESHOLD would
    # never be reachable.
    backproj_hist = cv2.normalize(signature["hist"], None, 0, 255, cv2.NORM_MINMAX)
    backproj = cv2.calcBackProject([hsv], [0, 1], backproj_hist, [0, 180, 0, 256], 1)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    _, mask = cv2.threshold(backproj, COLOR_BACKPROJ_THRESHOLD, 255, cv2.THRESH_BINARY)
    # A single-colored object (e.g. a glossy box) often gets split into
    # several blobs by a highlight or shadow on part of its surface. Close
    # first to bridge those internal gaps back into one blob, then open to
    # drop small unrelated noise specks elsewhere in the frame.
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.dilate(mask, kernel, iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    # Even after closing, a stubborn highlight can leave the object as a
    # couple of separate-but-adjacent blobs. Merge every blob near the
    # largest one (rather than keeping only the single biggest) so the
    # reacquired box covers the whole object instead of one fragment.
    noise_area = ref_w * ref_h * 0.03
    significant = [c for c in contours if cv2.contourArea(c) >= noise_area]
    if not significant:
        return None

    primary = max(significant, key=cv2.contourArea)
    px, py, pw, ph = cv2.boundingRect(primary)
    margin = max(ref_w, ref_h) * 0.5
    region_x0, region_y0 = px - margin, py - margin
    region_x1, region_y1 = px + pw + margin, py + ph + margin

    def near_primary(contour):
        cx, cy, cw, ch = cv2.boundingRect(contour)
        return not (cx + cw < region_x0 or cx > region_x1 or cy + ch < region_y0 or cy > region_y1)

    merged_points = np.vstack([c for c in significant if near_primary(c)])
    x, y, w, h = cv2.boundingRect(merged_points)
    if w * h < ref_w * ref_h * MIN_COLOR_BLOB_AREA_RATIO:
        return None

    return _clip_and_verify(frame, (x, y, w, h), signature)


def _clip_and_verify(frame, bbox, signature):
    """Clips bbox to the frame and confirms its color histogram still
    resembles the stored signature, to reject false-positive candidates."""
    frame_h, frame_w = frame.shape[:2]
    x, y, w, h = bbox
    x, y = max(0, int(x)), max(0, int(y))
    w, h = min(int(w), frame_w - x), min(int(h), frame_h - y)
    if w <= 0 or h <= 0:
        return None

    candidate_hist = compute_hsv_hist(frame, (x, y, w, h))
    correlation = cv2.compareHist(signature["hist"], candidate_hist, cv2.HISTCMP_CORREL)
    if correlation < HIST_CORR_THRESHOLD:
        return None

    return (x, y, w, h)


def find_object(orb, matcher, frame, signature):
    """Scans the full frame for the object described by signature. Tries
    precise feature-based localization first, falling back to color-blob
    detection when the object lacks enough texture to match on."""
    candidate = _find_object_by_features(orb, matcher, frame, signature)
    if candidate is not None:
        return candidate
    return _find_object_by_color(frame, signature)


def verify_tracking(frame, bbox, signature):
    """Sanity-checks that a CSRT-reported bbox still looks like the tracked
    object. CSRT's update() can keep returning success even after the
    object has left the frame, by locking onto whatever background is left
    under the last-known box - this catches that case."""
    frame_h, frame_w = frame.shape[:2]
    x, y, w, h = [int(v) for v in bbox]
    x, y = max(0, x), max(0, y)
    w, h = min(w, frame_w - x), min(h, frame_h - y)
    if w <= 0 or h <= 0:
        return False
    candidate_hist = compute_hsv_hist(frame, (x, y, w, h))
    correlation = cv2.compareHist(signature["hist"], candidate_hist, cv2.HISTCMP_CORREL)
    return correlation >= TRACK_VERIFY_HIST_THRESHOLD


def draw_overlay(frame, bbox, drive_command=None):
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
        f"dx={offset_x} dy={offset_y}",
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 255),
        2,
    )

    if drive_command is not None:
        forward, left = drive_command
        cv2.putText(
            frame,
            f"drive: forward={forward:+.2f} left={left:+.2f}",
            (10, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 200, 0),
            2,
        )


def draw_status(frame, text, color=(0, 255, 255)):
    cv2.putText(frame, text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)


def apply_bbox(frame, bbox, args, car, driving_enabled):
    """Drives the car (if enabled) to keep bbox centered and draws the
    overlay. Shared by every place a valid bbox can come from this frame:
    normal CSRT tracking, a grace-period frame, or a freshly reacquired box
    from find_object."""
    _, _, bw, bh = bbox
    frame_h, frame_w = frame.shape[:2]
    area_pct = 100 * (bw * bh) / (frame_w * frame_h)
    print(f"Object area: {area_pct:.1f}% of frame")

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
                args.deadzone_x,
                args.max_speed,
                args.max_left,
            )
            car.set_target(*drive_command)
        else:
            car.stop_moving()
    draw_overlay(frame, bbox, drive_command)


def main():
    parser = argparse.ArgumentParser(description="ESP32-CAM live object tracker")
    parser.add_argument("--host", required=True, help="ESP32-CAM IP address, e.g. 192.168.1.50")
    parser.add_argument("--port", type=int, default=81, help="Stream server port (default 81)")
    parser.add_argument("--led", type=int, help="Onboard flash LED intensity, 0-255 (AI-Thinker board only)")
    parser.add_argument("--ae-level", type=int, choices=range(-2, 3), help="Auto exposure target level, -2 to 2")
    parser.add_argument("--brightness", type=int, choices=range(-2, 3), help="Brightness offset, -2 to 2")
    parser.add_argument("--saturation", type=int, choices=range(-2, 3), help="Color saturation, -2 to 2")
    parser.add_argument("--contrast", type=int, choices=range(-2, 3), help="Contrast, -2 to 2")
    parser.add_argument(
        "--car-host",
        help="Motor car IP address (the board running motor_car.ino). "
        "If given, the car auto-drives to keep the tracked object centered.",
    )
    parser.add_argument("--car-port", type=int, default=80, help="Motor car HTTP port (default 80)")
    parser.add_argument("--kp-left", type=float, default=0.3, help="Proportional gain for left/right rotation correction")
    parser.add_argument("--kp-forward", type=float, default=0.6, help="Proportional gain for forward/back distance correction")
    parser.add_argument("--deadzone-x", type=float, default=0.15, help="Horizontal dead zone as a fraction of half frame width")
    parser.add_argument("--max-speed", type=float, default=0.4, help="Cap on forward/back drive magnitude, 0-1 (default 0.4)")
    parser.add_argument("--max-left", type=float, default=0.2, help="Cap on rotation (left/right) drive magnitude, 0-1 (default 0.2)")
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

    window = "ESP32-CAM Object Tracker"
    tracker = None
    signature = None
    lost_streak = 0
    driving_enabled = True
    orb = cv2.ORB_create(nfeatures=ORB_FEATURES)
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)

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

            if tracker is not None:
                ok, bbox = tracker.update(frame)
                if ok and verify_tracking(frame, bbox, signature):
                    lost_streak = 0
                    apply_bbox(frame, bbox, args, car, driving_enabled)
                elif ok and lost_streak < TRACK_LOSS_GRACE_FRAMES:
                    # Appearance check failed but CSRT still thinks it has
                    # the object - likely a partial exit/occlusion. Keep
                    # driving/tracking for a few frames rather than dropping
                    # on one bad frame.
                    lost_streak += 1
                    apply_bbox(frame, bbox, args, car, driving_enabled)
                else:
                    tracker = None
                    lost_streak = 0
                    if car is not None:
                        car.stop_moving()
            elif signature is not None:
                candidate = find_object(orb, matcher, frame, signature)
                if candidate is not None:
                    tracker = create_tracker()
                    tracker.init(frame, candidate)
                    apply_bbox(frame, candidate, args, car, driving_enabled)
                else:
                    if car is not None:
                        car.stop_moving()
                    draw_status(frame, "Object out of view - searching...", (0, 165, 255))
            else:
                if car is not None:
                    car.stop_moving()
                draw_status(frame, "Press 'r' to select an object to track")

            cv2.imshow(window, frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break
            elif key == ord("r"):
                lost_streak = 0
                if car is not None:
                    car.stop_moving()
                roi = cv2.selectROI(window, frame, fromCenter=False, showCrosshair=True)
                if roi[2] > 0 and roi[3] > 0:
                    tracker = create_tracker()
                    tracker.init(frame, roi)
                    signature = compute_signature(orb, frame, roi)
                    driving_enabled = True
                else:
                    tracker = None
                    signature = None
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
