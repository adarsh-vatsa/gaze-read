#!/usr/bin/env python3
"""Webcam gaze tracker for the gaze-read skill.

Subcommands:
  calibrate   full-screen dots, fits gaze -> screen mapping, saves calibration.json
  start       starts the background tracker (needs a calibration)
  stop        stops the background tracker
  status      prints whether the tracker runs and the last samples
  run         the tracker loop itself (used by `start`)
  test        grabs a few frames and reports whether a face is found

Data lives in ~/.claude/gaze-read (override with GAZE_READ_HOME).
"""
import argparse
import json
import math
import os
import random
import signal
import subprocess
import sys
import time
from collections import deque
from pathlib import Path

HOME = Path(os.environ.get("GAZE_READ_HOME", Path.home() / ".claude" / "gaze-read"))
MODEL = HOME / "face_landmarker.task"
CALIB = HOME / "calibration.json"
SAMPLES = HOME / "samples.csv"
PIDFILE = HOME / "tracker.pid"
CONFIG = HOME / "config.json"
LOG = HOME / "tracker.log"
SCREEN = HOME / "screen.jsonl"
SHOT = HOME / "window.png"
ANCHORS = HOME / "anchors.jsonl"

DEFAULT_CONFIG = {
    "camera": 0,            # OpenCV camera index (0 is usually the built-in camera)
    "fps": 10,              # frames processed per second by the background tracker
    # window owner names (prefix match) where Claude's replies are read
    "apps": ["Terminal", "iTerm2", "Ghostty", "WezTerm", "Warp", "kitty", "Alacritty",
             "Code", "Cursor", "Claude", "T3 Code"],
    "input_strip_pt": 150,  # bottom strip of the window (prompt box) that does not count as reading
    "wpm": 230,             # reading speed used to turn looking time into a percent
    "lead_s": 10,           # seconds before the reply finished that still count (reading while it streams)
    "keep_s": 1800,         # seconds of samples kept on disk
    "screen_read_s": 1.0,
    "drift_window_s": 1800,  # known-gaze points (clicks, typing) from this many seconds set the drift correction
    "drift_min_points": 3,
    "geo_window_s": 14400,  # known points from the last 4 h train the geometric model
    "geo_model": False,     # experimental geometric model (see README); off by default   # how often the reply window is captured and its text located
}


def load_config():
    cfg = dict(DEFAULT_CONFIG)
    if CONFIG.exists():
        cfg.update(json.loads(CONFIG.read_text()))
    else:
        HOME.mkdir(parents=True, exist_ok=True)
        CONFIG.write_text(json.dumps(DEFAULT_CONFIG, indent=2))
    return cfg


# ---------------------------------------------------------------- screen / windows

def screen_size():
    from Quartz import CGDisplayBounds, CGMainDisplayID
    b = CGDisplayBounds(CGMainDisplayID())
    return float(b.size.width), float(b.size.height)


def front_window(apps, front_only=True):
    """(x, y, w, h, window_id) of the frontmost normal window if it belongs to one of `apps`.
    With front_only=False, the first such window anywhere in the stack."""
    from Quartz import (CGWindowListCopyWindowInfo, kCGNullWindowID,
                        kCGWindowListExcludeDesktopElements, kCGWindowListOptionOnScreenOnly)
    wins = CGWindowListCopyWindowInfo(
        kCGWindowListOptionOnScreenOnly | kCGWindowListExcludeDesktopElements, kCGNullWindowID)
    for w in wins:  # ordered front to back
        if w.get("kCGWindowLayer", 1) != 0:
            continue
        b = w["kCGWindowBounds"]
        if b["Height"] < 200 or b["Width"] < 200:
            continue
        owner = w.get("kCGWindowOwnerName", "") or ""
        if any(owner.startswith(a) for a in apps):
            return (b["X"], b["Y"], b["Width"], b["Height"], int(w["kCGWindowNumber"]))
        if front_only:
            return None
    return None


# ---------------------------------------------------------------- screen text

def ocr_image(path):
    """Text lines in an image with macOS Vision: [(text, x, y, w, h)] normalised, origin top-left."""
    import objc
    import Vision
    from Foundation import NSURL
    with objc.autorelease_pool():
        handler = Vision.VNImageRequestHandler.alloc().initWithURL_options_(NSURL.fileURLWithPath_(str(path)), None)
        req = Vision.VNRecognizeTextRequest.alloc().init()
        req.setRecognitionLevel_(Vision.VNRequestTextRecognitionLevelAccurate)
        req.setUsesLanguageCorrection_(False)
        handler.performRequests_error_([req], None)
        out = []
        for obs in req.results() or []:
            cand = obs.topCandidates_(1)
            if not cand:
                continue
            bb = obs.boundingBox()
            out.append((str(cand[0].string()), bb.origin.x, 1.0 - bb.origin.y - bb.size.height,
                        bb.size.width, bb.size.height))
        return out


def capture_window(wid, path):
    r = subprocess.run(["screencapture", "-x", "-o", "-l", str(wid), "-t", "png", str(path)],
                       capture_output=True, timeout=10)
    return r.returncode == 0 and Path(path).exists()


def read_window_text(win, path=SHOT):
    """Capture a window and return its text lines in screen points: [[text, x, y, w, h]]."""
    x, y, w, h, wid = win
    if not capture_window(wid, path):
        return None
    lines = []
    for text, bx, by, bw, bh in ocr_image(path):
        lines.append([text, round(x + bx * w), round(y + by * h), round(bw * w), round(bh * h)])
    return lines


def write_snapshot(t, win, lines, kind):
    with open(SCREEN, "a") as fh:
        fh.write(json.dumps({"t": round(t, 2), "win": list(win[:4]), "lines": lines, "kind": kind}) + "\n")


def scroll_shift(prev, cur, top, bottom):
    """Vertical shift in pixels of the text body between two captures (negative: text moved up)."""
    import cv2
    import numpy as np
    h, w = cur.shape
    a = prev[top:bottom, int(w * 0.3):].astype(np.float32)
    c = cur[top:bottom, int(w * 0.3):].astype(np.float32)
    if a.shape != c.shape or a.size == 0:
        return None
    a, c = cv2.resize(a, None, fx=0.5, fy=0.5), cv2.resize(c, None, fx=0.5, fy=0.5)
    win = cv2.createHanningWindow((a.shape[1], a.shape[0]), cv2.CV_32F)
    (sx, sy), resp = cv2.phaseCorrelate(a, c, win)
    if resp < 0.15 or abs(sx) > 2:
        return None
    return sy * 2.0


def dictation_active():
    """True while Superwhisper shows a window bigger than its menu bar icon (its recording overlay)."""
    from Quartz import CGWindowListCopyWindowInfo, kCGNullWindowID, kCGWindowListOptionOnScreenOnly
    for w in CGWindowListCopyWindowInfo(kCGWindowListOptionOnScreenOnly, kCGNullWindowID):
        b = w["kCGWindowBounds"]
        if (str(w.get("kCGWindowOwnerName", "")).startswith("Superwhisper")
                and w.get("kCGWindowLayer") != 25        # 25 = menu bar icon
                and b["Height"] > 50 and b["Width"] > 50):
            return True
    return False


def screen_reader(cfg, stop, events):
    """Background thread. Records where each text line of the front reply window sits.

    - every `screen_read_s`, and every 0.25 s while the user scrolls, captures the window;
    - on a scroll, moves the known lines by the measured image shift at once ("shift" snapshot),
      and reads the text again with OCR when the scrolling stops ("ocr" snapshot);
    - when the prompt box changes with no key press during or right after dictation, reports a
      known-gaze point (the user looks at the text appearing in the box) through `events`."""
    import cv2
    import numpy as np
    from Quartz import (CGEventSourceSecondsSinceLastEventType, kCGEventKeyDown, kCGEventScrollWheel,
                        kCGEventSourceStateCombinedSessionState as SRC)
    strip = float(cfg["input_strip_pt"])
    period = float(cfg["screen_read_s"])
    prev = None  # {"img", "win", "lines"}
    last_cap, last_dict, pending_ocr, warned = 0.0, 0.0, False, False
    while not stop["flag"]:
        now = time.time()
        try:
            if dictation_active():
                if now - last_dict > 5:
                    print(f"screen_reader: dictation seen at {time.strftime('%H:%M:%S')}", file=sys.stderr, flush=True)
                last_dict = now
            scrolling = CGEventSourceSecondsSinceLastEventType(SRC, kCGEventScrollWheel) < 0.3
            due = (now - last_cap >= period or (scrolling and now - last_cap >= 0.25)
                   or (pending_ocr and not scrolling and now - last_cap >= 0.25))
            if not due:
                time.sleep(0.05)
                continue
            win = front_window(cfg["apps"])
            if not win:
                time.sleep(0.2)
                continue
            last_cap = now
            if not capture_window(win[4], SHOT):
                if not warned:
                    print("screen_reader: window capture failed (Screen Recording permission?)",
                          file=sys.stderr, flush=True)
                    warned = True
                continue
            img = cv2.imread(str(SHOT), cv2.IMREAD_GRAYSCALE)
            x, y, w, h, _ = win
            scale = img.shape[1] / w
            same = prev is not None and prev["win"] == win[:4] and prev["img"].shape == img.shape
            if same:
                small = lambda im: cv2.resize(im, (192, int(192 * im.shape[0] / im.shape[1]))).astype(np.int16)
                changed = float(np.abs(small(img) - small(prev["img"])).mean()) > 1.0
                if not changed and not pending_ocr:
                    continue
                box_top = int((h - strip) * scale)
                body_top = int(h * 0.08 * scale)
                box_changed = float(np.abs(img[box_top:].astype(np.int16) - prev["img"][box_top:]).mean()) > 1.5
                key_age = CGEventSourceSecondsSinceLastEventType(SRC, kCGEventKeyDown)
                if box_changed and key_age > 1.5 and now - last_dict < 10:
                    events.append((now, "dictation", y + h - strip / 2))
                if changed and scrolling and prev["lines"]:
                    sy = scroll_shift(prev["img"], img, body_top, box_top)
                    if sy is not None and abs(sy) >= 2:
                        dy = sy / scale
                        lines = [[t, lx, round(ly + dy), lw, lh] for t, lx, ly, lw, lh in prev["lines"]
                                 if y + h * 0.08 <= ly + dy and ly + dy + lh <= y + h - strip]
                        write_snapshot(now, win, lines, "shift")
                        prev["lines"] = lines
                if scrolling:
                    prev["img"] = img
                    pending_ocr = True
                    continue
            lines = [[t, round(x + bx * w), round(y + by * h), round(bw * w), round(bh * h)]
                     for t, bx, by, bw, bh in ocr_image(SHOT)]
            write_snapshot(now, win, lines, "ocr")
            prev = {"img": img, "win": win[:4], "lines": lines}
            pending_ocr = False
        except Exception as e:  # keep tracking gaze even if a capture fails
            print(f"screen_reader: {e!r}", file=sys.stderr, flush=True)
            time.sleep(0.5)


# ---------------------------------------------------------------- face features

# MediaPipe face mesh indices: eye corners, eyelids, iris centres.
EYE_A = {"corners": (33, 133), "lids": (159, 145)}
EYE_B = {"corners": (362, 263), "lids": (386, 374)}
IRIS_CENTRES = (468, 473)


class FaceFeatures:
    """Turns a camera frame into a small feature vector: iris position inside each eye + head pose."""

    def __init__(self):
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions, vision
        if not MODEL.exists():
            sys.exit(f"Face model missing: {MODEL}. Run setup.sh first.")
        opts = vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(MODEL)),
            running_mode=vision.RunningMode.VIDEO,
            num_faces=1,
            output_facial_transformation_matrixes=True,
        )
        self.mp = mp
        self.landmarker = vision.FaceLandmarker.create_from_options(opts)
        self.last_ts = 0
        self.obs = None  # geometric observation of the last frame (see GeoGaze)

    def __call__(self, frame_bgr):
        import cv2
        import numpy as np
        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        img = self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=rgb)
        ts = max(self.last_ts + 1, int(time.monotonic() * 1000))
        self.last_ts = ts
        res = self.landmarker.detect_for_video(img, ts)
        if not res.face_landmarks:
            return None
        lms = res.face_landmarks[0]
        if len(lms) < 478:
            return None
        pts = np.array([[p.x * w, p.y * h] for p in lms])

        feats = []
        irises = [pts[i] for i in IRIS_CENTRES]
        for eye in (EYE_A, EYE_B):
            a, b = pts[eye["corners"][0]], pts[eye["corners"][1]]
            mid = (a + b) / 2
            iris = min(irises, key=lambda p: np.linalg.norm(p - mid))  # robust to index order
            axis = b - a
            L = np.linalg.norm(axis)
            if L < 1e-3:
                return None
            ux = axis / L
            uy = np.array([-ux[1], ux[0]])
            d = iris - mid
            top, bot = pts[eye["lids"][0]], pts[eye["lids"][1]]
            feats += [d @ ux / L, d @ uy / L, (bot - top) @ uy / L]

        M = np.array(res.facial_transformation_matrixes[0])
        R = M[:3, :3]
        yaw = math.atan2(-R[2, 0], math.hypot(R[0, 0], R[1, 0]))
        pitch = math.atan2(R[2, 1], R[2, 2])
        roll = math.atan2(R[1, 0], R[0, 0])
        feats += [yaw, pitch, roll, M[0, 3], M[1, 3], M[2, 3]]
        eye_mid = (pts[33] + pts[133] + pts[362] + pts[263]) / 4.0
        self.obs = [(feats[0] + feats[3]) / 2, (feats[1] + feats[4]) / 2,
                    eye_mid[0] / w, eye_mid[1] / h, w / h, float(-M[2, 3])] + [float(v) for v in R.flatten()]
        return np.array(feats, dtype=float)


def open_camera(cfg):
    import cv2
    cap = cv2.VideoCapture(int(cfg["camera"]))
    if not cap.isOpened():
        sys.exit(f"Cannot open camera {cfg['camera']}. Check camera permission or set 'camera' in {CONFIG}.")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    return cap


# ---------------------------------------------------------------- regression

def _design(F, mu, sd):
    import numpy as np
    Z = (F - mu) / sd
    return np.hstack([np.ones((len(Z), 1)), Z])


def fit_ridge(F, Y, lam):
    import numpy as np
    mu, sd = F.mean(0), F.std(0) + 1e-6
    X = _design(F, mu, sd)
    I = np.eye(X.shape[1])
    I[0, 0] = 0
    W = np.linalg.solve(X.T @ X + lam * len(X) * I, X.T @ Y)
    return {"mu": mu.tolist(), "sd": sd.tolist(), "W": W.tolist()}


def predict(model, f):
    import numpy as np
    X = _design(np.atleast_2d(f), np.array(model["mu"]), np.array(model["sd"]))
    return (X @ np.array(model["W"]))[0]


def fit_with_cv(F, Y, groups):
    """Pick the ridge strength by leave-one-dot-out error, then fit on all data."""
    import numpy as np
    best = None
    for lam in (0.001, 0.01, 0.1, 1.0, 10.0):
        errs = []
        for g in np.unique(groups):
            tr, te = groups != g, groups == g
            m = fit_ridge(F[tr], Y[tr], lam)
            p = np.array([predict(m, f) for f in F[te]]).mean(0)
            errs.append(np.linalg.norm(p - Y[te][0]))
        e = float(np.mean(errs))
        if best is None or e < best[1]:
            best = (lam, e)
    return fit_ridge(F, Y, best[0]), best


# ---------------------------------------------------------------- calibration

def calibrate(cfg):
    import cv2
    import numpy as np

    sw, sh = screen_size()
    W, H = int(sw), int(sh)
    cap = open_camera(cfg)
    ff = FaceFeatures()
    win = "gaze-calibration"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(win, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    def frame_canvas():
        return np.full((H, W, 3), 30, np.uint8)

    def text(img, s, y, scale=1.0):
        size = cv2.getTextSize(s, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)[0]
        cv2.putText(img, s, ((W - size[0]) // 2, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (230, 230, 230), 2, cv2.LINE_AA)

    # Intro: wait for space, show whether a face is found.
    while True:
        ok, frame = cap.read()
        f = ff(frame) if ok else None
        img = frame_canvas()
        text(img, "Gaze calibration", H // 2 - 80, 1.4)
        text(img, "Sit as you normally read. Follow each dot with your eyes.", H // 2 - 20)
        text(img, "Face found" if f is not None else "No face found - check light and camera", H // 2 + 30, 0.9)
        text(img, "Press SPACE to start, ESC to cancel", H // 2 + 90, 0.9)
        cv2.imshow(win, img)
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            cv2.destroyAllWindows()
            sys.exit("Calibration cancelled.")
        if k == 32:
            break

    def show_dot(fx, fy, settle=0.9, collect=1.3):
        x, y = int(fx * W), int(fy * H)
        feats = []
        t0 = time.monotonic()
        while True:
            el = time.monotonic() - t0
            if el > settle + collect:
                break
            ok, frame = cap.read()
            f = ff(frame) if ok else None
            if el > settle and f is not None:
                feats.append(f)
            img = frame_canvas()
            r = int(22 - 14 * min(1.0, el / settle))
            cv2.circle(img, (x, y), r, (60, 140, 255), -1, cv2.LINE_AA)
            cv2.circle(img, (x, y), 3, (255, 255, 255), -1, cv2.LINE_AA)
            cv2.imshow(win, img)
            if (cv2.waitKey(1) & 0xFF) == 27:
                cv2.destroyAllWindows()
                sys.exit("Calibration cancelled.")
        return feats, (x / W * sw, y / H * sh)

    grid = [(x, y) for x in (0.05, 0.5, 0.95) for y in (0.06, 0.5, 0.94)]
    grid += [(x, y) for x in (0.27, 0.73) for y in (0.28, 0.72)]
    grid += [(0.27, 0.06), (0.73, 0.06), (0.27, 0.94), (0.73, 0.94)]
    random.shuffle(grid)

    F, Y, G = [], [], []
    for i, (fx, fy) in enumerate(grid):
        feats, target = show_dot(fx, fy)
        if len(feats) < 4:
            continue
        # drop the frames furthest from the median (blinks, glances)
        arr = np.array(feats)
        dist = np.linalg.norm((arr - np.median(arr, 0)) / (arr.std(0) + 1e-6), axis=1)
        keep = arr[dist <= np.percentile(dist, 80)]
        F += list(keep)
        Y += [target] * len(keep)
        G += [i] * len(keep)

    if len(set(G)) < 8:
        cv2.destroyAllWindows()
        sys.exit("Too few dots had a visible face. Improve the light and try again.")

    F, Y, G = np.array(F), np.array(Y), np.array(G)
    model, (lam, cv_err) = fit_with_cv(F, Y, G)

    # Validation on fresh points.
    val_err = []
    for fx, fy in [(random.uniform(0.15, 0.85), random.uniform(0.15, 0.85)) for _ in range(4)]:
        feats, target = show_dot(fx, fy)
        if feats:
            p = np.mean([predict(model, f) for f in feats], 0)
            val_err.append(float(np.linalg.norm(p - np.array(target))))

    cap.release()
    img = frame_canvas()
    v = f"{np.mean(val_err):.0f}" if val_err else "n/a"
    text(img, f"Done. Average error about {v} points on a {W}x{H} screen.", H // 2)
    text(img, "Press any key", H // 2 + 60, 0.9)
    cv2.imshow(win, img)
    cv2.waitKey(2500)
    cv2.destroyAllWindows()
    for _ in range(5):
        cv2.waitKey(1)

    model.update({
        "screen": [sw, sh], "lambda": lam, "cv_error_pt": cv_err,
        "validation_error_pt": float(np.mean(val_err)) if val_err else None,
        "dots": int(len(set(G.tolist()))), "created": time.strftime("%Y-%m-%d %H:%M"),
    })
    CALIB.write_text(json.dumps(model, indent=1))
    print(json.dumps({k: model[k] for k in ("dots", "lambda", "cv_error_pt", "validation_error_pt", "screen")}))
    print(f"Saved {CALIB}")


# ---------------------------------------------------------------- geometric gaze model

class GeoGaze:
    """Gaze as a 3D ray from the eyes, intersected with the screen plane.

    MediaPipe gives the head pose in centimetres in camera space (camera at the origin, looking
    down -z, y up). The built-in camera sits in the lid above the screen, so the screen lies in
    the camera's z = 0 plane, below the camera. Only the mapping from iris offset to eye angle
    (a0, a1, b0, b1) and a small screen placement correction (cx, cy, sx, sy) are personal; they are
    fitted on the fly from known gaze points. Head movement is handled by the geometry, so the
    model does not need to have seen a posture before.

    Observation vector (from FaceFeatures.obs):
      [iris_x, iris_y, eye_u, eye_v, aspect, depth_cm, R00..R22]"""

    TAN_V = math.tan(math.radians(63.0 / 2))  # MediaPipe's default vertical field of view

    def __init__(self, screen_pt, screen_cm):
        self.sw, self.sh = screen_pt
        self.cm_per_pt = screen_cm[0] / screen_pt[0]
        self.p = None
        self.n = 0
        self.err = None
        self.held = None

    def prior(self):
        k = 1.0 / self.cm_per_pt
        return {"cx": self.sw / 2, "cy": -0.6 * k, "sx": -k, "sy": -k}

    def _predict(self, p, O):
        import numpy as np
        a0, a1, b0, b1, cx, cy, sx, sy = p
        D = O[:, 5]
        E = np.stack([(2 * O[:, 2] - 1) * self.TAN_V * O[:, 4] * D,
                      (1 - 2 * O[:, 3]) * self.TAN_V * D, -D], 1)
        gh = np.stack([a0 + a1 * O[:, 0], b0 + b1 * O[:, 1], np.ones(len(O))], 1)
        R = O[:, 6:15].reshape(-1, 3, 3)
        g = np.einsum("nij,nj->ni", R, gh)
        gz = np.where(np.abs(g[:, 2]) < 1e-3, 1e-3, g[:, 2])
        s = -E[:, 2] / gz
        P = E + s[:, None] * g
        return np.stack([cx + sx * P[:, 0], cy + sy * P[:, 1]], 1)

    def predict(self, obs):
        import numpy as np
        if self.p is None or obs is None:
            return None
        return self._predict(self.p, np.array([obs], float))[0]

    def fit(self, O, T, has_x):
        """O: (n,15) observations, T: (n,2) target points (x may be nan), has_x: (n,) bool."""
        import numpy as np
        from scipy.optimize import least_squares
        pr = self.prior()
        prior = np.array([pr["cx"], pr["cy"], pr["sx"], pr["sy"]])
        prior_w = np.array([1 / 300, 1 / 300, 3 / abs(pr["sx"]), 3 / abs(pr["sy"])])  # soft pull to the physical layout

        def resid(p):
            P = self._predict(p, O)
            rx = np.where(has_x, P[:, 0] - np.nan_to_num(T[:, 0]), 0.0)
            ry = P[:, 1] - T[:, 1]
            reg = (p[4:] - prior) * prior_w * 100.0
            return np.concatenate([rx, ry, reg])

        best = None
        for sa in (1, -1):
            for sb in (1, -1):
                p0 = [0.0, 3.0 * sa, 0.0, 3.0 * sb, pr["cx"], pr["cy"], pr["sx"], pr["sy"]]
                try:
                    r = least_squares(resid, p0, loss="soft_l1", f_scale=120.0, max_nfev=400)
                except Exception:
                    continue
                if best is None or r.cost < best.cost:
                    best = r
        if best is None:
            return
        self.p = best.x
        P = self._predict(self.p, O)
        e = np.where(has_x, np.hypot(P[:, 0] - np.nan_to_num(T[:, 0]), P[:, 1] - T[:, 1]), np.abs(P[:, 1] - T[:, 1]))
        self.err = float(np.median(e))
        self.n = len(O)


# ---------------------------------------------------------------- drift correction

POSE_SCALE = [0.12, 0.12, 0.15, 6.0, 6.0, 8.0]  # yaw, pitch, roll (rad); x, y, z (cm): one "step" of posture


class Drift:
    """Keeps the calibration lined up when the user sits differently.

    Known gaze points: at a mouse click the user looks at the cursor (x and y known); while typing or
    dictating into the T3 Code / Claude prompt box they look at the box (only y known). For each point
    the head pose is stored. The correction is a ridge fit of (known - raw) on head pose, so it learns
    how the error changes with posture. A pose far from the calibration and from every known point is
    flagged as uncertain."""

    def __init__(self, cfg, screen, model):
        from Quartz import kCGEventSourceStateCombinedSessionState
        self.src = kCGEventSourceStateCombinedSessionState
        self.window_s = float(cfg["drift_window_s"])
        self.min_pts = int(cfg["drift_min_points"])
        self.strip = float(cfg["input_strip_pt"])
        self.sw, self.sh = screen
        self.calib_pose = list(model["mu"][6:12])
        self.pts = []  # (t, kind, dx or None, dy, z)
        self.geo_pts = []  # (t, obs, target_x or None, target_y)
        self.last_poll = time.time()
        self.last_key_pt = 0.0
        self.fx = self.fy = None
        from Quartz import CGDisplayScreenSize, CGMainDisplayID
        mm = CGDisplayScreenSize(CGMainDisplayID())
        self.geo = GeoGaze(screen, (mm.width / 10.0, mm.height / 10.0))
        self.geo_window_s = float(cfg.get("geo_window_s", 4 * 3600))
        self.geo_busy, self.geo_fitted_n = False, 0
        self.geo_enabled = bool(cfg.get("geo_model", False))
        if ANCHORS.exists():  # survive a restart
            now = time.time()
            for l in ANCHORS.read_text().splitlines():
                a = json.loads(l)
                if a["t"] >= now - self.window_s and "z" in a:
                    self.pts.append((a["t"], a["kind"], a["dx"], a["dy"], a["z"]))
                if a["t"] >= now - self.geo_window_s and a.get("o"):
                    self.geo_pts.append((a["t"], a["o"], a["target"][0], a["target"][1]))
            self._fit()
            self._fit_geo(background=False)

    def z(self, f):
        return [(f[6 + i] - self.calib_pose[i]) / POSE_SCALE[i] for i in range(6)]

    def add(self, t, kind, target, raw, f, obs=None):
        dx = None if target[0] is None else target[0] - raw[0]
        dy = target[1] - raw[1]
        if (dx is not None and abs(dx) > 2500) or abs(dy) > 2500:
            return  # too far off to be a real look
        z = [round(v, 3) for v in self.z(f)]
        self.pts.append((t, kind, dx, dy, z))
        tx = None if target[0] is None else round(float(target[0]), 1)
        ty = round(float(target[1]), 1)
        o = [round(float(v), 5) for v in obs] if obs is not None else None
        if o:
            self.geo_pts.append((t, o, tx, ty))
        with open(ANCHORS, "a") as fh:
            fh.write(json.dumps({"t": round(t, 2), "kind": kind, "dx": None if dx is None else round(dx),
                                 "dy": round(dy), "z": z, "target": [tx, ty], "o": o}) + "\n")
        self._fit()
        if len(self.geo_pts) - self.geo_fitted_n >= 3:
            self._fit_geo()

    def _fit_geo(self, background=True):
        """Refit the geometric model on recent known points (needs 10, with 4 clicks)."""
        import numpy as np
        cutoff = time.time() - self.geo_window_s
        self.geo_pts = [g for g in self.geo_pts if g[0] >= cutoff][-600:]
        pts = list(self.geo_pts)
        if not self.geo_enabled or self.geo_busy or len(pts) < 10 or sum(1 for g in pts if g[2] is not None) < 4:
            return
        O = np.array([g[1] for g in pts], float)
        T = np.array([[np.nan if g[2] is None else g[2], g[3]] for g in pts], float)
        has_x = np.array([g[2] is not None for g in pts])

        def work():
            try:
                # held-out check first: train on the earlier 70%, test on the later 30%
                k = int(len(O) * 0.7)
                held = None
                if len(O) >= 15 and has_x[:k].sum() >= 4:
                    chk = GeoGaze((self.sw, self.sh), (self.geo.cm_per_pt * self.sw, 0))
                    chk.fit(O[:k], T[:k], has_x[:k])
                    if chk.p is not None:
                        P = chk._predict(chk.p, O[k:])
                        e = np.where(has_x[k:], np.hypot(P[:, 0] - np.nan_to_num(T[k:, 0]), P[:, 1] - T[k:, 1]),
                                     np.abs(P[:, 1] - T[k:, 1]))
                        held = float(np.median(e))
                new = GeoGaze((self.sw, self.sh), (self.geo.cm_per_pt * self.sw, 0))
                new.fit(O, T, has_x)
                if new.p is not None:
                    new.held = held
                    self.geo = new
                    print(f"geo fit: {new.n} points, train error {new.err:.0f} pt, "
                          f"held-out error {held if held is None else round(held)} pt", file=sys.stderr, flush=True)
            finally:
                self.geo_busy = False
        self.geo_busy, self.geo_fitted_n = True, len(pts)
        if background:
            import threading
            threading.Thread(target=work, daemon=True).start()
        else:
            work()

    def use_geo(self):
        held = getattr(self.geo, "held", None)
        return (self.geo_enabled and self.geo.p is not None and held is not None
                and held < 100)  # must pass a held-out test

    @staticmethod
    def _ridge(Z, r, lam=0.05):
        import numpy as np
        X = np.hstack([np.ones((len(Z), 1)), Z])
        I = np.eye(X.shape[1]) * lam * len(Z)
        I[0, 0] = 0
        return np.linalg.solve(X.T @ X + I, X.T @ r)

    def _fit_axis(self, Z, r):
        import numpy as np
        if len(r) < self.min_pts:
            return None
        if len(r) < 8:  # too few for posture terms: plain median shift
            return np.array([np.median(r)] + [0.0] * 6)
        w = self._ridge(Z, r)
        res = r - np.hstack([np.ones((len(Z), 1)), Z]) @ w
        mad = np.median(np.abs(res - np.median(res))) + 1e-6
        keep = np.abs(res) <= max(150.0, 3 * 1.4826 * mad)  # drop looks that were elsewhere
        return self._ridge(Z[keep], r[keep]) if keep.sum() >= 8 else w

    def _fit(self):
        import numpy as np
        cutoff = time.time() - self.window_s
        self.pts = [p for p in self.pts if p[0] >= cutoff][-400:]
        px = [p for p in self.pts if p[2] is not None]
        self.fx = self._fit_axis(np.array([p[4] for p in px]).reshape(-1, 6), np.array([p[2] for p in px], float))
        self.fy = self._fit_axis(np.array([p[4] for p in self.pts]).reshape(-1, 6),
                                 np.array([p[3] for p in self.pts], float))

    def observe(self, t, raw, f, win, events, obs=None):
        """Call once per frame with the raw (uncorrected) gaze estimate and features, or None when no face."""
        from Quartz import (CGEventCreate, CGEventGetLocation, CGEventSourceSecondsSinceLastEventType,
                            kCGEventKeyDown, kCGEventLeftMouseDown)
        since = t - self.last_poll
        self.last_poll = t
        pending = []
        while events:
            pending.append(events.pop(0))
        if raw is None:
            return
        for et, kind, ty in pending:
            if t - et < 1.0:
                self.add(t, kind, (None, ty), raw, f, obs)
        if CGEventSourceSecondsSinceLastEventType(self.src, kCGEventLeftMouseDown) <= since:
            m = CGEventGetLocation(CGEventCreate(None))
            if 0 <= m.x <= self.sw and 0 <= m.y <= self.sh:
                self.add(t, "click", (m.x, m.y), raw, f, obs)
                return
        if win and CGEventSourceSecondsSinceLastEventType(self.src, kCGEventKeyDown) <= since \
                and t - self.last_key_pt > 2.0:
            x, y, w, h, _ = win
            if y + h <= self.sh:
                self.last_key_pt = t
                self.add(t, "typing", (None, y + h - self.strip / 2), raw, f, obs)

    def apply(self, g, f):
        z = [1.0] + self.z(f)
        dx = sum(a * b for a, b in zip(self.fx, z)) if self.fx is not None else 0.0
        dy = sum(a * b for a, b in zip(self.fy, z)) if self.fy is not None else 0.0
        return (g[0] + dx, g[1] + dy)

    def known_pose(self, f):
        """False when the head pose is far from the calibration and from every recent known gaze point."""
        z = self.z(f)
        if max(abs(v) for v in z) <= 1.0:
            return True
        return any(max(abs(a - b) for a, b in zip(z, p[4])) <= 1.0 for p in self.pts)

    def summary(self):
        n = len(self.pts)
        clicks = sum(1 for p in self.pts if p[1] == "click")
        dict_n = sum(1 for p in self.pts if p[1] == "dictation")
        base = [0.0] * 6
        cx = self.fx[0] if self.fx is not None else 0.0
        cy = self.fy[0] if self.fy is not None else 0.0
        return n, clicks, dict_n, cx, cy


# ---------------------------------------------------------------- background tracker

def running_pid():
    try:
        pid = int(PIDFILE.read_text().strip())
        os.kill(pid, 0)
        return pid
    except (OSError, ValueError):
        return None


def trim_samples(keep_s):
    cutoff = time.time() - keep_s
    if SAMPLES.exists():
        lines = SAMPLES.read_text().splitlines(True)
        kept = [l for l in lines if l and float(l.split(",", 1)[0]) >= cutoff]
        tmp = SAMPLES.with_suffix(".tmp")
        tmp.write_text("".join(kept))
        tmp.replace(SAMPLES)
    if ANCHORS.exists():
        keep = [l for l in ANCHORS.read_text().splitlines(True)
                if json.loads(l)["t"] >= time.time() - float(DEFAULT_CONFIG["geo_window_s"])]
        ANCHORS.write_text("".join(keep))
    if SCREEN.exists():
        recs = SCREEN.read_text().splitlines(True)
        idx = [i for i, l in enumerate(recs) if json.loads(l)["t"] >= cutoff]
        start = max(0, (idx[0] if idx else len(recs)) - 1)  # keep the snapshot active at the cutoff
        tmp = SCREEN.with_suffix(".tmp")
        tmp.write_text("".join(recs[start:]))
        tmp.replace(SCREEN)


def run(cfg):
    import numpy as np
    if not CALIB.exists():
        sys.exit("No calibration. Run: gaze calibrate")
    model = json.loads(CALIB.read_text())
    PIDFILE.write_text(str(os.getpid()))
    stop = {"flag": False}
    signal.signal(signal.SIGTERM, lambda *_: stop.update(flag=True))
    signal.signal(signal.SIGINT, lambda *_: stop.update(flag=True))

    cap = open_camera(cfg)
    ff = FaceFeatures()
    import threading
    events = []  # known-gaze events from the screen reader (dictation into the prompt box)
    threading.Thread(target=screen_reader, args=(cfg, stop, events), daemon=True).start()
    period = 1.0 / float(cfg["fps"])
    strip = float(cfg["input_strip_pt"])
    hist = deque(maxlen=5)
    drift = Drift(cfg, screen_size(), model)
    win, win_t, trim_t = None, 0.0, time.time()
    out = open(SAMPLES, "a", buffering=1)
    try:
        while not stop["flag"]:
            t0 = time.time()
            ok, frame = cap.read()
            if t0 - win_t > 0.5:
                win, win_t = front_window(cfg["apps"]), t0
            f = ff(frame) if ok else None
            raw = predict(model, f) if f is not None else None
            obs = ff.obs if f is not None else None
            drift.observe(t0, raw, f, win, events, obs)
            if f is None:
                hist.clear()
                out.write(f"{t0:.2f},,,0,0\n")
            else:
                geo = drift.geo.predict(obs) if drift.use_geo() else None
                if geo is not None and np.all(np.isfinite(geo)):
                    hist.append(np.array(geo))
                    gx, gy = np.median(np.array(hist), 0)
                    known = 1  # the geometry covers head movement
                else:
                    hist.append(raw)
                    gx, gy = drift.apply(np.median(np.array(hist), 0), f)
                    known = int(drift.known_pose(f))
                inside = 0
                if win:
                    x, y, w, h, _ = win
                    inside = int(x <= gx <= x + w and y <= gy <= y + h - strip)
                out.write(f"{t0:.2f},{gx:.0f},{gy:.0f},{inside},{known}\n")
            if t0 - trim_t > 600:
                out.close()
                trim_samples(cfg["keep_s"])
                out = open(SAMPLES, "a", buffering=1)
                trim_t = t0
            dt = period - (time.time() - t0)
            if dt > 0:
                time.sleep(dt)
    finally:
        out.close()
        cap.release()
        if PIDFILE.exists() and PIDFILE.read_text().strip() == str(os.getpid()):
            PIDFILE.unlink()


def start(cfg):
    if running_pid():
        print(f"Tracker already running (pid {running_pid()}).")
        return
    if not CALIB.exists():
        sys.exit("No calibration. Run: gaze calibrate")
    log = open(LOG, "a")
    p = subprocess.Popen([sys.executable, os.path.abspath(__file__), "run"],
                         stdout=log, stderr=log, stdin=subprocess.DEVNULL, start_new_session=True)
    time.sleep(3)
    if p.poll() is not None:
        sys.exit(f"Tracker exited at start. See {LOG}")
    print(f"Tracker running (pid {p.pid}).")


def stop_tracker():
    pid = running_pid()
    if not pid:
        print("Tracker not running.")
        return
    os.kill(pid, signal.SIGTERM)
    for _ in range(30):
        if not running_pid():
            break
        time.sleep(0.1)
    print("Tracker stopped.")


def status(cfg):
    pid = running_pid()
    print(f"Tracker: {'running, pid ' + str(pid) if pid else 'not running'}")
    if CALIB.exists():
        c = json.loads(CALIB.read_text())
        print(f"Calibration: {c.get('created')}, {c.get('dots')} dots, "
              f"validation error {c.get('validation_error_pt') or 0:.0f} pt")
    else:
        print("Calibration: none")
    if ANCHORS.exists() and CALIB.exists():
        d = Drift(cfg, screen_size(), json.loads(CALIB.read_text()))
        n, clicks, dict_n, cx, cy = d.summary()
        print(f"Drift correction: {n} known gaze points in the last {int(cfg['drift_window_s'] // 60)} min "
              f"({clicks} clicks, {dict_n} dictation), shift at calibration posture x {cx:+.0f} pt, y {cy:+.0f} pt, "
              f"{'posture-aware' if n >= 8 else 'plain shift'}")
        g = d.geo
        gp = len(d.geo_pts)
        print(f"Geometric model (experimental, config geo_model={d.geo_enabled}): "
              f"{'ACTIVE' if d.use_geo() else 'not active'}; {gp} points with head geometry"
              + (f", held-out error {g.held:.0f} pt (must be under 100)" if g.held is not None else " (needs 15 points to test)"))
    if SAMPLES.exists():
        lines = SAMPLES.read_text().splitlines()[-50:]
        face = [l for l in lines if l.split(",")[1]]
        inside = [l for l in face if l.split(",")[3] == "1"]
        unsure = [l for l in face if len(l.split(",")) > 4 and l.split(",")[4] == "0"]
        print(f"Last {len(lines)} samples: face found in {len(face)}, on the reply area in {len(inside)}, "
              f"posture uncertain in {len(unsure)}")
        if face:
            print("Latest gaze point (pt):", ",".join(face[-1].split(",")[1:3]))


def test(cfg):
    cap = open_camera(cfg)
    ff = FaceFeatures()
    found = 0
    for _ in range(15):
        ok, frame = cap.read()
        if ok and ff(frame) is not None:
            found += 1
    cap.release()
    print(f"Camera {cfg['camera']}: face found in {found} of 15 frames.")
    win = front_window(cfg["apps"], front_only=False)
    if not win:
        print("Screen: no T3 Code or Claude window found.")
        return
    t0 = time.time()
    lines = read_window_text(win, HOME / "test_window.png")
    if lines is None:
        print("Screen: capture failed. Give Terminal the Screen Recording permission.")
    elif not lines:
        print("Screen: captured, but no text found. Give Terminal the Screen Recording permission.")
    else:
        print(f"Screen: read {len(lines)} text lines in {time.time() - t0:.1f} s. Example: {lines[len(lines)//2][0]!r}")


def evaluate(cfg):
    """Honest check on real data: train on the earlier 70% of known points, test on the later 30%."""
    import numpy as np
    from Quartz import CGDisplayScreenSize, CGMainDisplayID
    pts = [json.loads(l) for l in ANCHORS.read_text().splitlines()] if ANCHORS.exists() else []
    pts = [a for a in pts if a.get("o")]
    if len(pts) < 15:
        print(f"Only {len(pts)} known points with head geometry. Need 15 (click around for a few minutes).")
        return
    pts.sort(key=lambda a: a["t"])
    k = int(len(pts) * 0.7)
    tr, te = pts[:k], pts[k:]
    mm = CGDisplayScreenSize(CGMainDisplayID())
    sw, sh = screen_size()
    g = GeoGaze((sw, sh), (mm.width / 10, mm.height / 10))
    O = lambda P: np.array([a["o"] for a in P], float)
    Tg = lambda P: np.array([[np.nan if a["target"][0] is None else a["target"][0], a["target"][1]] for a in P], float)
    hx = lambda P: np.array([a["target"][0] is not None for a in P])
    g.fit(O(tr), Tg(tr), hx(tr))
    def err(pred, P):
        T, h = Tg(P), hx(P)
        return np.where(h, np.hypot(pred[:, 0] - np.nan_to_num(T[:, 0]), pred[:, 1] - T[:, 1]), np.abs(pred[:, 1] - T[:, 1]))
    e_geo = err(g._predict(g.p, O(te)), te)
    # baseline: dot calibration + plain shift learned on the training part
    dx = np.median([a["dx"] for a in tr if a["dx"] is not None]); dy = np.median([a["dy"] for a in tr])
    raw = np.array([[a["target"][0] if a["target"][0] is not None else 0, a["target"][1]] for a in te], float) \
        - np.array([[a["dx"] if a["dx"] is not None else 0, a["dy"]] for a in te], float)
    e_base = err(raw + [dx, dy], te)
    span = (te[-1]["t"] - tr[0]["t"]) / 60
    print(f"{len(tr)} training points, {len(te)} later test points, over {span:.0f} min.")
    print(f"Median error on the later points: geometric model {np.median(e_geo):.0f} pt, "
          f"old model with shift {np.median(e_base):.0f} pt.")
    print(f"Typical line height is about 20 pt; a paragraph is about 60-150 pt.")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["calibrate", "start", "stop", "status", "run", "test", "eval"])
    args = ap.parse_args()
    cfg = load_config()
    {"calibrate": calibrate, "start": start, "stop": lambda c: stop_tracker(),
     "status": status, "run": run, "test": test, "eval": evaluate}[args.cmd](cfg)


if __name__ == "__main__":
    main()
