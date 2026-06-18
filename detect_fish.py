"""
Production-grade Shark & Fish Detector
=======================================
Detection  : Roboflow aquarium-combined v3  (shark / fish / stingray / jellyfish …)
Tracker    : SORT  —  Kalman filter  +  Hungarian assignment  +  IoU matching
Class      : Temporal voting  —  tracks accumulate class history, majority wins
NMS        : Class-aware IoU suppression  (shark-shark, fish-fish, cross-class)
Smoothing  : Exponential Moving Average on box coordinates
Tightening : GrabCut inside each loose model box
Drawing    : Thick dark-red/green boxes, scaled for 4K, drop-shadow labels
"""

import cv2
import numpy as np
import requests
import base64
import os
import sys
from collections import defaultdict
from scipy.optimize import linear_sum_assignment
from dotenv import load_dotenv

load_dotenv()

# ─── CONFIG ───────────────────────────────────────────────────────────────────
API_KEY      = os.environ["ROBOFLOW_API_KEY"]
API_URL      = "https://serverless.roboflow.com/aquarium-combined/3"
INPUT        = "Input.mp4"
OUTPUT       = "output.mp4"

CONF         = 0.18       # min detection confidence (lower = more detections)
SKIP         = 1          # API every N frames

API_W        = 1280       # frame width sent to API (higher = better for 4K)

# NMS
IOU_NMS      = 0.35       # IoU threshold for class-aware NMS (lower = more aggressive merging)

# SORT tracker
MAX_AGE      = 8          # frames before a track is deleted
MIN_HITS     = 2          # frames before a track is shown
IOU_TRACK    = 0.25       # IoU threshold for Kalman assignment

# Smoothing
EMA_ALPHA    = 0.65       # EMA weight for box coords (higher = faster response)

# Class voting
VOTE_WINDOW  = 12         # last N frames used for class vote

# Drawing
BOX_THICK    = 5
FONT_SCALE   = 1.0
# ──────────────────────────────────────────────────────────────────────────────

RED   = (0,   0,  200)
GREEN = (0,  175,  30)

SHARK_CLASSES = {
    "shark","hammerhead","hammerhead shark","bull shark",
    "tiger shark","whale shark","nurse shark","oceanic whitetip",
}
LABEL = {
    "hammerhead":       "hammerhead",
    "hammerhead shark": "hammerhead",
    "shark":            "shark",
    "oceanic whitetip": "whitetip",
    "bull shark":       "bull shark",
    "stingray":         "stingray",
    "jellyfish":        "jellyfish",
    "starfish":         "starfish",
    "penguin":          "penguin",
    "puffin":           "puffin",
    "fish":             "fish",
}


def box_color(label):  return RED if label in SHARK_CLASSES else GREEN
def disp(label):       return LABEL.get(label, label)


# ═══════════════════════════════════════════════════════════════════════════════
# 1.  ENHANCEMENT
# ═══════════════════════════════════════════════════════════════════════════════

def enhance(frame):
    """CLAHE + mild red-channel boost to compensate for underwater blue cast."""
    # Red boost
    b, g, r = cv2.split(frame)
    r = cv2.add(r, 25)
    frame = cv2.merge([b, g, r])
    # CLAHE on L channel
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b_ = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8)).apply(l)
    return cv2.cvtColor(cv2.merge([l, a, b_]), cv2.COLOR_LAB2BGR)


# ═══════════════════════════════════════════════════════════════════════════════
# 2.  BOX TIGHTENING  (GrabCut with HSV fallback)
# ═══════════════════════════════════════════════════════════════════════════════

def tighten_grabcut(frame, x1, y1, x2, y2):
    """Use GrabCut to find tight foreground mask, return tight bounding rect."""
    H, W = frame.shape[:2]
    rx1, ry1 = max(x1, 0), max(y1, 0)
    rx2, ry2 = min(x2, W-1), min(y2, H-1)
    bw, bh = rx2 - rx1, ry2 - ry1
    if bw < 20 or bh < 20:
        return x1, y1, x2, y2

    # GrabCut needs ≥ 2 px margin inside the rect
    margin = 4
    rect = (margin, margin, bw - 2*margin, bh - 2*margin)
    roi  = frame[ry1:ry2, rx1:rx2].copy()

    try:
        mask = np.zeros(roi.shape[:2], np.uint8)
        bgdModel = np.zeros((1, 65), np.float64)
        fgdModel = np.zeros((1, 65), np.float64)
        cv2.grabCut(roi, mask, rect, bgdModel, fgdModel, 3, cv2.GC_INIT_WITH_RECT)
        fg = np.where((mask == 2) | (mask == 0), 0, 255).astype(np.uint8)
    except Exception:
        fg = None

    # Fallback to HSV if GrabCut fails or returns empty
    if fg is None or fg.sum() == 0:
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        bg  = cv2.inRange(hsv, np.array([88, 45, 30]), np.array([140, 255, 255]))
        fg  = cv2.bitwise_not(bg)

    # Morphological clean-up
    k  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN,  k, iterations=1)
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, k, iterations=3)

    pts = cv2.findNonZero(fg)
    if pts is None:
        return x1, y1, x2, y2

    tx, ty, tw, th = cv2.boundingRect(pts)
    pad  = 10
    nx1  = max(rx1 + tx - pad,  0)
    ny1  = max(ry1 + ty - pad,  0)
    nx2  = min(rx1 + tx + tw + pad, W)
    ny2  = min(ry1 + ty + th + pad, H)

    orig_area  = max((x2-x1)*(y2-y1), 1)
    tight_area = (nx2-nx1)*(ny2-ny1)
    if 0.10 < tight_area/orig_area < 0.90:
        return nx1, ny1, nx2, ny2
    return x1, y1, x2, y2


# ═══════════════════════════════════════════════════════════════════════════════
# 3.  CLASS-AWARE NMS
# ═══════════════════════════════════════════════════════════════════════════════

def iou(a, b):
    ix1 = max(a[0], b[0]); iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2]); iy2 = min(a[3], b[3])
    inter = max(0, ix2-ix1) * max(0, iy2-iy1)
    if inter == 0: return 0.0
    ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter / max(ua, 1)


def fix_class(det, frame_h, frame_w):
    """Per-detection post-processing: filter noise, reclassify mismatches."""
    x1, y1, x2, y2 = det["x1"], det["y1"], det["x2"], det["y2"]
    w = x2 - x1; h = y2 - y1
    if w <= 0 or h <= 0:
        return None
    ar = w / max(h, 1)

    # ── Noise / false-positive filters ───────────────────────────────────────
    # Very flat box anywhere in frame → surface reflection or noise, never a real animal
    if ar > 2.0 and h < frame_h * 0.10:
        return None
    # Tiny box — too small to be reliable
    if w * h < (frame_w * 0.015) * (frame_h * 0.015):
        return None
    # Tiny noise at absolute frame border
    if max(w, h) < frame_w * 0.02 and (x1 < 4 or x2 > frame_w - 4 or y1 < 4):
        return None

    # ── Class fixes ──────────────────────────────────────────────────────────
    # Reclassify elongated "fish" → shark
    if det["label"] == "fish":
        size = max(w, h)
        if size >= 80 and (ar > 1.8 or ar < 0.55):
            det["label"] = "shark"
        elif size >= 50 and (ar > 2.5 or ar < 0.40):
            det["label"] = "shark"
    # Reclassify "jellyfish" at frame edges or elongated → shark fin/tail
    if det["label"] == "jellyfish":
        if ar > 1.8 or ar < 0.55 or x1 < 15 or x2 > frame_w - 15:
            det["label"] = "shark"
    return det


def context_reclassify(dets, frame_h, frame_w):
    """If ≥2 sharks in scene, treat any remaining large 'fish' as shark."""
    sharks = [d for d in dets if d["label"] in SHARK_CLASSES]
    if len(sharks) >= 2:
        for d in dets:
            if d["label"] == "fish":
                w = d["x2"] - d["x1"]; h = d["y2"] - d["y1"]
                if max(w, h) >= 70:
                    d["label"] = "shark"
    return dets


def nms(dets, iou_thresh=IOU_NMS):
    """Class-aware NMS with cross-class near-duplicate suppression."""
    if not dets:
        return dets
    dets = sorted(dets, key=lambda d: d["conf"], reverse=True)
    kept = []
    suppressed = set()
    for i, d in enumerate(dets):
        if i in suppressed:
            continue
        box_i = (d["x1"], d["y1"], d["x2"], d["y2"])
        area_i = max((d["x2"]-d["x1"])*(d["y2"]-d["y1"]), 1)
        for j, e in enumerate(dets):
            if j <= i or j in suppressed:
                continue
            box_j = (e["x1"], e["y1"], e["x2"], e["y2"])
            area_j = max((e["x2"]-e["x1"])*(e["y2"]-e["y1"]), 1)
            ov = iou(box_i, box_j)
            # Cross-class near-duplicate: same box, different label → keep higher conf
            if ov > 0.65:
                suppressed.add(j)
                continue
            # Same-class duplicate
            if d["label"] == e["label"] and ov > iou_thresh:
                suppressed.add(j)
                continue
            # Fish mostly inside shark box → suppress fish
            if d["label"] in SHARK_CLASSES and e["label"] not in SHARK_CLASSES:
                ix1 = max(box_i[0], box_j[0]); iy1 = max(box_i[1], box_j[1])
                ix2 = min(box_i[2], box_j[2]); iy2 = min(box_i[3], box_j[3])
                inter = max(0, ix2-ix1)*max(0, iy2-iy1)
                if inter / area_j > 0.75:
                    suppressed.add(j)
        kept.append(d)
    return kept


# ═══════════════════════════════════════════════════════════════════════════════
# 4.  KALMAN FILTER  (SORT formulation)
# ═══════════════════════════════════════════════════════════════════════════════

def _xywh_to_xyxy(x):
    cx, cy, s, r = x[0,0], x[1,0], x[2,0], x[3,0]
    s = max(s, 1.0)
    w = np.sqrt(s * max(r, 0.1))
    h = s / max(w, 1.0)
    return np.array([cx-w/2, cy-h/2, cx+w/2, cy+h/2])


def _xyxy_to_state(x1, y1, x2, y2):
    w = x2 - x1; h = y2 - y1
    cx = x1 + w/2; cy = y1 + h/2
    s = w * h
    r = w / max(h, 1.0)
    return np.array([[cx], [cy], [s], [r]])


class KalmanTrack:
    _id_counter = 0

    def __init__(self, det):
        KalmanTrack._id_counter += 1
        self.id   = KalmanTrack._id_counter
        self.age  = 0
        self.hits = 1
        self.hit_streak   = 1
        self.time_no_hit  = 0

        # Class voting
        self.class_hist  = [det["label"]] * 3   # pre-seed with first detection
        self.conf_hist   = [det["conf"]]

        # EMA-smoothed box
        self.ema = np.array([det["x1"], det["y1"], det["x2"], det["y2"]], dtype=float)

        # ── Kalman state: [cx, cy, s, r,  dcx, dcy, ds] ──
        z = _xyxy_to_state(det["x1"], det["y1"], det["x2"], det["y2"])

        self.x = np.zeros((7, 1))
        self.x[:4] = z

        # State transition (constant velocity)
        self.F = np.eye(7)
        self.F[0,4]=self.F[1,5]=self.F[2,6]=1.0

        # Measurement matrix
        self.H = np.zeros((4, 7))
        self.H[:4, :4] = np.eye(4)

        # Process noise
        self.Q = np.diag([1., 1., 10., 1., 0.01, 0.01, 0.0001])

        # Measurement noise
        self.R = np.diag([1., 1., 10., 1.]) * 10.

        # Initial covariance
        self.P = np.diag([10., 10., 100., 10., 1e4, 1e4, 1e4])

    # ── Kalman predict ────────────────────────────────────────────────────────
    def predict(self):
        if self.x[2, 0] + self.x[6, 0] <= 0:
            self.x[6, 0] = 0.0
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        self.time_no_hit += 1
        if self.time_no_hit > 0:
            self.hit_streak = 0
        return _xywh_to_xyxy(self.x)

    # ── Kalman update ─────────────────────────────────────────────────────────
    def update(self, det):
        z = _xyxy_to_state(det["x1"], det["y1"], det["x2"], det["y2"])
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(7) - K @ self.H) @ self.P

        self.hits += 1
        self.hit_streak += 1
        self.time_no_hit = 0
        self.age += 1

        # Class vote
        self.class_hist.append(det["label"])
        if len(self.class_hist) > VOTE_WINDOW:
            self.class_hist.pop(0)
        self.conf_hist.append(det["conf"])
        if len(self.conf_hist) > VOTE_WINDOW:
            self.conf_hist.pop(0)

        # EMA box smooth
        new_box = np.array([det["x1"], det["y1"], det["x2"], det["y2"]], dtype=float)
        self.ema = EMA_ALPHA * new_box + (1 - EMA_ALPHA) * self.ema

    @property
    def voted_class(self):
        """Majority class over VOTE_WINDOW history — prevents single-frame flips."""
        votes = defaultdict(float)
        for i, c in enumerate(self.class_hist):
            conf = self.conf_hist[min(i, len(self.conf_hist)-1)]
            votes[c] += conf
        # If any shark vote exists, bias toward shark to avoid fish misclassification
        shark_weight = sum(v for c, v in votes.items() if c in SHARK_CLASSES)
        other_weight = sum(v for c, v in votes.items() if c not in SHARK_CLASSES)
        if shark_weight > 0 and other_weight > 0:
            votes = {c: (v * 2.0 if c in SHARK_CLASSES else v) for c, v in votes.items()}
        return max(votes, key=votes.get)

    @property
    def avg_conf(self):
        return float(np.mean(self.conf_hist)) if self.conf_hist else 0.0

    @property
    def smoothed_box(self):
        return tuple(int(v) for v in self.ema)


# ═══════════════════════════════════════════════════════════════════════════════
# 5.  SORT  (multi-object tracker)
# ═══════════════════════════════════════════════════════════════════════════════

class SORT:
    def __init__(self):
        self.tracks: list[KalmanTrack] = []

    def update(self, dets):
        # Predict all existing tracks
        predicted = []
        for t in self.tracks:
            pred = t.predict()
            predicted.append(pred)

        # Build cost matrix (1 - IoU)
        if predicted and dets:
            cost = np.ones((len(predicted), len(dets)))
            for i, pred_box in enumerate(predicted):
                for j, det in enumerate(dets):
                    det_box = (det["x1"], det["y1"], det["x2"], det["y2"])
                    cost[i, j] = 1.0 - iou(pred_box, det_box)

            row_ind, col_ind = linear_sum_assignment(cost)
            matched_t, matched_d = set(), set()

            for r, c in zip(row_ind, col_ind):
                if cost[r, c] < (1.0 - IOU_TRACK):
                    self.tracks[r].update(dets[c])
                    matched_t.add(r)
                    matched_d.add(c)

        else:
            matched_t, matched_d = set(), set()

        # Spawn new tracks for unmatched detections
        for j, det in enumerate(dets):
            if j not in matched_d:
                self.tracks.append(KalmanTrack(det))

        # Remove stale tracks
        self.tracks = [t for t in self.tracks if t.time_no_hit <= MAX_AGE]

        # Return only tracks that have been seen enough
        active = []
        for t in self.tracks:
            if t.hit_streak >= MIN_HITS or t.time_no_hit == 0:
                x1, y1, x2, y2 = t.smoothed_box
                active.append({
                    "id":    t.id,
                    "label": t.voted_class,
                    "conf":  t.avg_conf,
                    "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                })
        return active


# ═══════════════════════════════════════════════════════════════════════════════
# 6.  ROBOFLOW INFERENCE
# ═══════════════════════════════════════════════════════════════════════════════

def infer(small, sx, sy):
    H_s, W_s = small.shape[:2]
    _, buf = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 95])
    try:
        r = requests.post(
            API_URL,
            params={"api_key": API_KEY, "confidence": int(CONF * 100)},
            data=base64.b64encode(buf).decode(),
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            timeout=20,
        )
        r.raise_for_status()
        preds = r.json().get("predictions", [])
    except Exception as e:
        print(f"\n  [api] {e}")
        return []

    out = []
    for p in preds:
        cx = p["x"]; cy = p["y"]
        w  = p["width"]; h = p["height"]
        det = {
            "x1": int(cx-w/2), "y1": int(cy-h/2),
            "x2": int(cx+w/2), "y2": int(cy+h/2),
            "label": p.get("class","fish").lower(),
            "conf":  p.get("confidence", 0.0),
        }
        fixed = fix_class(det, H_s, W_s)
        if fixed:
            out.append(fixed)

    out = nms(out)
    out = context_reclassify(out, H_s, W_s)

    # Scale coords from API space → full-resolution space
    for d in out:
        d["x1"] = int(d["x1"] * sx); d["y1"] = int(d["y1"] * sy)
        d["x2"] = int(d["x2"] * sx); d["y2"] = int(d["y2"] * sy)
    return out


# ═══════════════════════════════════════════════════════════════════════════════
# 7.  DRAWING
# ═══════════════════════════════════════════════════════════════════════════════

def draw_box(frame, det):
    x1, y1, x2, y2 = det["x1"], det["y1"], det["x2"], det["y2"]
    lbl   = det["label"]
    conf  = det["conf"]
    tid   = det["id"]
    color = box_color(lbl)
    name  = disp(lbl)
    bw    = BOX_THICK

    # Drop shadow
    cv2.rectangle(frame, (x1+3, y1+3), (x2+3, y2+3), (0,0,0), bw+2)
    # Main border
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, bw)
    # Inner highlight
    bright = tuple(min(255, int(c*1.4)) for c in color)
    cv2.rectangle(frame, (x1+bw, y1+bw), (x2-bw, y2-bw), bright, 1)

    # Corner L-brackets
    ln = max(20, min(52, (x2-x1)//5, (y2-y1)//5))
    ct = bw + 3
    for (cx_, cy_, sx_, sy_) in [(x1,y1,1,1),(x2,y1,-1,1),(x1,y2,1,-1),(x2,y2,-1,-1)]:
        cv2.line(frame, (cx_, cy_), (cx_+sx_*ln, cy_),        color, ct)
        cv2.line(frame, (cx_, cy_), (cx_,          cy_+sy_*ln), color, ct)

    # Label
    tag  = f"{name}  {conf*100:.0f}%"
    font = cv2.FONT_HERSHEY_SIMPLEX
    sc   = FONT_SCALE
    tk   = 2
    (tw, th_), bl = cv2.getTextSize(tag, font, sc, tk)
    pad  = 8
    ty   = max(y1-4, th_+pad*2)

    cv2.rectangle(frame, (x1+2, ty-th_-pad+2), (x1+tw+pad*2+2, ty+bl+3), (0,0,0), -1)
    cv2.rectangle(frame, (x1,   ty-th_-pad),   (x1+tw+pad*2,   ty+bl+1), color,   -1)
    cv2.putText(frame, tag, (x1+pad, ty), font, sc, (255,255,255), tk, cv2.LINE_AA)


def draw_hud(frame, tracks):
    counts = defaultdict(int)
    for t in tracks:
        counts[disp(t["label"])] += 1
    y = 46
    for sp in sorted(counts):
        color = box_color(sp if sp in SHARK_CLASSES else "fish")
        text  = f"{sp}: {counts[sp]}"
        cv2.putText(frame, text, (18, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,0,0), 4, cv2.LINE_AA)
        cv2.putText(frame, text, (18, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color,   1, cv2.LINE_AA)
        y += 38


def draw_branding(frame):
    h    = frame.shape[0]
    text = "Shark & Fish Detector  |  SORT + Kalman + Class-Voting  |  aquarium-combined v3"
    cv2.putText(frame, text, (12, h-14), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (0,0,0),       3, cv2.LINE_AA)
    cv2.putText(frame, text, (12, h-14), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (210,210,210), 1, cv2.LINE_AA)


# ═══════════════════════════════════════════════════════════════════════════════
# 8.  MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    if not os.path.exists(INPUT):
        print(f"[ERROR] '{INPUT}' not found.")
        sys.exit(1)

    cap   = cv2.VideoCapture(INPUT)
    W     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps   = cap.get(cv2.CAP_PROP_FPS) or 25
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    api_h = int(H * API_W / W)
    sx    = W / API_W
    sy    = H / api_h

    out    = cv2.VideoWriter(OUTPUT, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    tracker = SORT()

    print(f"\nVideo  : {W}x{H} | {total} frames | {fps:.1f} fps")
    print(f"Tracker: SORT (Kalman + Hungarian)  |  class voting window = {VOTE_WINDOW}")
    print(f"Output : {OUTPUT}\n")

    idx     = 0
    tracks  = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        idx += 1

        if (idx - 1) % SKIP == 0:
            small = cv2.resize(enhance(frame), (API_W, api_h))
            dets  = infer(small, sx, sy)   # fix_class + NMS + context_reclassify inside

            # Tighten on the API-scale image (fast) then scale coords back up
            for d in dets:
                sx1 = int(d["x1"] / sx); sy1 = int(d["y1"] / sy)
                sx2 = int(d["x2"] / sx); sy2 = int(d["y2"] / sy)
                tx1, ty1, tx2, ty2 = tighten_grabcut(small, sx1, sy1, sx2, sy2)
                d["x1"] = int(tx1 * sx); d["y1"] = int(ty1 * sy)
                d["x2"] = int(tx2 * sx); d["y2"] = int(ty2 * sy)

            # SORT update
            tracks = tracker.update(dets)

        for t in tracks:
            draw_box(frame, t)
        draw_hud(frame, tracks)
        draw_branding(frame)
        out.write(frame)

        pct  = idx / total * 100
        bar  = "#" * int(pct/2) + "-" * (50 - int(pct/2))
        print(f"  [{bar}] {pct:5.1f}%  frame {idx}/{total}", end="\r", flush=True)

    cap.release()
    out.release()
    print(f"\n\nDone!  ->  {os.path.abspath(OUTPUT)}")


if __name__ == "__main__":
    main()
