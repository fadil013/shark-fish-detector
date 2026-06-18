"""
Perfect Shark & Fish Detector — Image Mode
==========================================
  Swarm  : 4x3 tiled (12 tiles) + full-image bonus pass at 0.04 conf
  Single : 3-pass pipeline
             Pass 1 — full image (main shark)
             Pass 2 — shark crop (validate / tighten)
             Pass 3 — belly tiles 400-500 px wide @ 1280 API
                       => 3x upscale of pilot fish (80 px native -> 240 px API)
                       => model can finally see them
"""

import cv2
import numpy as np
import requests
import base64
import os
from collections import Counter
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.environ["ROBOFLOW_API_KEY"]
API_URL = "https://serverless.roboflow.com/aquarium-combined/3"

CONF_SWARM  = 0.06   # low — catches faint background hammerheads
CONF_SINGLE = 0.01   # very low — pilot fish can be faint
CONF_BELLY  = 0.04   # belly-tile pass — slightly higher to suppress noise

API_W_TILE  = 1024   # per swarm tile
API_W_FULL  = 1280   # full-image / single passes
API_W_BELLY = 1280   # belly-region tiles (big upscale of small crop)

# (input, output, swarm)
IMAGES = [
    ("1.png",           "1_out.png",     True),
    ("2.png",           "2_out.png",     False),
    ("frame_swarm.jpg", "swarm_out.jpg", True),
    ("frame_mid.jpg",   "mid_out.jpg",   False),
    ("frame_late.jpg",  "late_out.jpg",  False),
]

RED   = (30,  30, 220)
GREEN = (30, 190,  40)

SHARK_CLASSES = {
    "shark", "hammerhead", "hammerhead shark", "bull shark",
    "tiger shark", "whale shark", "nurse shark", "oceanic whitetip",
}
DISP = {
    "hammerhead":       "Hammerhead",
    "hammerhead shark": "Hammerhead",
    "shark":            "Shark",
    "oceanic whitetip": "Whitetip",
    "stingray":         "Stingray",
    "jellyfish":        "Jellyfish",
    "fish":             "Fish",
    "starfish":         "Starfish",
    "penguin":          "Penguin",
    "puffin":           "Puffin",
}


# ═════════════════════════════════════════════════════════════════════════════
# 1.  Preprocessing
# ═════════════════════════════════════════════════════════════════════════════

def enhance(frame):
    b, g, r = cv2.split(frame)
    r = cv2.add(r, 22)
    frame = cv2.merge([b, g, r])
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b_ = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=2.8, tileGridSize=(8, 8)).apply(l)
    return cv2.cvtColor(cv2.merge([l, a, b_]), cv2.COLOR_LAB2BGR)


# ═════════════════════════════════════════════════════════════════════════════
# 2.  Geometry
# ═════════════════════════════════════════════════════════════════════════════

def iou(a, b):
    ix1 = max(a[0], b[0]); iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2]); iy2 = min(a[3], b[3])
    inter = max(0, ix2-ix1) * max(0, iy2-iy1)
    if inter == 0:
        return 0.0
    return inter / ((a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter)


def rect_overlap(a, b):
    return not (a[2] < b[0] or a[0] > b[2] or a[3] < b[1] or a[1] > b[3])


# ═════════════════════════════════════════════════════════════════════════════
# 3.  Per-detection fixes
# ═════════════════════════════════════════════════════════════════════════════

def fix_class(det, H, W, swarm):
    x1, y1, x2, y2 = det["x1"], det["y1"], det["x2"], det["y2"]
    w = x2-x1; h = y2-y1
    if w <= 0 or h <= 0:
        return None
    ar = w / max(h, 1)
    size = max(w, h)
    lbl  = det["label"]

    # ── Universal noise ───────────────────────────────────────────────────────
    if ar > 3.5 and h < H*0.06 and y2 < H*0.45:   # flat surface reflection
        return None
    if size < W*0.015 and (x1 < 4 or x2 > W-4 or y1 < 4):  # border sliver
        return None

    # ── Swarm ─────────────────────────────────────────────────────────────────
    if swarm:
        if lbl in ("fish", "jellyfish", "stingray", "penguin", "starfish"):
            det["label"] = "hammerhead" if size >= 25 else None
            if det["label"] is None:
                return None
        if max(w, h) < 50:          # tiny tile-edge noise
            return None
        if ar > 2.8 and h < H*0.07: # flat box
            return None
        if w < 55 and (x1 <= 2 or x2 >= W-2):  # narrow frame-edge artifact
            return None

    # ── Single-shark ──────────────────────────────────────────────────────────
    else:
        if lbl in SHARK_CLASSES:
            if ar > 2.5 and h < 55:   # flat water-reflection
                return None
            if size < 60:              # too small for any shark
                return None
        if lbl == "fish":
            if size >= 200 and (ar > 2.2 or ar < 0.45):
                det["label"] = "shark"
            elif ar > 3.2 and h < 30:  # very flat and short → surface reflection artifact
                return None
        elif lbl == "jellyfish":
            if ar > 2.0 or ar < 0.50 or x1 < 15 or x2 > W-15:
                det["label"] = "shark"
            elif size < 120:   # small jellyfish near shark = pilot fish misclassified
                det["label"] = "fish"

    return det


# ═════════════════════════════════════════════════════════════════════════════
# 4.  Context reclassification
# ═════════════════════════════════════════════════════════════════════════════

def context_reclassify(dets, swarm):
    if swarm:
        for d in dets:
            if d["label"] == "shark":
                d["label"] = "hammerhead"
    return dets


# ═════════════════════════════════════════════════════════════════════════════
# 5.  NMS
# ═════════════════════════════════════════════════════════════════════════════

def nms(dets, iou_thresh):
    if not dets:
        return dets
    dets = sorted(dets, key=lambda d: d["conf"], reverse=True)
    kept = []
    suppressed = set()
    for i, d in enumerate(dets):
        if i in suppressed:
            continue
        bi = (d["x1"], d["y1"], d["x2"], d["y2"])
        ai = max((d["x2"]-d["x1"])*(d["y2"]-d["y1"]), 1)
        for j, e in enumerate(dets):
            if j <= i or j in suppressed:
                continue
            bj = (e["x1"], e["y1"], e["x2"], e["y2"])
            aj = max((e["x2"]-e["x1"])*(e["y2"]-e["y1"]), 1)
            ov = iou(bi, bj)

            if ov > 0.65:              # cross-class near-duplicate
                suppressed.add(j); continue

            if d["label"] == e["label"]:
                if ov > iou_thresh:    # same-class IoU duplicate
                    suppressed.add(j); continue
                # Same-class containment: smaller box 65%+ inside bigger → suppress
                ix1=max(bi[0],bj[0]); iy1=max(bi[1],bj[1])
                ix2=min(bi[2],bj[2]); iy2=min(bi[3],bj[3])
                ic = max(0,ix2-ix1)*max(0,iy2-iy1)
                if ic / min(ai, aj) > 0.65:
                    suppressed.add(j); continue
        kept.append(d)
    return kept


# ═════════════════════════════════════════════════════════════════════════════
# 5b. Post-NMS: drop false-positive fish = shark body re-detected
# ═════════════════════════════════════════════════════════════════════════════

def filter_shark_body_fish(dets, H, W):
    min_dim = min(H, W)
    sharks  = [d for d in dets if d["label"] in SHARK_CLASSES]
    out = []
    for d in dets:
        if d["label"] in SHARK_CLASSES:
            out.append(d); continue
        fw = d["x2"]-d["x1"]; fh = d["y2"]-d["y1"]
        if max(fw, fh) > min_dim * 0.08:       # too large for pilot fish
            continue
        bi = (d["x1"], d["y1"], d["x2"], d["y2"])
        bad = False
        for s in sharks:
            sbx = (s["x1"], s["y1"], s["x2"], s["y2"])
            if iou(bi, sbx) > 0.28:
                bad = True; break
            # Fish 92%+ inside shark bbox AND far from ALL shark edges
            # → shark body texture detected as fish (false positive)
            ix1_=max(bi[0],sbx[0]); iy1_=max(bi[1],sbx[1])
            ix2_=min(bi[2],sbx[2]); iy2_=min(bi[3],sbx[3])
            ic_ = max(0,ix2_-ix1_)*max(0,iy2_-iy1_)
            fi_ = max(fw*fh, 1)
            if ic_ / fi_ > 0.92:
                sw_s = max(sbx[2]-sbx[0], 1); sh_s = max(sbx[3]-sbx[1], 1)
                fx_c = (bi[0]+bi[2])/2; fy_c = (bi[1]+bi[3])/2
                epx = min(fx_c-sbx[0], sbx[2]-fx_c) / sw_s
                epy = min(fy_c-sbx[1], sbx[3]-fy_c) / sh_s
                if epx > 0.15 and epy > 0.15:  # far from all edges → false positive
                    bad = True; break
        if not bad:
            out.append(d)
    return out


# ═════════════════════════════════════════════════════════════════════════════
# 5c. Post-NMS single-mode: merge partial re-detections into one shark box
# ═════════════════════════════════════════════════════════════════════════════

def merge_partial_sharks(dets, swarm):
    """Fold secondary shark detections (tail/fin close-ups) into the main shark."""
    if swarm:
        return dets
    sharks = [d for d in dets if d["label"] in SHARK_CLASSES]
    other  = [d for d in dets if d["label"] not in SHARK_CLASSES]
    if len(sharks) <= 1:
        return dets
    main = max(sharks, key=lambda d: (d["x2"]-d["x1"])*(d["y2"]-d["y1"]))
    kept = [main]
    for s in sharks:
        if s is main:
            continue
        sw = max(1, s["x2"] - s["x1"])
        x_overlap = max(0, min(main["x2"], s["x2"]) - max(main["x1"], s["x1"]))
        if x_overlap / sw > 0.40:
            # Secondary shark x-range is 40%+ within main → partial detection, merge
            main["x1"] = min(main["x1"], s["x1"])
            main["y1"] = min(main["y1"], s["y1"])
            main["x2"] = max(main["x2"], s["x2"])
            main["y2"] = max(main["y2"], s["y2"])
        else:
            kept.append(s)
    return kept + other


# ═════════════════════════════════════════════════════════════════════════════
# 6.  API helpers
# ═════════════════════════════════════════════════════════════════════════════

def _post(tile, api_w, conf):
    h, w = tile.shape[:2]
    api_h = int(h * api_w / w)
    small = cv2.resize(tile, (api_w, api_h))
    _, buf = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 96])
    payload = base64.b64encode(buf).decode()
    params  = {"api_key": API_KEY, "confidence": max(1, int(conf * 100))}
    for attempt in range(4):
        try:
            r = requests.post(
                API_URL, params=params,
                data=payload,
                headers={"Content-Type": "application/x-www-form-urlencoded"},
                timeout=40,
            )
            r.raise_for_status()
            break
        except Exception as exc:
            if attempt == 3:
                raise
            import time; time.sleep(3 * (attempt + 1))
    sx = w / api_w; sy = h / api_h
    out = []
    for p in r.json().get("predictions", []):
        cx=p["x"]; cy=p["y"]; pw=p["width"]; ph=p["height"]
        out.append({
            "x1": int((cx-pw/2)*sx), "y1": int((cy-ph/2)*sy),
            "x2": int((cx+pw/2)*sx), "y2": int((cy+ph/2)*sy),
            "label": p.get("class","fish").lower(),
            "conf":  p.get("confidence", 0.0),
        })
    return out


# ═════════════════════════════════════════════════════════════════════════════
# 7.  Swarm: 4x3 tiled inference + bonus full-image pass
# ═════════════════════════════════════════════════════════════════════════════

def infer_tiled(img, n_cols, n_rows, conf):
    H, W = img.shape[:2]
    overlap = 0.35
    tile_w = int(W/(n_cols-overlap*(n_cols-1))) if n_cols>1 else W
    tile_h = int(H/(n_rows-overlap*(n_rows-1))) if n_rows>1 else H
    step_x = int(tile_w*(1-overlap)) if n_cols>1 else W
    step_y = int(tile_h*(1-overlap)) if n_rows>1 else H
    tile_w = min(tile_w, W); tile_h = min(tile_h, H)

    all_dets = []
    n_calls = 0
    for row in range(n_rows):
        for col in range(n_cols):
            tx1 = min(col*step_x, W-tile_w); ty1 = min(row*step_y, H-tile_h)
            tile = img[ty1:ty1+tile_h, tx1:tx1+tile_w]
            raw  = _post(tile, API_W_TILE, conf)
            n_calls += 1
            for d in raw:
                d["x1"]+=tx1; d["y1"]+=ty1; d["x2"]+=tx1; d["y2"]+=ty1
            all_dets.extend(raw)
            print(f"    [{row},{col}]({tx1},{ty1}) -> {len(raw)} dets", flush=True)

    # Bonus full-image pass at lower confidence to catch faint background sharks
    print(f"    bonus full-image pass (conf={conf*0.5:.3f})...", flush=True)
    bonus = _post(img, API_W_TILE, conf * 0.5)
    print(f"    bonus: {len(bonus)} additional raw dets")
    all_dets.extend(bonus)

    print(f"    {n_calls+1} API calls | {len(all_dets)} raw total")
    return all_dets, H, W


# ═════════════════════════════════════════════════════════════════════════════
# 8.  Single-shark: 3-pass pipeline
#     Pass 1 — full image              → shark
#     Pass 2 — shark-region crop       → confirm / tighten
#     Pass 3 — belly tiles 400-500 px  → pilot fish (3x upscale)
# ═════════════════════════════════════════════════════════════════════════════

def infer_full(img, conf):
    H, W = img.shape[:2]
    all_raw = []

    # ── Pass 1: full image ───────────────────────────────────────────────────
    raw1 = _post(img, API_W_FULL, conf)
    all_raw.extend(raw1)

    # Get bounding box of all sharks found in pass 1
    sharks1 = [p for p in raw1 if p["label"].lower() in SHARK_CLASSES]

    if sharks1:
        sx1 = min(p["x1"] for p in sharks1); sy1 = min(p["y1"] for p in sharks1)
        sx2 = max(p["x2"] for p in sharks1); sy2 = max(p["y2"] for p in sharks1)
        sw  = sx2 - sx1; sh = sy2 - sy1

        # ── Pass 2: shark-region zoom (15% pad each side) ────────────────────
        px = int(sw*0.15); py = int(sh*0.15)
        cx1=max(0,sx1-px); cy1=max(0,sy1-py)
        cx2=min(W-1,sx2+px); cy2=min(H-1,sy2+py)
        crop2 = img[cy1:cy2, cx1:cx2]
        if crop2.size > 0:
            raw2 = _post(crop2, API_W_FULL, conf)
            for p in raw2:
                p["x1"]+=cx1; p["y1"]+=cy1; p["x2"]+=cx1; p["y2"]+=cy1
            all_raw.extend(raw2)

        # ── Pass 3: perimeter tiles for pilot fish ───────────────────────────
        # Use the LARGEST shark from pass1 only (avoids bloated bounding box
        # from distant background detections bloating the search region).
        ms = max(sharks1, key=lambda p: (p["x2"]-p["x1"])*(p["y2"]-p["y1"]))
        msx1, msy1, msx2, msy2 = ms["x1"], ms["y1"], ms["x2"], ms["y2"]
        msw = msx2 - msx1; msh = msy2 - msy1

        def _fish_strip(ry1, ry2, rx1, rx2, tag):
            """Tile-scan one strip, collect only fish detections."""
            rw = rx2-rx1; rh = ry2-ry1
            if rw <= 0 or rh <= 0:
                return
            nt = max(1, round(rw / 430))
            tw_ = rw // nt
            print(f"    {tag}: {nt} tile(s) ~{tw_}x{rh}px"
                  f" -> ~{int(API_W_BELLY/max(tw_,1)*80)}px fish", flush=True)
            for i in range(nt):
                tx1 = rx1 + i * tw_
                tx2 = min(tx1 + tw_, rx2)
                tile = img[ry1:ry2, tx1:tx2]
                if tile.size == 0:
                    continue
                raw_ = _post(tile, API_W_BELLY, CONF_BELLY)
                fish_ = []
                for p in raw_:
                    if p["label"].lower() in SHARK_CLASSES:
                        continue
                    p["label"] = "fish"
                    p["x1"] += tx1; p["y1"] += ry1
                    p["x2"] += tx1; p["y2"] += ry1
                    fish_.append(p)
                all_raw.extend(fish_)
                print(f"      tile {i+1}/{nt}: {len(raw_)} raw, "
                      f"{len(fish_)} fish kept", flush=True)

        # 3a. Bottom strip: 10% above shark bottom edge → 40% below
        _fish_strip(
            max(0,     msy2 - int(msh * 0.10)),
            min(H-1,   msy2 + int(msh * 0.40)),
            max(0,     msx1 - int(msw * 0.05)),
            min(W-1,   msx2 + int(msw * 0.05)),
            "pass3-bottom",
        )
        # 3b. Left side strip: 20% of shark width to the left of shark, mid-body y
        _fish_strip(
            max(0,     msy1 + int(msh * 0.10)),
            min(H-1,   msy2 - int(msh * 0.10)),
            max(0,     msx1 - int(msw * 0.20)),
            min(W-1,   msx1 + int(msw * 0.05)),
            "pass3-left",
        )
        # 3c. Right side strip: 20% of shark width to the right of shark, mid-body y
        _fish_strip(
            max(0,     msy1 + int(msh * 0.10)),
            min(H-1,   msy2 - int(msh * 0.10)),
            max(0,     msx2 - int(msw * 0.05)),
            min(W-1,   msx2 + int(msw * 0.20)),
            "pass3-right",
        )
        # 3d. Interior grid: 3×2 scan inside shark body for pilot fish alongside
        # Pilot fish swim right alongside the shark body (within its bounding box
        # when seen in a 2D photo). The API will detect the shark body as "shark"
        # (filtered) or as large "fish" (filtered by size). Real pilot fish are
        # small → survive both filters.
        ic = 3; ir = 2  # columns, rows
        itw = (msx2 - msx1) // ic; ith = (msy2 - msy1) // ir
        print(f"    pass3-interior: {ic}x{ir} grid, each ~{itw}x{ith}px"
              f" -> ~{int(API_W_BELLY/max(itw,1)*80)}px fish", flush=True)
        for row in range(ir):
            for col in range(ic):
                ix1 = msx1 + col * itw; ix2 = min(msx1 + (col+1)*itw, msx2)
                iy1 = msy1 + row * ith;  iy2 = min(msy1 + (row+1)*ith, msy2)
                tile = img[iy1:iy2, ix1:ix2]
                if tile.size == 0:
                    continue
                raw_ = _post(tile, API_W_BELLY, CONF_BELLY * 1.5)  # stricter conf
                fish_ = []
                for p in raw_:
                    if p["label"].lower() in SHARK_CLASSES:
                        continue
                    p["label"] = "fish"
                    p["x1"] += ix1; p["y1"] += iy1
                    p["x2"] += ix1; p["y2"] += iy1
                    fish_.append(p)
                all_raw.extend(fish_)
                if fish_:
                    print(f"      [{row},{col}]: {len(raw_)} raw, "
                          f"{len(fish_)} fish kept", flush=True)

    return all_raw, H, W


# ═════════════════════════════════════════════════════════════════════════════
# 9.  Drawing (label collision avoidance)
# ═════════════════════════════════════════════════════════════════════════════

def draw(frame, dets, swarm):
    H, W = frame.shape[:2]
    rs   = max(W, H) / 1500.0
    bw   = max(3, int(4 * rs))
    font = cv2.FONT_HERSHEY_SIMPLEX
    fs   = max(0.38, (0.50 if swarm else 0.66) * rs)
    tk   = max(1, int(2 * rs))

    # Draw large boxes first (bigger = more important = label priority)
    ordered = sorted(dets,
                     key=lambda d: (d["x2"]-d["x1"])*(d["y2"]-d["y1"]),
                     reverse=True)
    placed = []  # placed label rects

    for d in ordered:
        x1=max(0,d["x1"]); y1=max(0,d["y1"])
        x2=min(W-1,d["x2"]); y2=min(H-1,d["y2"])
        if x2<=x1 or y2<=y1:
            continue
        lbl   = d["label"]
        conf  = d["conf"]
        color = RED if lbl in SHARK_CLASSES else GREEN
        name  = DISP.get(lbl, lbl.title())

        # Box
        cv2.rectangle(frame, (x1+3,y1+3), (x2+3,y2+3), (0,0,0), bw+2)
        cv2.rectangle(frame, (x1,y1), (x2,y2), color, bw)
        bright = tuple(min(255,int(c*1.4)) for c in color)
        cv2.rectangle(frame, (x1+bw,y1+bw), (x2-bw,y2-bw), bright, 1)

        # Corner brackets
        ln = max(8, min(32, (x2-x1)//6, (y2-y1)//6))
        ct = bw + 2
        for (cx_,cy_,sx_,sy_) in [(x1,y1,1,1),(x2,y1,-1,1),(x1,y2,1,-1),(x2,y2,-1,-1)]:
            cv2.line(frame, (cx_,cy_), (cx_+sx_*ln,cy_), color, ct)
            cv2.line(frame, (cx_,cy_), (cx_,cy_+sy_*ln), color, ct)

        # Label (skip if overlaps an existing one)
        tag = f"{name}  {conf*100:.0f}%"
        (tw,th_),bl = cv2.getTextSize(tag, font, fs, tk)
        pad = 4
        ty  = max(y1-2, th_+pad*2)
        lr  = (x1, ty-th_-pad, x1+tw+pad*2, ty+bl+1)
        if not any(rect_overlap(lr, pr) for pr in placed):
            cv2.rectangle(frame, (lr[0]+2,lr[1]+2), (lr[2]+2,lr[3]+2), (0,0,0), -1)
            cv2.rectangle(frame, (lr[0],lr[1]), (lr[2],lr[3]), color, -1)
            cv2.putText(frame, tag, (lr[0]+pad, ty), font, fs, (255,255,255), tk, cv2.LINE_AA)
            placed.append(lr)

    # HUD
    counts = Counter(DISP.get(d["label"],d["label"].title()) for d in dets)
    hy  = max(40, int(42*rs)); hdy = max(26, int(32*rs)); hfs = max(0.65, 0.80*rs)
    for sp in sorted(counts):
        is_shark = any(d["label"] in SHARK_CLASSES
                       for d in dets if DISP.get(d["label"],d["label"].title())==sp)
        color = RED if is_shark else GREEN
        text  = f"{sp}: {counts[sp]}"
        cv2.putText(frame, text, (14,hy), font, hfs, (0,0,0), 4, cv2.LINE_AA)
        cv2.putText(frame, text, (14,hy), font, hfs, color,   2, cv2.LINE_AA)
        hy += hdy

    # Footer
    footer = "Shark & Fish Detector  |  Tiled 4x3 + Belly Pass  |  aquarium-combined v3"
    cv2.putText(frame, footer, (12,H-14), font, 0.42, (0,0,0),       3, cv2.LINE_AA)
    cv2.putText(frame, footer, (12,H-14), font, 0.42, (200,200,200), 1, cv2.LINE_AA)


# ═════════════════════════════════════════════════════════════════════════════
# 10. Per-image pipeline
# ═════════════════════════════════════════════════════════════════════════════

def process(src, dst, swarm):
    if not os.path.exists(src):
        print(f"  [skip] {src}"); return
    frame = cv2.imread(src)
    if frame is None:
        print(f"  [!] Cannot read {src}"); return

    H_orig, W_orig = frame.shape[:2]
    print(f"\n-- {src}  ({W_orig}x{H_orig})  [{'SWARM' if swarm else 'SINGLE'}] --")

    enhanced = enhance(frame)

    if swarm:
        print("  Tiling 4x3 (12 tiles) + bonus full pass...")
        raw, H, W = infer_tiled(enhanced, 4, 3, CONF_SWARM)
        nms_thresh = 0.18
    else:
        print("  3-pass: full + shark zoom + belly tiles...")
        raw, H, W = infer_full(enhanced, CONF_SINGLE)
        nms_thresh = 0.38

    dets = [fix_class(d, H, W, swarm) for d in raw]
    dets = [d for d in dets if d is not None]
    dets = context_reclassify(dets, swarm)
    dets = nms(dets, nms_thresh)
    if not swarm:
        dets = filter_shark_body_fish(dets, H, W)
        dets = merge_partial_sharks(dets, swarm)

    sharks = [d for d in dets if d["label"] in SHARK_CLASSES]
    fish   = [d for d in dets if d["label"] not in SHARK_CLASSES]
    print(f"  Result: {len(sharks)} sharks | {len(fish)} fish  (total {len(dets)})")
    for d in sorted(dets, key=lambda x: x["conf"], reverse=True):
        name = DISP.get(d["label"], d["label"])
        print(f"    [{name:12s}] {d['conf']*100:5.1f}%  "
              f"({d['x1']},{d['y1']},{d['x2']},{d['y2']})")

    out_frame = frame.copy()
    draw(out_frame, dets, swarm)
    enc = [cv2.IMWRITE_JPEG_QUALITY, 97] if dst.lower().endswith(".jpg") else []
    cv2.imwrite(dst, out_frame, enc)
    print(f"  Saved -> {os.path.abspath(dst)}")


# ═════════════════════════════════════════════════════════════════════════════
# 11. Entry point
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("\n=== Perfect Shark & Fish Detector ===")
    for src, dst, swarm in IMAGES:
        process(src, dst, swarm)
    print("\n=== Done! ===\n")
