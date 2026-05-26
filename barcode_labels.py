"""
Barcode Label Orientation Correction & Decoding  (dataset: barcode_1..39.jpg)
============================================================================
Tasks:
  1. Correct orientation -- bars vertical, text at bottom, deskewed, tight crop.
  2. Decode the barcode value.

Self-contained -- no dependency on barcode_pipeline.py.

Run:
    python barcode_labels.py
    -- or --
    C:\\Users\\hp\\AppData\\Local\\Programs\\Python\\Python313\\python.exe barcode_labels.py
Deps: opencv-python, zxing-cpp, numpy.
"""

import re, sys, json, shutil, math
from pathlib import Path

import cv2
import numpy as np
import zxingcpp

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
INPUT_DIR  = Path(__file__).parent
OUTPUT_DIR = INPUT_DIR / "barcode_labels_output_v2"
NAME_RE    = re.compile(r"^barcode_(\d+)\.(jpg|jpeg|png|bmp|tif|tiff)$", re.IGNORECASE)
FORMATS    = [
    zxingcpp.BarcodeFormat.Code128,
    zxingcpp.BarcodeFormat.Code39,
    zxingcpp.BarcodeFormat.EAN8,
    zxingcpp.BarcodeFormat.EAN13,
]


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------

def _to_gray(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img


def _rotate_arb(img, angle_deg):
    """Rotate by angle_deg (degrees), expanding canvas to fit."""
    h, w = img.shape[:2]
    cx, cy = w / 2.0, h / 2.0
    M = cv2.getRotationMatrix2D((cx, cy), angle_deg, 1.0)
    cos_a, sin_a = abs(M[0, 0]), abs(M[0, 1])
    nw = int(h * sin_a + w * cos_a)
    nh = int(h * cos_a + w * sin_a)
    M[0, 2] += nw / 2.0 - cx
    M[1, 2] += nh / 2.0 - cy
    return cv2.warpAffine(img, M, (nw, nh),
                          flags=cv2.INTER_CUBIC,
                          borderMode=cv2.BORDER_REPLICATE)


def _bar_angle(gray):
    """Dominant bar angle via HoughLinesP. Returns degrees (bars are near-vertical lines)."""
    eq    = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    edges = cv2.Canny(eq, 30, 100)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180,
                             threshold=40, minLineLength=20, maxLineGap=5)
    if lines is None:
        return 0.0
    angles = []
    for x1, y1, x2, y2 in lines[:, 0]:
        a = math.degrees(math.atan2(y2 - y1, x2 - x1)) if x2 != x1 else 90.0
        if a > 45:   a -= 90
        elif a < -45: a += 90
        angles.append(a)
    return float(np.median(angles)) if angles else 0.0


# ---------------------------------------------------------------------------
# Quad-based upright crop (for decoded images)
# ---------------------------------------------------------------------------

def _region_crop_from_quad(src, barcode):
    """
    Cut a region around the barcode quad from src.
    Pad is sized to include the text row (0.55× bar-width) on all sides,
    keeping the crop tight enough that _make_upright_nocode sees mostly label.
    """
    p  = barcode.position
    xs = [p.top_left.x, p.top_right.x, p.bottom_right.x, p.bottom_left.x]
    ys = [p.top_left.y, p.top_right.y, p.bottom_right.y, p.bottom_left.y]
    tl = np.array([p.top_left.x, p.top_left.y], dtype=float)
    tr = np.array([p.top_right.x, p.top_right.y], dtype=float)
    lx = float(np.linalg.norm(tr - tl))   # bar-stripe width
    pad = max(int(lx * 0.55), 20)          # enough for text row + white border
    H, W = src.shape[:2]
    x1 = max(0, int(min(xs)) - pad)
    y1 = max(0, int(min(ys)) - pad)
    x2 = min(W, int(max(xs)) + pad)
    y2 = min(H, int(max(ys)) + pad)
    crop = src[y1:y2, x1:x2]
    return crop if crop.size > 0 else src


def _warp_from_quad(src, barcode):
    """
    Produce an upright crop (bars vertical, text at bottom) for a decoded image.

    Cuts a generous region around the quad, then runs _make_upright_nocode
    on that region.  This avoids all fragile vector geometry.
    """
    region = _region_crop_from_quad(src, barcode)
    return _make_upright_nocode(region)


# ---------------------------------------------------------------------------
# Orientation correction for UNREAD images (no quad available)
# ---------------------------------------------------------------------------

def _make_upright_nocode(img):
    """
    Produce an upright crop without a decoded quad:
      1. Deskew via dominant bar angle.
      2. Bars vertical: rotate 90 CW if row-variance > col-variance.
      3. Landscape: rotate 90 CW if portrait.
      4. Text at bottom: if dense (bar) rows are in bottom half, flip 180.
    No _trim_white (unreliable on plastic-bag backgrounds).
    """
    # 1. Deskew
    gray = _to_gray(img)
    ang  = _bar_angle(gray)
    work = _rotate_arb(img, ang) if abs(ang) > 0.5 else img.copy()

    # 2. Bars vertical: after this step bars are columns (high col-variance).
    g  = _to_gray(work)
    eq = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(g)
    _, bw = cv2.threshold(eq, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    col_var = float(np.var(np.mean(bw, axis=0)))
    row_var = float(np.var(np.mean(bw, axis=1)))
    if row_var > col_var:
        work = cv2.rotate(work, cv2.ROTATE_90_CLOCKWISE)

    # 3. Portrait: rotate 90 CW so bars become rows and text is top/bottom.
    #    Now the image is taller than wide; "rows" run perpendicular to bars.
    if work.shape[1] > work.shape[0]:
        work = cv2.rotate(work, cv2.ROTATE_90_CLOCKWISE)

    # 4. Text at bottom: in portrait orientation, bars occupy the top portion
    #    (rows with high ink density) and the text row is below them.
    #    Compare mean ink in top-half rows vs bottom-half rows.
    #    If bottom half has more ink → flip 180 so text moves to bottom.
    g2  = _to_gray(work)
    eq2 = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(g2)
    _, bw2 = cv2.threshold(eq2, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    rowink = np.mean(bw2, axis=1) / 255.0
    mid        = len(rowink) // 2
    top_ink    = float(np.mean(rowink[:mid]))
    bottom_ink = float(np.mean(rowink[mid:]))
    if bottom_ink > top_ink * 1.15:   # bottom is meaningfully denser → flip
        work = cv2.rotate(work, cv2.ROTATE_180)

    return work


# ---------------------------------------------------------------------------
# Decode helpers
# ---------------------------------------------------------------------------

def _decode(img_bgr):
    gray = _to_gray(img_bgr)
    return zxingcpp.read_barcodes(gray, formats=FORMATS, try_rotate=True)


def _decode_variants(img_bgr):
    """
    Try CLAHE / Otsu / sharpen / upscale on img_bgr.
    Returns list of barcodes (decoded from a grayscale variant).
    NOTE: returned barcodes have quad coords in img_bgr space (not the variant),
    so callers should use img_bgr as the base when quad accuracy matters.
    For upscale variant the coords are scaled -- handled by caller.
    """
    gray = _to_gray(img_bgr)
    variants = []
    variants.append(('clahe',  cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(gray), 1.0))
    _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    variants.append(('otsu',   otsu,  1.0))
    k = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]], np.float32)
    variants.append(('sharp',  cv2.filter2D(gray, -1, k), 1.0))
    variants.append(('up2x',   cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC), 2.0))
    for name, v, scale in variants:
        hits = zxingcpp.read_barcodes(v, formats=FORMATS, try_rotate=True)
        if hits:
            if scale != 1.0:
                # scale quad coords back to original image space
                hits = _rescale_quads(hits, 1.0 / scale)
            return hits
    return []


def _rescale_quads(hits, factor):
    """Return hits list with quad coordinates multiplied by factor."""
    # zxingcpp result objects are read-only; rebuild via a wrapper
    class _Pt:
        def __init__(self, x, y): self.x = x; self.y = y
    class _Pos:
        def __init__(self, tl, tr, br, bl):
            self.top_left = tl; self.top_right = tr
            self.bottom_right = br; self.bottom_left = bl
    class _BC:
        def __init__(self, orig, pos):
            self.text = orig.text; self.format = orig.format
            self.orientation = orig.orientation; self.position = pos
    out = []
    for bc in hits:
        p = bc.position
        pos = _Pos(
            _Pt(p.top_left.x * factor,     p.top_left.y * factor),
            _Pt(p.top_right.x * factor,    p.top_right.y * factor),
            _Pt(p.bottom_right.x * factor, p.bottom_right.y * factor),
            _Pt(p.bottom_left.x * factor,  p.bottom_left.y * factor),
        )
        out.append(_BC(bc, pos))
    return out


def _detect_regions(img):
    """Return (x,y,w,h) bounding boxes of candidate bar regions."""
    gray = _to_gray(img)
    eq   = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(gray)
    _, bw = cv2.threshold(eq, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kh = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
    kv = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 15))
    comb = cv2.bitwise_and(cv2.dilate(bw, kh), cv2.dilate(bw, kv))
    k2   = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    comb = cv2.morphologyEx(comb, cv2.MORPH_CLOSE, k2, iterations=3)
    cnts, _ = cv2.findContours(comb, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    area0 = img.shape[0] * img.shape[1]
    out = []
    for c in cnts:
        if cv2.contourArea(c) < area0 * 0.005:
            continue
        x, y, w, h = cv2.boundingRect(c)
        if w >= 30 and h >= 10:
            out.append((x, y, w, h))
    return out


def _padded_crop(img, x, y, w, h, pad=20):
    H, W = img.shape[:2]
    return img[max(0,y-pad):min(H,y+h+pad), max(0,x-pad):min(W,x+w+pad)]


# ---------------------------------------------------------------------------
# Decode ladder
# ---------------------------------------------------------------------------

def decode_ladder(img):
    """
    Return (base_bgr, barcode_obj) or (None, None).
    base_bgr is the exact image the barcode was decoded from, so that the
    quad coordinates in barcode_obj are valid for base_bgr.
    """
    # a) raw
    hits = _decode(img)
    if hits: return img, hits[0]

    # b) bar-angle deskew
    ang  = _bar_angle(_to_gray(img))
    desk = _rotate_arb(img, ang) if abs(ang) > 0.5 else img
    if desk is not img:
        hits = _decode(desk)
        if hits: return desk, hits[0]

    # c) preprocessing variants on deskewed image
    hits = _decode_variants(desk)
    if hits: return desk, hits[0]

    # d) region crops (full-scene images)
    for (x, y, w, h) in _detect_regions(img):
        crop = _padded_crop(img, x, y, w, h)
        if crop.size == 0: continue
        hits = _decode(crop)
        if hits: return crop, hits[0]
        ca   = _bar_angle(_to_gray(crop))
        cd   = _rotate_arb(crop, ca) if abs(ca) > 0.5 else crop
        hits = _decode(cd)
        if hits: return cd, hits[0]
        hits = _decode_variants(cd)
        if hits: return cd, hits[0]

    return None, None


def _quad_dict(position):
    p = position
    return {
        "top_left":     [p.top_left.x,     p.top_left.y],
        "top_right":    [p.top_right.x,    p.top_right.y],
        "bottom_right": [p.bottom_right.x, p.bottom_right.y],
        "bottom_left":  [p.bottom_left.x,  p.bottom_left.y],
    }


# ---------------------------------------------------------------------------
# Per-image
# ---------------------------------------------------------------------------

def process_label_image(path: Path) -> dict:
    img = cv2.imread(str(path))
    if img is None:
        print(f"  [WARN] cannot load {path.name}")
        return {"source_image": path.name, "decoded": False,
                "data": None, "type": None, "orientation": None,
                "quad": None, "output_file": None}

    stem = path.stem
    base, bc = decode_ladder(img)

    if bc is not None:
        data    = bc.text
        upright = _warp_from_quad(base, bc)
        if upright is None or upright.size == 0:
            # Fallback: orientation correction without quad
            upright = _make_upright_nocode(img)
        safe    = "".join(c if (c.isalnum() or c in "-_.") else "_" for c in data)[:40]
        fname   = f"{stem}__{safe}.jpg"
        cv2.imwrite(str(OUTPUT_DIR / fname), upright, [cv2.IMWRITE_JPEG_QUALITY, 95])
        print(f"  [OK] {path.name}: '{data}' ({bc.format}) "
              f"orient={int(bc.orientation)}deg -> {fname}")
        return {"source_image": path.name, "decoded": True,
                "data": data, "type": str(bc.format),
                "orientation": int(bc.orientation),
                "quad": _quad_dict(bc.position), "output_file": fname}

    # Undecodable: orientation correction only
    upright = _make_upright_nocode(img)
    fname   = f"{stem}__UNREAD.jpg"
    cv2.imwrite(str(OUTPUT_DIR / fname), upright, [cv2.IMWRITE_JPEG_QUALITY, 95])
    print(f"  [--] {path.name}: not decoded -> {fname} (orientation corrected)")
    return {"source_image": path.name, "decoded": False,
            "data": None, "type": None, "orientation": None,
            "quad": None, "output_file": fname}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
    OUTPUT_DIR.mkdir()

    images = [p for p in INPUT_DIR.iterdir() if p.is_file() and NAME_RE.match(p.name)]
    images.sort(key=lambda p: int(NAME_RE.match(p.name).group(1)))
    if not images:
        print(f"No barcode_<N>.jpg images found in {INPUT_DIR}")
        sys.exit(1)

    print(f"Found {len(images)} label image(s). Output -> {OUTPUT_DIR}\n")
    records = []
    for p in images:
        records.append(process_label_image(p))

    decoded = sum(1 for r in records if r["decoded"])
    print(f"\n{'='*60}")
    print(f"DONE: {decoded}/{len(images)} decoded "
          f"({len(images)-decoded} UNREAD / orientation-only)")
    print(f"{'='*60}")

    with open(OUTPUT_DIR / "results.json", "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)

    with open(OUTPUT_DIR / "results.csv", "w", encoding="utf-8") as f:
        f.write("source_image,data,type,orientation_deg,decoded,output_file\n")
        for r in records:
            data   = "" if r["data"]        is None else r["data"]
            typ    = "" if r["type"]        is None else r["type"]
            orient = "" if r["orientation"] is None else r["orientation"]
            f.write(f'"{r["source_image"]}","{data}","{typ}",'
                    f'{orient},{r["decoded"]},"{r["output_file"]}"\n')

    print(f"JSON -> {OUTPUT_DIR/'results.json'}")
    print(f"CSV  -> {OUTPUT_DIR/'results.csv'}")


if __name__ == "__main__":
    main()
