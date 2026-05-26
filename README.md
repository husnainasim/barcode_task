# Barcode Orientation Correction & Decoding Pipeline

Turns photos of barcode-tagged items into clean, upright, deskewed crops — one
image per barcode — and decodes each one. Built for the supplied dataset of
plastic-wrapped items (tilted Code 128 "binary" labels under glare) and flat
paper labels (EAN-8/13, Code 39).

## What it produces

For every barcode found, one JPEG in `barcode_output/` that is:
- **Tightly cropped** to the white paper label (bars + human-readable digits + label border) — never the whole scene.
- **Upright and deskewed**: bars vertical, the code reading left-to-right, the digit string at the bottom.

Plus `results.json` (full machine-readable report incl. orientation + barcode
position quad) and `results.csv` (spreadsheet summary).

---

## How it works (industry-standard decode-first approach)

The core idea: **decode first, then use the decoder's reported geometry to crop
and deskew.** `zxing-cpp` returns, for each barcode, a `position` quad (the four
corners of the bars) and an `orientation` (degrees). That geometry is exact, so
we use it instead of guessing the rotation from pixel heuristics.

Per image, two passes — both routed through the same quad-based cropper:

1. **Region pass.** Detect candidate label regions (morphological bar-pattern
   detection, which works even for white-label-on-white-item). For each region:
   1. Decode the crop as-is.
   2. If that fails, **deskew** via the bar angle (`HoughLinesP`) and decode
      again — this is the key recall step for the tilted plastic-wrap labels,
      which `zxing` cannot read until the bars are roughly vertical.
   3. If still failing, retry with preprocessing variants (CLAHE / Otsu /
      adaptive threshold / sharpen / upscale).
   4. If everything fails *and* the crop genuinely contains a bar pattern, save
      it as `…__UNREAD.jpg` for manual inspection (background-texture regions
      are filtered out so the output isn't flooded).
2. **Full-image pass.** Decode the whole image to catch anything the region
   pass missed; each hit is still cropped tightly from its quad.

### The quad-based cropper (`crop_upright_from_quad`)

`zxing` orders the quad corners in the barcode's own reading frame
(`TL→TR` along the bars, `TL→BL` toward the printed digits). The quad bounds
only the bars — often a thin scan band whose height doesn't reflect the real
bar length — so the crop window is sized **relative to the barcode width**
(which scales with bar height + text):

- along the bar row: half-width × `(1 + 2·pad_x)` → quiet zones at both ends;
- above the bars: `up_frac · width` → small headroom;
- below the bars: `dn_frac · width` → captures bar height + the digit row.

A single perspective warp (`getPerspectiveTransform` + `warpPerspective`)
deskews and rectifies. Because the window is built in the barcode's own frame,
bars end up vertical with text at the bottom **by construction** for normal
reading directions. The one exception zxing flags explicitly — `orientation ==
180` (symbol read upside-down) — gets a corrective 180° flip. Finally the crop
is trimmed to the white label rectangle.

### Symbology filter

Decoding is restricted to **Code 128, Code 39, EAN-8, EAN-13**. This removes the
spurious GS1 DataBar Stacked `(01)…` false-positives that plastic-wrap texture
otherwise produces.

---

## Setup

- Python 3.11+ (tested on 3.13).
- `zxing-cpp` ships self-contained binaries — **no external ZBar/DLL needed** on
  Windows.

```bash
cd "e:\drive-download-20260526T004106Z-3-001"
pip install -r requirements.txt
```

`requirements.txt`: `opencv-python`, `zxing-cpp`, `numpy`.

---

## Run

```bash
python barcode_pipeline.py
```

Processes every image (`.jpg/.jpeg/.png/.bmp/.tiff`) in the script's directory
and writes results to `barcode_output/` (the folder is wiped and recreated each
run).

---

## Output files

| File | Description |
|------|-------------|
| `<source>__r<N>__<data>.jpg` | Upright, deskewed crop for each decoded barcode (`r<N>` = region index, or `rfull` from the full-image pass) |
| `<source>__r<NN>__UNREAD.jpg` | A bar-pattern region that could not be decoded (glare/wrap/torn label) — kept for inspection |
| `results.json` | Per-barcode: `source_image`, `region`, `region_box`, `orientation`, `quad`, `data`, `type`, `output_file` |
| `results.csv` | `source_image, region, data, type, orientation_deg, output_file` |

---

## Notes & limitations

- A few labels in the dataset are physically **torn** (digits missing) or shot
  at very steep angles; these still come out upright but may show partial digits
  — the best achievable from the source photo.
- `UNREAD` crops are often alternate views of a code already decoded from another
  photo; the report deduplicates decoded values per image.
