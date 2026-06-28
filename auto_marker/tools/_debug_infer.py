"""Debug script for _infer_missing_paren_markers."""
import sys
import logging
from pathlib import Path

import fitz
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
logging.basicConfig(level=logging.INFO, format="%(name)s | %(message)s")

from services.pipeline_service import OCR_DPI, _file_hash, _load_ocr_cache
from core.image_processor import preprocess_image
from core.layout_analyzer import (
    analyze_layout,
    _find_all_question_markers,
    _question_match,
    _infer_missing_paren_markers,
    bbox_center_x,
    bbox_center_y,
    _COLUMN_GAP_THRESHOLD,
    _INFER_MISSING_Y_WINDOW,
)

pdf = ROOT / "data" / "test_samples" / "301_2026-06-18_003.pdf"
fh = _file_hash(pdf)
det_boxes, ocr_records = _load_ocr_cache(fh, 0)

# Check the 悠然见南山 record
print("=== OCR records with no question marker ===")
for rec in ocr_records:
    text = rec.get("rec_text", "")
    qi = _question_match(text)
    all_markers = _find_all_question_markers(text)
    if qi is None and not all_markers:
        bbox = rec.get("rec_bbox")
        if not bbox:
            continue
        has_chinese = any("\u4e00" <= ch <= "\u9fff" for ch in text)
        if has_chinese:
            cx = bbox_center_x(bbox)
            cy = bbox_center_y(bbox)
            print(f"  text={text!r} cx={cx:.0f} cy={cy:.0f} bbox={bbox}")

# Build markers manually (same as _detect_questions_from_boxes)
print("\n=== Building markers ===")
from core.layout_analyzer import _QIDX_OFFSET_PAREN, _QIDX_OFFSET_DOT, _QIDX_OFFSET_CIRCLE

markers = []
for rec in ocr_records:
    text = rec.get("rec_text", "")
    all_matches = _find_all_question_markers(text)
    if not all_matches:
        continue
    bbox = rec.get("rec_bbox")
    if not bbox:
        continue
    cy = bbox_center_y(bbox)
    cx = bbox_center_x(bbox)
    has_circle = any(200 <= qi < 210 for qi, _ in all_matches)
    for qi, matched_text in all_matches:
        if has_circle and 100 <= qi < 200:
            continue
        markers.append({
            "q_idx": qi,
            "marker_bbox": bbox,
            "marker_text": text,
            "y_center": cy,
            "x_center": cx,
        })

# Dedup by q_idx
markers_dict = {}
for m in markers:
    qi = m["q_idx"]
    if qi not in markers_dict or m["y_center"] < markers_dict[qi]["y_center"]:
        markers_dict[qi] = m
unique_markers = list(markers_dict.values())

print(f"Unique markers: {len(unique_markers)}")
for m in sorted(unique_markers, key=lambda x: x["q_idx"]):
    print(f"  q_idx={m['q_idx']:3d} cy={m['y_center']:.0f} cx={m['x_center']:.0f} text={m['marker_text']!r}")

# Check paren sequence
paren_qidxs = sorted(m["q_idx"] for m in unique_markers if 0 <= m["q_idx"] < 100)
print(f"\nParen q_idx sequence: {paren_qidxs}")
existing = set(paren_qidxs)
missing = [qi for qi in range(min(paren_qidxs), max(paren_qidxs) + 1) if qi not in existing]
print(f"Missing: {missing}")

# Run inference
print("\n=== Running _infer_missing_paren_markers ===")
_infer_missing_paren_markers(unique_markers, ocr_records, 0)

print(f"\nMarkers after inference: {len(unique_markers)}")
for m in sorted(unique_markers, key=lambda x: x["q_idx"]):
    inferred = m.get("inferred", False)
    tag = " [INFERRED]" if inferred else ""
    print(f"  q_idx={m['q_idx']:3d} cy={m['y_center']:.0f} cx={m['x_center']:.0f} text={m['marker_text']!r}{tag}")

# Check used_bboxes
print("\n=== Checking used_bboxes ===")
used_bboxes = set(tuple(m["marker_bbox"]) for m in unique_markers)
for rec in ocr_records:
    text = rec.get("rec_text", "")
    bbox = rec.get("rec_bbox")
    if not bbox:
        continue
    if tuple(bbox) in used_bboxes:
        print(f"  USED: text={text!r} bbox={bbox}")
