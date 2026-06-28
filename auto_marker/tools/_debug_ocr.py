"""调试脚本 — 查看OCR识别文本"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np

import fitz
from PIL import Image

from core.text_detector import detect_text

pdf_path = ROOT / "data" / "test_samples" / "301_2026-06-18_003.pdf"
doc = fitz.open(str(pdf_path))
page = doc.load_page(0)
pix = page.get_pixmap(matrix=fitz.Matrix(200/72, 200/72), colorspace=fitz.csRGB)
img = Image.frombytes('RGB', (pix.width, pix.height), pix.samples)
doc.close()

det_boxes, ocr_records = detect_text(np.array(img), 0)
print('=== 全部OCR识别记录 (%d条) ===' % len(ocr_records))
for i, r in enumerate(ocr_records):
    cy = (r['rec_bbox'][1] + r['rec_bbox'][3]) / 2.0
    ch = r['rec_bbox'][3] - r['rec_bbox'][1]
    print('  [%d] "%s" (score=%.3f) bbox=%s y_center=%.1f h=%d' % (
        i, r['rec_text'], r['rec_score'], r['rec_bbox'], cy, ch))

print()
print('=== 打印det_boxes高度分类 ===')
from core.layout_analyzer import analyze_layout

layout = analyze_layout(det_boxes, np.array(img).shape[0], 0,
                        ocr_records=ocr_records)  # 传入ocr_records
for i, r in enumerate(layout['printed_boxes']):
    bbox = r['bbox_pixel']
    h = bbox[3] - bbox[1]
    cy = (bbox[1] + bbox[3]) / 2.0
    print('  [%d] type=%s bbox=%s h=%d y_center=%.1f' % (i, r['type'], bbox, h, cy))

print()
print('=== 题号检测结果 ===')
qr = layout.get('question_regions', {})
for pg, regions in qr.items():
    print('  第 %d 页: %d 道题' % (pg + 1, len(regions)))
    for r in regions:
        print('    q_idx=%d marker="%s" y=[%.0f, %.0f] marker_bbox=%s' % (
            r['q_idx'], r['marker_text'], r['y_start'], r['y_end'], r['marker_bbox']))

print()
print('=== handwriting_boxes 题号分配 ===')
for hb in layout['handwriting_boxes']:
    bbox = hb['bbox_pixel']
    cy = (bbox[1] + bbox[3]) / 2.0
    print('  bbox=%s cy=%.1f q_idx=%s' % (bbox, cy, hb.get('question_idx')))
