"""
调试 — 探查新流水线提取的字符序列。
"""
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

from pathlib import Path

import fitz
import numpy as np
from PIL import Image

# 渲染 PDF 第一页
pdf_path = Path(__file__).resolve().parent / "data" / "incoming" / "301_2025-03-20_001.pdf"
if not pdf_path.exists():
    pdf_path = Path(__file__).resolve().parent / "301_2025-03-20_001.pdf"
doc = fitz.open(str(pdf_path))
page = doc.load_page(0)
pix = page.get_pixmap(matrix=fitz.Matrix(200/72, 200/72), colorspace=fitz.csRGB)
img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
img_np = np.array(img)
doc.close()

print(f"图片尺寸: {img_np.shape[1]}x{img_np.shape[0]}")

# Step 1: 文本检测 + 识别
from core.text_detector import detect_text

det_boxes, ocr_records = detect_text(img_np, page_idx=0)
print(f"\n检测框: {len(det_boxes)} 个")
print(f"OCR 记录: {len(ocr_records)} 条")

# Step 2: 版面分析
from core.layout_analyzer import analyze_layout

layout_result = analyze_layout(det_boxes, img_np.shape[0], page_idx=0)
hw_boxes = layout_result["handwriting_boxes"]
print(f"\n手写区域: {len(hw_boxes)} 个")
print(f"印刷区域: {len(layout_result['printed_boxes'])} 个")

# Step 3: 手写识别匹配
from core.handwriting_recognizer import recognize_handwriting

results = recognize_handwriting(img_np, hw_boxes, ocr_records, page_idx=0)

# 打印字符序列
chars = [r["char"] for r in results]
print(f"\n提取的字符序列 ({len(chars)} 字):")
print("".join(chars))

# 按 question_idx 分组
from collections import defaultdict

by_q = defaultdict(list)
for r in results:
    by_q[r.get("question_idx")].append(r["char"])
for q_idx in sorted(by_q.keys()):
    print(f"  题 {q_idx+1}: {''.join(by_q[q_idx])}")

# 打印 OCR 记录 rec_texts (全量)
print(f"\n=== 全量 OCR 识别文本 ({len(ocr_records)} 条) ===")
for i, rec in enumerate(ocr_records):
    print(f"  [{i}] '{rec['rec_text']}' (score={rec['rec_score']:.3f}, bbox={rec['rec_bbox']})")

# 打印 handwriting_boxes 与 matched OCR 记录的对应关系
print("\n=== 手写区域匹配详情 ===")
for i, hw in enumerate(hw_boxes):
    print(f"  HW[{i}]: bbox={hw['bbox_pixel']}, q_idx={hw.get('question_idx')}")
