"""输出 Qwen3-VL 对每个手写区域的原始识别内容。"""
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
from core.layout_analyzer import analyze_layout
from core.qwen_vl_recognizer import recognize_with_qwen_vl, _call_qwen_vl

PDF = str(ROOT / "data" / "test_samples" / "301_2026-06-18_003.pdf")

# 渲染 + 预处理
doc = fitz.open(PDF)
page = doc[0]
zoom = OCR_DPI / 72
pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), colorspace=fitz.csRGB)
img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
doc.close()

proc = np.array(preprocess_image(img))
print(f"图片尺寸: {proc.shape[1]}x{proc.shape[0]}")

# 加载 OCR 缓存做版面分析
fh = _file_hash(Path(PDF))
det_boxes, ocr_records = _load_ocr_cache(fh, 0)
layout = analyze_layout(det_boxes, proc.shape[0], 0, ocr_records=ocr_records)
hw_boxes = layout["handwriting_boxes"]
print(f"手写区域: {len(hw_boxes)} 个\n")

# 逐区域输出 Qwen3-VL 原始识别结果
for i, box in enumerate(hw_boxes):
    bbox = box["bbox_pixel"]
    q_idx = box.get("question_idx")
    x0, y0, x2, y2 = bbox
    cropped = proc[y0:y2, x0:x2]
    raw_text = _call_qwen_vl(cropped)
    chars = [ch for ch in raw_text if ord(ch) >= 0x4E00]
    print(f"[{i:2d}] q_idx={q_idx} bbox=({x0},{y0},{x2},{y2}) size={x2-x0}x{y2-y0}")
    print(f"     原始: '{raw_text}'")
    print(f"     汉字: '{''.join(chars)}' ({len(chars)}字)")
    print()
