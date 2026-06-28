"""Dump layout_analyzer 输出 + 模型 x 范围对比。

用于分析问题 1 (X 累积偏移)：模型 x 跨度 vs 手写框 x 跨度。
"""
import sys
import logging
from pathlib import Path

import fitz
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
logging.basicConfig(level=logging.WARNING, format="%(name)s | %(message)s")

PDF = str(ROOT / "data" / "test_samples" / "301_2026-06-18_003.pdf")

from services.pipeline_service import OCR_DPI, _file_hash, _load_ocr_cache
from core.image_processor import preprocess_image
from core.layout_analyzer import analyze_layout

# 渲染 + 预处理
doc = fitz.open(PDF)
page = doc[0]
zoom = OCR_DPI / 72
pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), colorspace=fitz.csRGB)
img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
doc.close()

proc = np.array(preprocess_image(img))
img_h, img_w = proc.shape[:2]
print(f"图片尺寸: {img_w}x{img_h} (WxH)")

# 加载 OCR 缓存做版面分析
fh = _file_hash(Path(PDF))
det_boxes, ocr_records = _load_ocr_cache(fh, 0)
layout = analyze_layout(det_boxes, img_h, 0, ocr_records=ocr_records)
hw_boxes = layout["handwriting_boxes"]
question_regions = layout["question_regions"].get(0, [])

print(f"\n===== Question Regions ({len(question_regions)} 个) =====")
for r in sorted(question_regions, key=lambda x: x["q_idx"]):
    qi = r["q_idx"]
    ys, ye = r["y_start"], r["y_end"]
    mb = r.get("marker_bbox", (0, 0, 0, 0))
    mcx = (mb[0] + mb[2]) / 2
    col = "左" if mcx < img_w / 2 else "右"
    print(f"  q_idx={qi:3d} y=[{ys},{ye}] marker_cx={mcx:.0f}({col}) text='{r.get('marker_text','')}'")

print(f"\n===== Handwriting Boxes ({len(hw_boxes)} 个) =====")
# 按 q_idx 分组
from collections import defaultdict
hw_by_q = defaultdict(list)
for box in hw_boxes:
    qi = box.get("question_idx")
    hw_by_q[qi].append(box["bbox_pixel"])

for qi in sorted(hw_by_q.keys(), key=lambda x: (x is None, x)):
    boxes = hw_by_q[qi]
    print(f"\n  q_idx={qi} ({len(boxes)} 个手写框):")
    for i, bbox in enumerate(boxes):
        x0, y0, x2, y2 = bbox
        cx = (x0 + x2) / 2
        cy = (y0 + y2) / 2
        col = "左" if cx < img_w / 2 else "右"
        print(f"    [{i}] bbox=({x0},{y0},{x2},{y2}) cx={cx:.0f}({col}) cy={cy:.0f} "
              f"size={x2-x0}x{y2-y0}")
    # 该题手写框的 x 范围
    all_x0 = min(b[0] for b in boxes)
    all_x2 = max(b[2] for b in boxes)
    print(f"    → 该题手写框 x 范围: [{all_x0}, {all_x2}] 跨度={all_x2-all_x0}px")

# 现在调用模型，对比 x 范围
print(f"\n===== 模型输出 vs 手写框 x 范围对比 =====")
from core.qwen_vl_recognizer import _call_qwen_vl_page_level, _parse_page_level_json

ANSWERS = (
    "（1）随君直到夜郎西\n"
    "（2）海内存知己\n"
    "（3）百般红紫斗芳菲\n"
    "（4）水中藻荇交横\n"
    "（5）人生自古谁无死\n"
    "（6）半竿斜日旧关城\n"
    "（7）采菊东篱下\n"
    "（8）悠然见南山\n"
    "① 质朴；② 绚丽\n"
    "下联：平凡物见证诉说追梦情"
)

original_np = np.array(img)
response = _call_qwen_vl_page_level(original_np, ANSWERS)
parsed = _parse_page_level_json(response)

if parsed is None:
    print("JSON 解析失败")
    sys.exit(1)

# 构建 region_by_qidx
region_by_qidx = {r["q_idx"]: r for r in question_regions}

for q_item in parsed:
    q_marker = q_item.get("q_marker", "")
    chars = q_item.get("chars", [])
    if not chars:
        continue

    # 题号 → q_idx
    from core.qwen_vl_recognizer import _question_marker_to_q_idx
    target_q_idx = _question_marker_to_q_idx(q_marker) if q_marker else None

    if target_q_idx is None or target_q_idx not in region_by_qidx:
        print(f"\n题号 '{q_marker}': q_idx={target_q_idx} 无匹配 region，跳过")
        continue

    region = region_by_qidx[target_q_idx]
    # 该题手写框 x 范围
    hw_boxes_for_q = [b["bbox_pixel"] for b in hw_boxes if b.get("question_idx") == target_q_idx]
    if hw_boxes_for_q:
        hw_x0 = min(b[0] for b in hw_boxes_for_q)
        hw_x2 = max(b[2] for b in hw_boxes_for_q)
    else:
        hw_x0, hw_x2 = 0, img_w

    # 模型 x 范围
    model_x0 = min(c["bbox_norm"][0] for c in chars) * img_w
    model_x2 = max(c["bbox_norm"][2] for c in chars) * img_w

    # region marker x
    mb = region.get("marker_bbox", (0, 0, 0, 0))
    marker_cx = (mb[0] + mb[2]) / 2

    print(f"\n题号 '{q_marker}' q_idx={target_q_idx} ({len(chars)} 字):")
    print(f"  模型 x 范围:    [{model_x0:.0f}, {model_x2:.0f}] 跨度={model_x2-model_x0:.0f}px")
    print(f"  手写框 x 范围:  [{hw_x0}, {hw_x2}] 跨度={hw_x2-hw_x0}px")
    print(f"  marker_cx={marker_cx:.0f} {'左列' if marker_cx < img_w/2 else '右列'}")
    ratio = (model_x2 - model_x0) / (hw_x2 - hw_x0) if hw_x2 > hw_x0 else 0
    print(f"  模型/手写框跨度比: {ratio:.2f}x {'⚠️ 模型跨度明显偏大' if ratio > 1.3 else '✅'}")
