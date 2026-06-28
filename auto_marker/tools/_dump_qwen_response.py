"""Dump Qwen3-VL 整页识别原始 JSON 响应，用于分析 bbox 是否等宽。

用法:
    python tools/_dump_qwen_response.py
"""
import json
import sys
from pathlib import Path

import fitz
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

TEST_PDF = ROOT / "data" / "test_samples" / "301_2026-06-18_003.pdf"
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


def pdf_to_rgb_array(pdf_path: str, page_idx: int = 0, dpi: int = 200) -> np.ndarray:
    """渲染 PDF 页为 RGB numpy array。"""
    doc = fitz.open(pdf_path)
    page = doc.load_page(page_idx)
    zoom = dpi / 72
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat, colorspace=fitz.csRGB)
    img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
    doc.close()
    return np.array(img)


def main():
    img = pdf_to_rgb_array(str(TEST_PDF), 0, 200)
    print(f"图像尺寸: {img.shape[1]}x{img.shape[0]} (WxH)")

    from core.qwen_vl_recognizer import _call_qwen_vl_page_level, _parse_page_level_json

    response = _call_qwen_vl_page_level(img, ANSWERS)
    print("\n===== 原始响应 =====")
    print(response)

    parsed = _parse_page_level_json(response)
    if parsed is None:
        print("\nJSON 解析失败")
        return

    img_h, img_w = img.shape[:2]
    print(f"\n===== 解析后逐字 bbox (img_w={img_w}, img_h={img_h}) =====")
    for q_item in parsed:
        q_marker = q_item.get("q_marker", "")
        chars = q_item.get("chars", [])
        invalid = q_item.get("invalid", False)
        print(f"\n题号 '{q_marker}' (invalid={invalid}, {len(chars)} 字):")
        if not chars:
            print("  (无字)")
            continue
        # 打印每字的 bbox_norm 和转换后的像素 bbox
        prev_x1 = None
        for i, ci in enumerate(chars):
            bn = ci["bbox_norm"]
            px = (int(bn[0] * img_w), int(bn[1] * img_h),
                  int(bn[2] * img_w), int(bn[3] * img_h))
            cx = (px[0] + px[2]) / 2
            cw = px[2] - px[0]
            gap = ""
            if prev_x1 is not None:
                gap = f"  gap_to_prev={px[0] - prev_x1}px"
            print(f"  [{i}] '{ci['char']}' bbox_norm=({bn[0]:.3f},{bn[1]:.3f},{bn[2]:.3f},{bn[3]:.3f})"
                  f" → px=({px[0]},{px[1]},{px[2]},{px[3]}) cx={cx:.0f} w={cw}px{gap}")
            prev_x1 = px[2]

        # 统计 x 间距
        if len(chars) >= 2:
            cxs = [(int(ci["bbox_norm"][0] * img_w) + int(ci["bbox_norm"][2] * img_w)) / 2
                   for ci in chars]
            gaps = [cxs[i + 1] - cxs[i] for i in range(len(cxs) - 1)]
            widths = [int(ci["bbox_norm"][2] * img_w) - int(ci["bbox_norm"][0] * img_w)
                      for ci in chars]
            print(f"  x 中心间距: {[round(g, 1) for g in gaps]}")
            print(f"  bbox 宽度:   {[w for w in widths]}")
            if len(set(round(g) for g in gaps)) == 1:
                print(f"  ⚠️  等宽间距: {gaps[0]:.1f}px (模型返回等宽 bbox)")
            else:
                print(f"  ✅ 非等宽间距 (模型返回真实 bbox)")


if __name__ == "__main__":
    main()
