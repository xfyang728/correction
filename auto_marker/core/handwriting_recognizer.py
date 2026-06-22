"""
手写提取与识别模块 — 从全图 OCR 结果中匹配手写区域。

阶段三：接收 layout_analyzer 输出的 handwriting_boxes 和
text_detector 返回的全图 ocr_records（单次 predict() 调用提取），
通过 bbox 重叠匹配，将识别文本分配给手写区域。

不再对每个手写区域独立调用 predict()（避免重复模型加载和子图检测失败），
改为从全图 OCR 结果中通过坐标匹配直接提取识别文本。

输出与现有 grader 兼容的 per_char_results 格式。
"""

import logging

logger = logging.getLogger("handwriting_recognizer")


def _bbox_center(bbox: tuple) -> tuple[float, float]:
    """返回 bbox 中心点 (cx, cy)。"""
    return ((bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0)


def _point_in_bbox(px: float, py: float, bbox: tuple) -> bool:
    """判断点 (px, py) 是否在 bbox (x0,y0,x2,y2) 内。"""
    return bbox[0] <= px <= bbox[2] and bbox[1] <= py <= bbox[3]


def _is_chinese_char(ch: str) -> bool:
    """判断是否为中文字符。"""
    code = ord(ch)
    return (
        0x4E00 <= code <= 0x9FFF
        or 0x3400 <= code <= 0x4DBF
        or 0xF900 <= code <= 0xFAFF
    )


def recognize_handwriting(
    img,
    handwriting_boxes: list[dict],
    ocr_records: list[dict],
    page_idx: int = 0,
) -> list[dict]:
    """从全图 OCR 记录中匹配手写区域，返回逐字结果。

    参数:
        img: RGB numpy array (H, W, 3) — 原图（仅用于获取尺寸）
        handwriting_boxes: layout_analyzer 输出的 handwriting_boxes，每项含：
            - bbox_pixel: (x0,y0,x2,y2) 手写区域在原图中的坐标
            - question_idx: 题号（可选）
            - confidence: 检测置信度
        ocr_records: text_detector 返回的全图识别记录，每项含：
            - rec_text: 识别文本
            - rec_score: 置信度
            - rec_bbox: (x0, y0, x1, y1) 识别框
            - page: 页码
        page_idx: 页码

    返回:
        [
            {
                "page": 0,
                "bbox_pixel": (x0, y0, x2, y2),  # 在原图中的坐标
                "char": "春",
                "confidence": 0.92,
                "question_idx": 0,
                "img_pixel_w": 2481,  # 原图宽度
                "img_pixel_h": 3508,  # 原图高度
            },
            ...
        ]
    """
    import numpy as np

    if not isinstance(img, np.ndarray):
        from PIL import Image
        img = np.array(Image.open(img) if isinstance(img, str) else img)

    img_h, img_w = img.shape[:2]

    # ---- Step 1: 为每个手写区域匹配 OCR 记录 ----
    # 匹配规则：OCR 记录 rec_bbox 的中心点落在 handwriting_bbox 内
    matched: list[dict] = []
    unmatched_boxes = list(handwriting_boxes)

    for rec in ocr_records:
        if rec.get("page", 0) != page_idx:
            continue
        rec_bbox = rec["rec_bbox"]  # (x0, y0, x1, y1)
        cx, cy = _bbox_center(rec_bbox)

        # 找包含该中心点的手写框
        best_box = None
        best_idx = -1
        for i, hw_box in enumerate(unmatched_boxes):
            hw_bbox = hw_box["bbox_pixel"]
            if _point_in_bbox(cx, cy, hw_bbox):
                best_box = hw_box
                best_idx = i
                break

        if best_box is not None:
            matched.append({
                "handwriting_box": best_box,
                "rec_text": rec["rec_text"],
                "rec_score": rec["rec_score"],
                "rec_bbox": rec_bbox,
            })
            # 已匹配的框不再参与后续匹配
            unmatched_boxes.pop(best_idx)

    logger.info(
        "第 %d 页: %d 个手写区域, 匹配到 %d 条识别记录, %d 个未匹配",
        page_idx + 1, len(handwriting_boxes), len(matched), len(unmatched_boxes),
    )

    # ---- Step 2: 将匹配结果切分为单字 ----
    all_results: list[dict] = []

    for m in matched:
        hw_box = m["handwriting_box"]
        bbox = hw_box["bbox_pixel"]
        question_idx = hw_box.get("question_idx")
        rec_text = m["rec_text"]
        rec_score = m["rec_score"]

        # 提取中文字符
        chars = [ch for ch in rec_text if _is_chinese_char(ch)]
        if not chars:
            logger.debug("匹配结果无中文字符: '%s'", rec_text)
            continue

        # 等宽切分：在原图坐标系下估算每个字符 bbox
        # 加 5% 内边距使中心点更贴近字符视觉中心
        bx0, by0, bx2, by2 = bbox
        region_w = bx2 - bx0
        char_w = region_w / len(chars)
        margin = char_w * 0.05  # 每侧收缩 5%，避免标记偏移到字符间隙

        for i, ch in enumerate(chars):
            cx0 = int(bx0 + i * char_w + margin)
            cx2 = int(bx0 + (i + 1) * char_w - margin)
            # 确保最小宽度
            if cx2 <= cx0:
                cx0 = int(bx0 + i * char_w)
                cx2 = int(bx0 + (i + 1) * char_w)

            all_results.append({
                "page": page_idx,
                "bbox_pixel": (cx0, by0, cx2, by2),
                "char": ch,
                "confidence": rec_score,
                "question_idx": question_idx,
                "img_pixel_w": img_w,
                "img_pixel_h": img_h,
            })

    logger.info(
        "第 %d 页: 匹配 %d 条记录 → %d 个字符",
        page_idx + 1, len(matched), len(all_results),
    )
    return all_results