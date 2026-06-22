"""
文本检测模块 — 使用 PaddleOCR 3.7 检测模型查找文本区域。

阶段一：对预处理后的图片运行文本检测，获取所有文本区域的 bounding box。
输出结果用于阶段二（版面分析与分类）和阶段三（手写区域裁剪识别）。

PaddleOCR 3.7 的 predict() 同时返回检测和识别结果。
本模块从一次 predict() 调用中提取检测框和识别结果，
检测框交由 layout_analyzer 分类，识别结果交由 handwriting_recognizer 匹配。
"""

import logging
import numpy as np

logger = logging.getLogger("text_detector")

_det_instance = None


def _get_detector():
    """获取或初始化 PaddleOCR 引擎单例。"""
    global _det_instance
    if _det_instance is None:
        from paddleocr import PaddleOCR

        _det_instance = PaddleOCR(lang='ch')
        logger.info("文本检测器就绪（PaddleOCR 3.7, PP-OCRv6 Medium）")
    return _det_instance


def detect_text(img, page_idx: int = 0) -> tuple[list[dict], list[dict]]:
    """对图片进行文本检测 + 识别，返回检测框列表和全图识别记录。

    PaddleOCR 3.7 的 predict() 一次调用同时返回检测和识别结果。
    本函数提取：
      - det_boxes → 用于版面分析（bbox 高度分类）
      - ocr_records → 用于手写识别（匹配 handwriting_bbox）

    参数:
        img: RGB numpy array (H, W, 3) 或 PIL Image 或文件路径
        page_idx: 页码（用于结果标记）

    返回:
        (det_boxes, ocr_records)

        det_boxes: 检测框列表
        [
            {
                "page": 0,
                "bbox_pixel": (x0, y0, x2, y2),  # 外接矩形（整数像素）
                "confidence": 0.9,                # 检测置信度（固定0.9，PaddleOCR 3.7不直接提供）
                "polygon": [[x1,y1],[x2,y2],[x3,y3],[x4,y4]],  # 原始多边形
            },
            ...
        ]

        ocr_records: 全图识别结果列表（与 dt_polys 1:1 对应）
        [
            {
                "rec_text": "春眠不觉晓",
                "rec_score": 0.95,
                "rec_bbox": (x0, y0, x1, y1),  # 识别框 [x0, y0, x1, y1]
                "page": 0,
            },
            ...
        ]
    """
    # 统一转 numpy array
    if not isinstance(img, np.ndarray):
        from PIL import Image
        img = np.array(Image.open(img) if isinstance(img, str) else img)

    detector = _get_detector()
    raw_pages = detector.predict(img)

    det_boxes: list[dict] = []
    ocr_records: list[dict] = []

    for page_result in raw_pages:
        data = getattr(page_result, 'json', None) or {}
        res = data.get('res')
        if res is None:
            logger.warning("json.res is None, skip page")
            continue

        # 提取检测框
        dt_polys = res.get('dt_polys', [])
        # 提取识别结果
        rec_texts = res.get('rec_texts', []) or []
        rec_scores = res.get('rec_scores', []) or []
        rec_boxes = res.get('rec_boxes', []) or []

        if not isinstance(dt_polys, list) or len(dt_polys) == 0:
            logger.warning("dt_polys 为空或格式异常, skip page")
            continue

        logger.info(
            "从 json.res 中提取 %d 个检测框, %d 条识别文本",
            len(dt_polys), len(rec_texts),
        )

        # 遍历 dt_polys（检测框列表）
        for i, poly in enumerate(dt_polys):
            # poly: [[x1,y1],[x2,y2],[x3,y3],[x4,y4]] — numpy array shape=(4,2)
            xs = [int(p[0]) for p in poly]
            ys = [int(p[1]) for p in poly]
            x0, y0 = min(xs), min(ys)
            x2, y2 = max(xs), max(ys)

            det_boxes.append({
                "page": page_idx,
                "bbox_pixel": (x0, y0, x2, y2),
                "confidence": 0.9,
                "polygon": poly.tolist() if hasattr(poly, 'tolist') else list(poly),
            })

            # 对应的识别结果（如果存在）
            if i < len(rec_texts) and rec_texts[i]:
                # rec_boxes[i] 为 [x0, y0, x1, y1]
                rec_box = rec_boxes[i] if i < len(rec_boxes) else None
                ocr_records.append({
                    "rec_text": rec_texts[i],
                    "rec_score": float(rec_scores[i]) if i < len(rec_scores) else 0.0,
                    "rec_bbox": tuple(rec_box) if rec_box is not None else (x0, y0, x2, y2),
                    "page": page_idx,
                })

    logger.info(
        "第 %d 页: 检测到 %d 个文本区域, %d 条识别记录",
        page_idx + 1, len(det_boxes), len(ocr_records),
    )
    return det_boxes, ocr_records