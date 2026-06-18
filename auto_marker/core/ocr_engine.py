"""
OCR 引擎 — PaddleOCR 3.6 (PP-OCRv6 Medium) 封装。

使用 PP-OCRv6 Medium 轻量高精度模型，支持手写体识别。
模型自动下载到 `model/` 目录下。
PP-OCRv6 为整行识别，本模块负责拆分为单字并保留 bbox。
"""

import logging
import os
from pathlib import Path

import fitz  # PyMuPDF
import numpy as np
from PIL import Image

logger = logging.getLogger("ocr")

# 模型存储根目录
MODEL_ROOT = Path(__file__).resolve().parent.parent / "model"


# 全局单例（PaddleOCR 初始化较慢，只做一次）
_ocr_instance = None


def _get_ocr():
    """获取或初始化 PaddleOCR 3.6 实例（PP-OCRv6 Medium）。"""
    global _ocr_instance
    if _ocr_instance is None:
        # 确保模型目录存在
        MODEL_ROOT.mkdir(parents=True, exist_ok=True)

        logger.info("正在初始化 PaddleOCR 3.6（PP-OCRv6 Medium，首次加载将下载模型）...")
        logger.info("模型下载路径: %s", MODEL_ROOT)

        try:
            # 设置 PaddleX 模型下载根目录，确保下载到 model/
            os.environ["PADDLE_PDX_CACHE_HOME"] = str(MODEL_ROOT)
            from paddleocr import PaddleOCR
        except ImportError:
            raise ImportError("请安装 paddleocr: pip install paddleocr>=3.6.0")

        _ocr_instance = PaddleOCR(
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
        )
        logger.info("PaddleOCR 3.6 就绪（PP-OCRv6 Medium）")
    return _ocr_instance


def _is_chinese_char(ch: str) -> bool:
    """判断是否为中文字符。"""
    code = ord(ch)
    return (
        0x4E00 <= code <= 0x9FFF   # CJK 统一汉字
        or 0x3400 <= code <= 0x4DBF  # CJK 扩展 A
        or 0xF900 <= code <= 0xFAFF  # CJK 兼容汉字
    )


def _split_line_to_chars(text: str, line_box, confidence: float, page_idx: int,
                         img_w: int, img_h: int) -> list[dict]:
    """将整行识别结果拆分为单字，bbox 按等宽切分。

    PP-OCRv5 返回整行文本（如"西湖莲叶"），需要拆成单字
    并为每个字估算 bbox 坐标。
    """
    x0, y0, x2, y2 = line_box
    line_w = x2 - x0
    line_h = y2 - y0

    # 提取所有中文字符
    chars = [ch for ch in text if _is_chinese_char(ch)]
    if not chars:
        return []

    # 按字符数等宽切分 bbox
    char_w = line_w / len(chars)
    results = []
    for i, ch in enumerate(chars):
        cx0 = int(x0 + i * char_w)
        cx2 = int(x0 + (i + 1) * char_w)
        results.append({
            "page": page_idx,
            "bbox_pixel": (cx0, int(y0), cx2, int(y2)),
            "char": ch,
            "confidence": confidence,
            "img_pixel_w": img_w,
            "img_pixel_h": img_h,
        })
    return results


def ocr_image(img: np.ndarray, page_idx: int = 0) -> list[dict]:
    """对单张图片（numpy array）进行 OCR。

    参数:
        img: RGB numpy array (H, W, 3)
        page_idx: 页码（用于结果标记）

    返回:
        [
            {
                "page": page_idx,
                "bbox_pixel": (x0, y0, x2, y2),   # 像素坐标
                "char": "好",
                "confidence": 0.95,
                "img_pixel_w": 2481,
                "img_pixel_h": 3508,
            },
            ...
        ]
    """
    reader = _get_ocr()
    img_h, img_w = img.shape[:2]
    logger.debug("OCR 第 %d 页: %dx%d 像素", page_idx + 1, img_w, img_h)

    # PP-OCRv6 使用 predict() API，直接接受 numpy array
    raw_results = reader.predict(input=img)

    results: list[dict] = []
    for res in raw_results:
        res_data = res.json.get("res", {})
        rec_texts = res_data.get("rec_texts", [])
        rec_scores = res_data.get("rec_scores", [])
        rec_boxes = res_data.get("rec_boxes", [])  # [x0, y0, x1, y1]

        for text, score, box in zip(rec_texts, rec_scores, rec_boxes):
            text = text.strip()
            if not text:
                continue

            # 拆分整行为单字
            char_results = _split_line_to_chars(
                text, box, score, page_idx, img_w, img_h
            )
            results.extend(char_results)

    logger.debug("第 %d 页: 识别到 %d 个字符", page_idx + 1, len(results))
    return results


def ocr_pdf(pdf_path: str, use_gpu: bool = False, dpi: int = 200) -> list[dict]:
    """对 PDF 逐页 OCR（保持原有接口，供其他模块调用）。

    参数:
        pdf_path: PDF 文件路径
        use_gpu: 是否使用 GPU（默认 False，PP-OCRv6 自动检测）
        dpi: 渲染 DPI（默认 200，平衡速度与精度）

    返回:
        与 ocr_image() 相同的结构
    """
    _get_ocr()

    logger.info("开始 OCR: %s (dpi=%d)", pdf_path, dpi)
    results = []

    doc = fitz.open(pdf_path)
    num_pages = len(doc)
    logger.info("PDF 共 %d 页", num_pages)

    for page_idx in range(num_pages):
        page = doc.load_page(page_idx)
        zoom = dpi / 72
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat, colorspace=fitz.csRGB)

        img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
        img_np = np.array(img)

        page_results = ocr_image(img_np, page_idx)
        results.extend(page_results)

    doc.close()
    logger.info("OCR 完成: %d 个字符", len(results))
    return results