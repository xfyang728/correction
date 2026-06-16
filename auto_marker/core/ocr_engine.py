"""
OCR 引擎 — PaddleOCR 封装。

使用 ch_PP-OCRv4 轻量模型，专为中文手写体优化。
支持直接对 numpy array 推理（避免临时文件 IO），也支持 PDF 路径。
"""

import logging
from pathlib import Path

import fitz  # PyMuPDF
import numpy as np
from PIL import Image

logger = logging.getLogger("ocr")


# 全局单例（PaddleOCR 初始化较慢，只做一次）
_ocr_instance = None


def _get_ocr(use_gpu: bool = False):
    """获取或初始化 PaddleOCR 实例。"""
    global _ocr_instance
    if _ocr_instance is None:
        logger.info("正在初始化 PaddleOCR（ch_PP-OCRv4，首次加载将下载模型）...")
        try:
            from paddleocr import PaddleOCR
        except ImportError:
            raise ImportError("请安装 paddleocr: pip install paddleocr")
        _ocr_instance = PaddleOCR(
            use_angle_cls=False,  # 非弯曲文本，无需角度分类
            lang='ch',
            use_gpu=False,
            show_log=False,
        )
        logger.info("PaddleOCR 就绪")
    return _ocr_instance


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
                "img_pixel_w": 1654,
                "img_pixel_h": 2339,
            },
            ...
        ]
    """
    reader = _get_ocr()
    img_h, img_w = img.shape[:2]
    logger.debug("OCR 第 %d 页: %dx%d 像素", page_idx + 1, img_w, img_h)

    # PaddleOCR 2.x 需要文件路径，保存临时文件
    from PIL import Image as PILImage
    pil_img = PILImage.fromarray(img)

    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        tmp_path = tmp.name
        pil_img.save(tmp_path, "PNG")

    try:
        # PaddleOCR 2.x 返回: [ [[x0,y0],[x1,y1],[x2,y2],[x3,y3]], (text, confidence) ]
        raw = reader.ocr(tmp_path, cls=False)
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    results: list[dict] = []
    if raw is None or len(raw) == 0:
        logger.debug("第 %d 页无识别结果", page_idx + 1)
        return results

    # PaddleOCR 2.x 返回: [ page_results ]
    # page_results = [ [[x0,y0],...], (text, confidence) ], ... ]
    page_results = raw[0] if isinstance(raw[0], list) else raw
    if page_results is None:
        return results

    for item in page_results:
        poly, (text, confidence) = item
        xs = [p[0] for p in poly]
        ys = [p[1] for p in poly]
        x0, y0 = min(xs), min(ys)
        x2, y2 = max(xs), max(ys)

        results.append({
            "page": page_idx,
            "bbox_pixel": (int(x0), int(y0), int(x2), int(y2)),
            "char": text.strip(),
            "confidence": confidence,
            "img_pixel_w": img_w,
            "img_pixel_h": img_h,
        })

    logger.debug("第 %d 页: 识别到 %d 个字符", page_idx + 1, len(results))
    return results


def ocr_pdf(pdf_path: str, use_gpu: bool = False, dpi: int = 200) -> list[dict]:
    """对 PDF 逐页 OCR（保持原有接口，供其他模块调用）。

    参数:
        pdf_path: PDF 文件路径
        use_gpu: 是否使用 GPU（默认 False）
        dpi: 渲染 DPI（默认 200）

    返回:
        与 ocr_image() 相同的结构
    """
    _get_ocr(use_gpu)

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