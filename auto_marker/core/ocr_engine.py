"""
OCR 引擎 — EasyOCR 封装。

将 PDF 每页转为图片，调用 EasyOCR 识别手写汉字，
返回结构化结果（页号、坐标、字符、置信度、图片尺寸）。
"""

import logging
import tempfile
from pathlib import Path

import pdfplumber
import fitz  # PyMuPDF — 无需 poppler，纯 Python

logger = logging.getLogger("ocr")


# 全局单例（EasyOCR 初始化较慢，只做一次）
_ocr_instance = None


def _get_ocr(use_gpu: bool = False):
    global _ocr_instance
    if _ocr_instance is None:
        import easyocr
        logger.info("正在初始化 EasyOCR（首次加载较慢）...")
        # gpu=True 需要 CUDA，默认用 CPU
        _ocr_instance = easyocr.Reader(["ch_sim", "en"], gpu=use_gpu)
        logger.info("EasyOCR 就绪")
    return _ocr_instance


def ocr_pdf(pdf_path: str, use_gpu: bool = False, dpi: int = 200) -> list[dict]:
    """
    对 PDF 逐页 OCR。

    返回:
        [
            {
                "page": 0,
                "bbox_pixel": (x0, y0, x2, y2),   # 像素坐标
                "char": "好",
                "confidence": 0.95,
                "img_pixel_w": 1654,
                "img_pixel_h": 2339,
            },
            ...
        ]
    """
    reader = _get_ocr(use_gpu)

    logger.info("开始 OCR: %s (dpi=%d)", pdf_path, dpi)
    results = []

    # 使用 PyMuPDF 渲染 PDF 页面为图片（无需 poppler）
    doc = fitz.open(pdf_path)
    num_pages = len(doc)
    logger.info("PDF 共 %d 页", num_pages)

    for page_idx in range(num_pages):
        page = doc.load_page(page_idx)

        # 渲染页面为 RGB 图片
        zoom = dpi / 72  # PDF 默认 72 DPI
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat, colorspace=fitz.csRGB)
        img_w, img_h = pix.width, pix.height
        logger.debug("第 %d 页: %dx%d 像素 (dpi=%d)", page_idx + 1, img_w, img_h, dpi)

        # 转为 PIL Image
        from PIL import Image
        import io
        img = Image.open(io.BytesIO(pix.tobytes("png")))

        # 保存为临时文件供 EasyOCR 读取
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            tmp_path = tmp.name
            img.save(tmp_path, "PNG")

        try:
            # EasyOCR 返回: [ ([[x0,y0],[x1,y1],[x2,y2],[x3,y3]], text, confidence), ... ]
            raw = reader.readtext(tmp_path)
        finally:
            Path(tmp_path).unlink(missing_ok=True)

        if not raw:
            logger.debug("第 %d 页无识别结果", page_idx + 1)
            continue

        for poly, text, confidence in raw:
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

    doc.close()

    logger.info("OCR 完成: %d 个字符", len(results))
    return results
