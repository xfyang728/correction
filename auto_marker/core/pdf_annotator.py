"""
PDF 批注生成 — 在原扫描 PDF 上叠加红圈/绿勾/橙三角。

使用 pdfplumber 读取页面图片实际位置，
reportlab 绘制批注图层，PyPDF2 合并。
"""

import io
import logging
from pathlib import Path

import pdfplumber
from reportlab.pdfgen import canvas
from PyPDF2 import PdfReader, PdfWriter

logger = logging.getLogger("annotator")


def annotate(original_pdf: str, graded_results: list[dict],
             output_dir: str | None = None) -> str:
    """
    在 PDF 上叠加批注。

    参数:
        original_pdf: 原始 PDF 路径
        graded_results: grader 的输出
        output_dir: 输出目录，默认同目录

    返回:
        批注后的 PDF 路径
    """
    if not graded_results:
        logger.warning("无批注结果，跳过批注生成")
        return original_pdf

    src = Path(original_pdf)
    out_dir = Path(output_dir) if output_dir else src.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    output_path = str(out_dir / f"{src.stem}_annotated{src.suffix}")

    pdf_reader = PdfReader(original_pdf)
    pdf_writer = PdfWriter()

    with pdfplumber.open(original_pdf) as pdf:
        for page_num in range(len(pdf.pages)):
            page = pdf.pages[page_num]

            # ---- 获取本页图片在页面上的实际位置 ----
            img_info = None
            img_page_w, img_page_h = page.width, page.height
            if page.images:
                img_info = page.images[0]
                img_page_w = img_info["x1"] - img_info["x0"]
                img_page_h = img_info["y1"] - img_info["y0"]

            # 收集本页批注
            page_results = [r for r in graded_results if r["page"] == page_num]

            # ---- 创建批注图层 ----
            packet = io.BytesIO()
            can = canvas.Canvas(packet, pagesize=(page.width, page.height))

            for r in page_results:
                bbox = r["bbox_pixel"]  # (x0, y0, x2, y2) 像素坐标
                x_img, y_img = bbox[0], bbox[1]
                status = r["status"]
                conf = r["confidence"]

                # 像素坐标 → 页面点坐标
                ocr_img_w = r["img_pixel_w"]
                ocr_img_h = r["img_pixel_h"]
                if img_info:
                    x_page = img_info["x0"] + (x_img / ocr_img_w) * img_page_w
                    y_page = (img_page_h - (y_img / ocr_img_h) * img_page_h) + img_info["y0"]
                else:
                    # 无图片信息时直接按页比例换算
                    x_page = x_img / ocr_img_w * page.width
                    y_page = page.height - (y_img / ocr_img_h * page.height)

                # 根据状态绘制不同标记
                if status == "wrong":
                    # 红圈
                    can.setStrokeColorRGB(1, 0, 0)
                    can.setLineWidth(2)
                    can.circle(x_page, y_page, 10)
                    # 在圈旁标注置信度
                    can.setFont("Helvetica", 6)
                    can.drawString(x_page + 12, y_page - 3, f"{conf:.0%}")

                elif status == "uncertain":
                    # 橙三角
                    can.setStrokeColorRGB(1, 0.6, 0)
                    can.setLineWidth(2)
                    can.setFillColorRGB(1, 0.8, 0.2)
                    can.beginPath()
                    can.moveTo(x_page, y_page + 8)
                    can.lineTo(x_page - 7, y_page - 6)
                    can.lineTo(x_page + 7, y_page - 6)
                    can.closePath()
                    can.fill()
                    can.stroke()

                else:  # correct
                    # 绿勾
                    can.setStrokeColorRGB(0, 0.6, 0)
                    can.setLineWidth(2)
                    can.line(x_page - 6, y_page - 1, x_page - 2, y_page + 5)
                    can.line(x_page - 2, y_page + 5, x_page + 7, y_page - 6)

            can.save()
            packet.seek(0)

            # ---- 合并原页 + 批注图层 ----
            overlay = PdfReader(packet)
            page_obj = pdf_reader.pages[page_num]
            page_obj.merge_page(overlay.pages[0])
            pdf_writer.add_page(page_obj)

    with open(output_path, "wb") as f:
        pdf_writer.write(f)

    logger.info("批注 PDF 已保存: %s (%d 个标记)", output_path, len(graded_results))
    return output_path
