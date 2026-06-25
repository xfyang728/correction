"""
PDF 批注生成 — 在原扫描 PDF 上叠加红圈/绿勾/橙三角。

使用 pdfplumber 读取页面图片实际位置，
reportlab 绘制批注图层，PyPDF2 合并。

支持两种标记模式：
  - 逐字模式（默认）：对每个字独立画勾/圈/三角
  - 按题模式：正确题号旁画大绿✓，错误字保留红圈/橙三角
"""

import io
import logging
from pathlib import Path

import pdfplumber
from PyPDF2 import PdfReader, PdfWriter
from reportlab.pdfgen import canvas

logger = logging.getLogger("annotator")


def _pixel_to_page(x_pixel: float, y_pixel: float,
                    ocr_img_w: int, ocr_img_h: int,
                    page_width: float, page_height: float) -> tuple[float, float]:
    """将 OCR 像素坐标转换为 PDF 页面点坐标。"""
    x_page = x_pixel / ocr_img_w * page_width
    y_page = page_height - (y_pixel / ocr_img_h * page_height)
    return x_page, y_page


def _draw_checkmark(can: canvas.Canvas, x: float, y: float, size: float):
    """绘制绿勾 ✓。"""
    can.setStrokeColorRGB(0, 0.6, 0)
    can.setLineWidth(2.5)
    # 左上 → 中下
    can.line(x - size * 0.7, y + size * 0.2,
             x - size * 0.2, y - size * 0.5)
    # 中下 → 右上
    can.line(x - size * 0.2, y - size * 0.5,
             x + size * 0.8, y + size * 0.6)


def _draw_wrong_circle(can: canvas.Canvas, x: float, y: float,
                       radius: float, conf: float):
    """绘制错误红圈。"""
    radius = min(radius, 18)
    can.setStrokeColorRGB(1, 0, 0)
    can.setLineWidth(2)
    can.circle(x, y, radius)
    can.setFont("Helvetica", 6)
    can.drawString(x + radius + 2, y - 3, f"{conf:.0%}")


def _draw_uncertain_triangle(can: canvas.Canvas, x: float, y: float, radius: float):
    """绘制存疑橙三角。"""
    radius = min(radius, 18)
    can.setStrokeColorRGB(1, 0.6, 0)
    can.setLineWidth(2)
    can.setFillColorRGB(1, 0.8, 0.2)
    r2 = radius * 0.8
    p = can.beginPath()
    p.moveTo(x, y + r2)
    p.lineTo(x - r2, y - r2 * 0.6)
    p.lineTo(x + r2, y - r2 * 0.6)
    p.close()
    can.drawPath(p, fill=1, stroke=1)


def annotate(original_pdf: str, graded_results: list[dict],
             output_dir: str | None = None,
             question_summary: dict[int, dict] | None = None) -> str:
    """
    在 PDF 上叠加批注。

    参数:
        original_pdf: 原始 PDF 路径
        graded_results: grader 的输出
        output_dir: 输出目录，默认同目录
        question_summary: 按题汇总信息（可选），来自 grader.summarize_by_question()
                         启用按题标记模式：
                         - 正确题目：在题号旁画大绿✓
                         - 错误/存疑字：保留红圈/橙三角
                         - 正确字：不画单字勾（减少视觉噪音）

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

    # 构建快速查找: 哪些 question_idx 是全部正确的
    # {page: {q_idx: True/False}}
    all_correct_map: dict[int, set[int]] = {}
    # 逐字标记题（看拼音写词语等）: {page: set[q_idx]}
    per_char_map: dict[int, set[int]] = {}
    if question_summary:
        for page_idx, questions in question_summary.items():
            all_correct_map[page_idx] = {
                q_idx for q_idx, qs in questions.items()
                if qs["all_correct"] and qs.get("question_type", "default") != "per_char"
            }
            per_char_map[page_idx] = {
                q_idx for q_idx, qs in questions.items()
                if qs.get("question_type") == "per_char"
            }

    with pdfplumber.open(original_pdf) as pdf:
        for page_num in range(len(pdf.pages)):
            page = pdf.pages[page_num]

            # 收集本页批注
            page_results = [r for r in graded_results if r["page"] == page_num]

            # ---- 如果本页无批注结果，直接添加原始页 ----
            if not page_results:
                page_obj = pdf_reader.pages[page_num]
                pdf_writer.add_page(page_obj)
                continue

            # ---- 创建批注图层 ----
            packet = io.BytesIO()
            can = canvas.Canvas(packet, pagesize=(page.width, page.height))

            # 判断该页是否启用按题模式
            use_question_mode = (
                question_summary is not None
                and page_num in question_summary
            )

            # 找出本页全部正确的题号集合
            page_correct_questions = all_correct_map.get(page_num, set())
            # 本页逐字标记题集合（看拼音写词语等）
            page_per_char_questions = per_char_map.get(page_num, set())

            for r in page_results:
                bbox = r["bbox_pixel"]
                status = r["status"]
                conf = r["confidence"]
                q_idx = r.get("question_idx")

                ocr_img_w = r["img_pixel_w"]
                ocr_img_h = r["img_pixel_h"]

                # ---- 坐标有效性验证 ----
                # bbox 应在 [0, img_w] × [0, img_h] 范围内
                bx0, by0, bx2, by2 = bbox
                if not (0 <= bx0 <= ocr_img_w and 0 <= bx2 <= ocr_img_w and
                        0 <= by0 <= ocr_img_h and 0 <= by2 <= ocr_img_h):
                    logger.warning("坐标越界: bbox=%s, img_size=%dx%d, skip",
                                   bbox, ocr_img_w, ocr_img_h)
                    continue

                # bbox 中心（用于红圈/橙三角，钳制到有效范围）
                cx_pixel = max(0, min((bbox[0] + bbox[2]) / 2, ocr_img_w))
                cy_pixel = max(0, min((bbox[1] + bbox[3]) / 2, ocr_img_h))
                # bbox 右下角（用于绿勾，钳制到有效范围）
                br_x_pixel = max(0, min(bbox[2], ocr_img_w))
                br_y_pixel = max(0, min(bbox[3], ocr_img_h))
                bw_pixel = bbox[2] - bbox[0]
                bh_pixel = bbox[3] - bbox[1]
                # 最小半径限制（避免 bbox 过小时标记不可见）
                mark_radius = max(max(bw_pixel, bh_pixel) * 0.6, 8)

                x_page, y_page = _pixel_to_page(
                    cx_pixel, cy_pixel,
                    ocr_img_w, ocr_img_h,
                    page.width, page.height,
                )
                # 绿勾锚点：字符右下角
                ck_x_page, ck_y_page = _pixel_to_page(
                    br_x_pixel, br_y_pixel,
                    ocr_img_w, ocr_img_h,
                    page.width, page.height,
                )
                scale = page.width / ocr_img_w if ocr_img_w > 0 else 1
                mark_r = mark_radius * scale

                # 半径上下限：8~18pt
                mark_r = max(min(mark_r, 18), 8)

                # ---- 逐字标记题（看拼音写词语等）----
                # 无论对错都逐字标记：正确画小绿勾，错误画红圈，存疑画橙三角
                if q_idx is not None and q_idx in page_per_char_questions:
                    if status == "wrong":
                        _draw_wrong_circle(can, x_page, y_page, mark_r, conf)
                    elif status == "uncertain":
                        _draw_uncertain_triangle(can, x_page, y_page, mark_r)
                    else:
                        _draw_checkmark(can, ck_x_page, ck_y_page, mark_r * 0.7)

                # ---- 按题模式 ----
                elif use_question_mode:
                    # 如果这个字属于全部正确的题 → 跳过逐字标记（题号旁已画大绿✓）
                    if q_idx is not None and q_idx in page_correct_questions:
                        continue

                    # 否则：按当前逻辑标记（错误红圈 / 存疑橙三角）
                    if status == "wrong":
                        _draw_wrong_circle(can, x_page, y_page, mark_r, conf)
                    elif status == "uncertain":
                        _draw_uncertain_triangle(can, x_page, y_page, mark_r)
                    # correct 状态的单字在题全对时已被跳过，这里不执行

                # ---- 逐字模式（默认） ----
                else:
                    if status == "wrong":
                        _draw_wrong_circle(can, x_page, y_page, mark_r, conf)
                    elif status == "uncertain":
                        _draw_uncertain_triangle(can, x_page, y_page, mark_r)
                    else:
                        _draw_checkmark(can, ck_x_page, ck_y_page, mark_r * 0.7)

            # ---- 按题模式：绘制题号旁的绿色大对号 ----
            # 跳过逐字标记题（per_char 题即使全对也不画大绿✓，而是逐字画小绿勾）
            if use_question_mode:
                for _q_idx, q_data in question_summary[page_num].items():
                    if not q_data["all_correct"]:
                        continue
                    if q_data.get("question_type") == "per_char":
                        continue
                    # 对勾位置：x 用答案末字右边缘，y 用题号垂直中心（保证竖向对齐）
                    answer_bbox = q_data.get("answer_bbox")
                    marker_bbox = q_data.get("marker_bbox")

                    # 使用第一项的 img 尺寸进行坐标转换
                    first_result = page_results[0] if page_results else None
                    if not first_result:
                        continue
                    ocr_img_w = first_result["img_pixel_w"]
                    ocr_img_h = first_result["img_pixel_h"]

                    if answer_bbox:
                        mx = max(0, min(answer_bbox[2], ocr_img_w))
                    elif marker_bbox:
                        mx = marker_bbox[2] + (marker_bbox[2] - marker_bbox[0]) * 0.3
                    else:
                        continue

                    if marker_bbox:
                        my = (marker_bbox[1] + marker_bbox[3]) / 2
                    elif answer_bbox:
                        my = (answer_bbox[1] + answer_bbox[3]) / 2
                    else:
                        continue

                    q_mark_size = 14

                    qx_page, qy_page = _pixel_to_page(
                        mx, my,
                        ocr_img_w, ocr_img_h,
                        page.width, page.height,
                    )

                    # 固定大小 14pt，不随 bbox 缩放
                    q_mark_size = 14

                    can.setStrokeColorRGB(0, 0.6, 0)
                    can.setLineWidth(3)
                    # 左上 → 中下
                    can.line(
                        qx_page - q_mark_size * 0.7, qy_page + q_mark_size * 0.2,
                        qx_page - q_mark_size * 0.2, qy_page - q_mark_size * 0.5,
                    )
                    # 中下 → 右上
                    can.line(
                        qx_page - q_mark_size * 0.2, qy_page - q_mark_size * 0.5,
                        qx_page + q_mark_size * 0.8, qy_page + q_mark_size * 0.6,
                    )

            can.save()
            packet.seek(0)

            # ---- 合并原页 + 批注图层 ----
            overlay = PdfReader(packet)
            page_obj = pdf_reader.pages[page_num]
            page_obj.merge_page(overlay.pages[0])
            pdf_writer.add_page(page_obj)

    with open(output_path, "wb") as f:
        pdf_writer.write(f)

    total_marks = len(graded_results)
    mode_str = "按题" if question_summary else "逐字"
    logger.info("批注 PDF 已保存: %s (%s模式, %d 个标记)",
                output_path, mode_str, total_marks)
    return output_path
