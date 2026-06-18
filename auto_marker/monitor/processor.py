"""
流水线调度 — 从文件名解析元数据，依次执行：预处理 → OCR → 版面分析 → 比对 → 批注 → 打印 → 归档。
"""

import io
import re
import shutil
import time
import logging
from pathlib import Path

import fitz
import numpy as np
from PIL import Image

logger = logging.getLogger("processor")

# 文件名契约: {班级}_{日期}_{序号}.pdf
PATTERN = re.compile(r"(\w+)_(\d{4}-\d{2}-\d{2})_(\d+)\.pdf$")

# 默认渲染 DPI（server 模型精度足够，200 DPI 平衡速度与精度）
OCR_DPI = 200

# 低置信度阈值 — 低于此值计为"低置信度字"
LOW_CONFIDENCE_THRESHOLD = 0.60
# 低置信度字占比超过此值则自动标记"需人工复核"
REVIEW_TRIGGER_RATIO = 0.30


def load_answers(class_name: str, date_str: str) -> list[str]:
    """按班级/日期加载标准答案。

    优先级:
        1. 数据库（Web UI 保存的答案）
        2. answers.txt 文件（手动编辑）
    """
    # 1. 尝试从数据库读取（Web UI 保存的答案优先）
    try:
        from db.crud import get_answer
        db_text = get_answer(class_name, date_str)
        if db_text:
            logger.info("从数据库加载答案（班级=%s, 日期=%s）: %d 字",
                        class_name, date_str, len(db_text))
            return list(db_text.replace("\n", "").replace(" ", ""))
    except Exception as e:
        logger.warning("数据库读取答案失败，回退到文件: %s", e)

    # 2. 回退到 answers.txt 文件
    answers_file = Path(__file__).resolve().parent.parent / "answers.txt"
    if not answers_file.exists():
        logger.warning("answers.txt 不存在，使用空答案集")
        return []
    text = answers_file.read_text(encoding="utf-8").strip()
    if not text:
        return []
    return list(text.replace("\n", "").replace(" ", ""))


def process_pdf(pdf_path: str) -> None:
    """完整批改流水线：预处理 → OCR → 版面分析 → 比对 → 批注 → 打印 → 归档。"""
    path = Path(pdf_path)
    if not path.exists():
        logger.error("文件不存在: %s", pdf_path)
        return

    t_start = time.time()

    # 0. 从文件名解析元数据
    m = PATTERN.search(path.name)
    if not m:
        logger.warning("文件名不符合约定 %s，放入 backup 跳过", path.name)
        _move_to(path, "backup")
        return
    class_name, date_str, seq = m.groups()
    logger.info("开始批改: %s | 班级=%s 日期=%s 序号=%s",
                path.name, class_name, date_str, seq)

    # 1. 加载答案
    answers = load_answers(class_name, date_str)
    logger.info("答案长度: %d 字", len(answers))

    # ----------------------------------------------------------------
    # 2. 四步流水线：渲染 → 预处理 → OCR → 版面分析
    # ----------------------------------------------------------------
    from core.image_processor import preprocess_image
    from core.ocr_engine import ocr_image
    from core.layout_analyzer import extract_student_answers

    doc = fitz.open(str(path))
    num_pages = len(doc)
    logger.info("PDF 共 %d 页", num_pages)

    all_ocr_results: list[dict] = []
    page_img_sizes: list[tuple[int, int]] = []  # 每页 (w, h)

    for page_idx in range(num_pages):
        page = doc.load_page(page_idx)
        zoom = OCR_DPI / 72
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat, colorspace=fitz.csRGB)

        # 转为 PIL Image
        img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
        page_img_sizes.append((img.width, img.height))

        # Step A: 图像预处理（去噪、二值化、增强、校正）
        processed_img = preprocess_image(img)
        logger.debug("第 %d 页: 预处理完成 (%dx%d)", page_idx + 1, img.width, img.height)

        # Step B: OCR（对预处理后的图片进行文字识别）
        img_np = np.array(processed_img)
        page_results = ocr_image(img_np, page_idx)
        all_ocr_results.extend(page_results)

    doc.close()

    if not all_ocr_results:
        logger.warning("OCR 无结果，放入 failed")
        _move_to(path, "failed")
        return

    # Step C: 版面分析 → 提取学生手写答案
    # 取最后一页的尺寸作为版面分析参考（单页用第一页也可）
    ref_w, ref_h = page_img_sizes[0] if page_img_sizes else (0, 0)
    student_answers = extract_student_answers(all_ocr_results, ref_w, ref_h)

    if not student_answers:
        logger.warning("未提取到学生手写答案，放入 failed")
        _move_to(path, "failed")
        return

    # ----------------------------------------------------------------
    # 3. 比对（按页分组，每页独立与答案做 DP 对齐）
    # ----------------------------------------------------------------
    from core.grader import grade

    # 按页分组（每页 = 一个学生的独立答卷）
    from collections import defaultdict
    page_answers = defaultdict(list)
    for r in student_answers:
        page_answers[r["page"]].append(r)

    # 每页独立与答案做 DP 对齐
    graded = []
    for page_idx in sorted(page_answers.keys()):
        page_graded = grade(page_answers[page_idx], answers)
        graded.extend(page_graded)
        logger.info("第 %d 页: OCR %d 字 → 对齐 %d 字",
                     page_idx + 1, len(page_answers[page_idx]), len(page_graded))

    logger.info("总对齐结果: %d 字（%d 页）", len(graded), len(page_answers))

    # ----------------------------------------------------------------
    # 3.5 可观测性指标计算（按页加权平均）
    # ----------------------------------------------------------------
    confidences = [r["confidence"] for r in graded if r["confidence"] is not None]
    avg_conf = sum(confidences) / len(confidences) if confidences else 0.0
    low_conf_count = sum(1 for c in confidences if c < LOW_CONFIDENCE_THRESHOLD)
    low_conf_ratio = low_conf_count / len(confidences) if confidences else 0.0
    process_time = time.time() - t_start
    needs_review = low_conf_ratio > REVIEW_TRIGGER_RATIO

    logger.info(
        "📊 可观测性: 平均置信度=%.1f%% | 低置信度字=%d/%d (%.1f%%) | 耗时=%.1fs | %s",
        avg_conf * 100,
        low_conf_count,
        len(confidences),
        low_conf_ratio * 100,
        process_time,
        "⚠️ 需人工复核" if needs_review else "✅ 质量正常",
    )

    # 4. 保存到数据库（含可观测性指标）
    try:
        from db.crud import save_task
        save_task(
            str(path), graded,
            avg_confidence=avg_conf,
            low_conf_ratio=low_conf_ratio,
            process_time=process_time,
            needs_review=needs_review,
        )
        logger.info("任务已保存到数据库")
    except Exception as e:
        logger.warning("数据库保存失败（不影响后续）: %s", e)

    # 5. 生成批注 PDF
    from core.pdf_annotator import annotate
    output_dir = Path(__file__).resolve().parent.parent / "data" / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    annotated_path = annotate(str(path), graded, str(output_dir))
    logger.info("批注 PDF 已生成: %s", annotated_path)

    # 6. 打印
    try:
        from core.printer import print_pdf
        print_pdf(annotated_path)
        logger.info("打印任务已发送")
    except Exception as e:
        logger.warning("打印失败（可稍后手动打印）: %s", e)

    # 7. 归档
    _move_to(path, "backup")
    logger.info("✅ 批改完成: %s", path.name)


def _move_to(path: Path, subdir: str):
    backup_root = Path(__file__).resolve().parent.parent / "data" / subdir
    backup_root.mkdir(parents=True, exist_ok=True)
    dest = backup_root / path.name
    # 避免重名
    if dest.exists():
        stem = path.stem
        dest = backup_root / f"{stem}_{int(__import__('time').time())}{path.suffix}"
    shutil.move(str(path), str(dest))
