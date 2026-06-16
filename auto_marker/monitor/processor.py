"""
流水线调度 — 从文件名解析元数据，依次执行：OCR → 比对 → 批注 → 打印 → 归档。
"""

import re
import shutil
import logging
from pathlib import Path

logger = logging.getLogger("processor")

# 文件名契约: {班级}_{日期}_{序号}.pdf
PATTERN = re.compile(r"(\w+)_(\d{4}-\d{2}-\d{2})_(\d+)\.pdf$")


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
    """完整批改流水线。"""
    path = Path(pdf_path)
    if not path.exists():
        logger.error("文件不存在: %s", pdf_path)
        return

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

    # 2. OCR
    from core.ocr_engine import ocr_pdf
    ocr_results = ocr_pdf(str(path))
    if not ocr_results:
        logger.warning("OCR 无结果，放入 failed")
        _move_to(path, "failed")
        return

    # 3. 比对
    from core.grader import grade
    graded = grade(ocr_results, answers)

    # 4. 保存到数据库
    try:
        from db.crud import save_task
        save_task(str(path), graded)
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
