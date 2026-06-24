"""批改流水线服务 — 封装从 PDF 到归档的完整流程。

三阶段流水线：渲染 → 预处理 → 文本检测 → 版面分析与分离 → 手写识别 → 比对 → 批注 → 打印 → 归档。
"""

import hashlib
import logging
import pickle
import re
import shutil
import time
from collections import defaultdict
from pathlib import Path

import fitz
import numpy as np
from PIL import Image

logger = logging.getLogger("pipeline")

# 文件名契约: {班级}_{日期}_{序号}.pdf
PATTERN = re.compile(r"(\w+)_(\d{4}-\d{2}-\d{2})_(\d+)\.pdf$")

# 默认渲染 DPI（server 模型精度足够，200 DPI 平衡速度与精度）
OCR_DPI = 200

# 低置信度阈值 — 低于此值计为"低置信度字"
LOW_CONFIDENCE_THRESHOLD = 0.60
# 低置信度字占比超过此值则自动标记"需人工复核"
REVIEW_TRIGGER_RATIO = 0.30

# OCR 重试参数
_OCR_MAX_RETRIES = 2
_OCR_RETRY_DELAY = 2.0  # 秒

# OCR 结果缓存目录
_CACHE_DIR = Path(__file__).resolve().parent.parent / "data" / "cache"


def _retry_ocr(fn, description: str = "OCR"):
    """对 OCR 调用添加重试，仅对可重试异常（超时、OOM 等）重试。"""
    for attempt in range(_OCR_MAX_RETRIES + 1):
        try:
            return fn()
        except Exception as e:
            if attempt == _OCR_MAX_RETRIES:
                raise
            logger.warning("%s 第 %d 次重试（共 %d 次）: %s",
                           description, attempt + 1, _OCR_MAX_RETRIES, e)
            time.sleep(_OCR_RETRY_DELAY)


def _file_hash(path: Path) -> str:
    """计算文件 MD5 哈希（用于缓存 key）。"""
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_ocr_cache(file_hash: str, page_idx: int) -> tuple | None:
    """从磁盘加载缓存的 OCR 结果，未命中返回 None。"""
    cache_file = _CACHE_DIR / f"{file_hash}_p{page_idx}.pkl"
    if cache_file.exists():
        try:
            with open(cache_file, "rb") as f:
                return pickle.load(f)
        except Exception:
            logger.debug("缓存读取失败: %s", cache_file)
    return None


def _save_ocr_cache(file_hash: str, page_idx: int, data: tuple) -> None:
    """将 OCR 结果保存到磁盘缓存。"""
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = _CACHE_DIR / f"{file_hash}_p{page_idx}.pkl"
    try:
        with open(cache_file, "wb") as f:
            pickle.dump(data, f)
    except Exception:
        logger.debug("缓存写入失败: %s", cache_file)


def _move_to(path: Path, subdir: str) -> None:
    """将文件移动到 data/{subdir}/ 目录。"""
    backup_root = Path(__file__).resolve().parent.parent / "data" / subdir
    backup_root.mkdir(parents=True, exist_ok=True)
    dest = backup_root / path.name
    if dest.exists():
        stem = path.stem
        dest = backup_root / f"{stem}_{int(time.time())}{path.suffix}"
    shutil.move(str(path), str(dest))


class PipelineService:
    """批改流水线服务 — 封装从 PDF 到归档的完整流程。"""

    def load_answers(self, class_name: str, date_str: str) -> list[str]:
        """按班级/日期加载标准答案。

        优先级:
            1. 数据库（Web UI 保存的答案）
            2. answers.txt 文件（手动编辑）
        """
        from services.answer_service import AnswerService

        # 1. 尝试从数据库读取（Web UI 保存的答案优先）
        try:
            db_text = AnswerService.get_answer(class_name, date_str)
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

    def process_pdf(self, pdf_path: str) -> None:
        """完整批改流水线：预处理 → OCR → 版面分析 → 比对 → 批注 → 打印 → 归档。"""
        path = Path(pdf_path)
        if not path.exists():
            logger.error("文件不存在: %s", pdf_path)
            return

        t_start = time.time()

        # 0. 从文件名解析元数据
        class_name, date_str, seq = self._parse_metadata(path)
        if class_name is None:
            logger.warning("文件名不符合约定 %s，放入 backup 跳过", path.name)
            _move_to(path, "backup")
            return
        logger.info("开始批改: %s | 班级=%s 日期=%s 序号=%s",
                    path.name, class_name, date_str, seq)

        # 1. 加载答案
        answers = self.load_answers(class_name, date_str)
        logger.info("答案长度: %d 字", len(answers))

        # 2. 三阶段流水线
        all_results, question_regions = self._run_ocr_pipeline(path)

        if not all_results:
            logger.warning("未识别到任何手写内容，放入 failed")
            _move_to(path, "failed")
            return

        # 3. 比对评分 + 按题汇总 + 可观测性指标
        graded, question_summary, avg_conf, low_conf_ratio, process_time, needs_review = \
            self._grade_results(all_results, answers, question_regions, t_start)

        # 4-7. 保存数据库 + 生成批注 PDF + 打印 + 归档
        self._save_and_output(
            path, graded, question_summary,
            class_name, date_str, seq,
            avg_conf, low_conf_ratio, process_time, needs_review,
        )
        logger.info("✅ 批改完成: %s", path.name)

    def _parse_metadata(self, path: Path) -> tuple[str | None, str | None, str | None]:
        """从文件名解析元数据 (class_name, date_str, seq)。"""
        m = PATTERN.search(path.name)
        if not m:
            return (None, None, None)
        return m.groups()

    def _run_ocr_pipeline(self, path: Path) -> tuple[list[dict], dict[int, list[dict]]]:
        """三阶段流水线：渲染 → 预处理 → 文本检测 → 版面分析 → 手写识别。

        Returns:
            (all_results, question_regions)
        """
        from core.handwriting_recognizer import recognize_handwriting
        from core.image_processor import preprocess_image
        from core.layout_analyzer import analyze_layout
        from core.text_detector import detect_text

        doc = fitz.open(str(path))
        num_pages = len(doc)
        logger.info("PDF 共 %d 页", num_pages)

        file_hash = _file_hash(path)

        all_results: list[dict] = []
        question_regions: dict[int, list[dict]] = {}

        for page_idx in range(num_pages):
            page = doc.load_page(page_idx)
            zoom = OCR_DPI / 72
            mat = fitz.Matrix(zoom, zoom)
            pix = page.get_pixmap(matrix=mat, colorspace=fitz.csRGB)

            img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)

            # Step A: 图像预处理
            processed_img = preprocess_image(img)
            processed_np = np.array(processed_img)
            logger.debug("第 %d 页: 预处理完成 (%dx%d)", page_idx + 1, img.width, img.height)

            # Step B: 文本检测（带缓存）
            cached = _load_ocr_cache(file_hash, page_idx)
            if cached is not None:
                det_boxes, ocr_records = cached
                logger.debug("第 %d 页: 命中 OCR 缓存", page_idx + 1)
            else:
                det_boxes, ocr_records = _retry_ocr(
                    lambda pn=processed_np, pi=page_idx: detect_text(pn, pi),
                    description=f"第 {page_idx + 1} 页文本检测",
                )
                _save_ocr_cache(file_hash, page_idx, (det_boxes, ocr_records))
                logger.debug("第 %d 页: OCR 结果已缓存", page_idx + 1)
            logger.debug("第 %d 页: 检测到 %d 个文本区域, %d 条识别记录",
                          page_idx + 1, len(det_boxes), len(ocr_records))

            # Step C: 版面分析与分离
            img_h = processed_np.shape[0]
            layout_result = analyze_layout(det_boxes, img_h, page_idx,
                                           ocr_records=ocr_records)
            hw_boxes = layout_result["handwriting_boxes"]
            logger.info("第 %d 页: 版面分析 → %d 个手写区域",
                         page_idx + 1, len(hw_boxes))

            for pg, regions in layout_result["question_regions"].items():
                question_regions.setdefault(pg, []).extend(regions)

            # Step D: 手写识别（支持 PaddleOCR / Qwen3-VL / 双路并行 / 整页识别）
            from core.recognition_config import RECOGNITION_ENGINE

            if RECOGNITION_ENGINE == "page_level":
                from core.qwen_vl_recognizer import recognize_page_level
                # 整页识别：使用原始图像（不做预处理）
                original_np = np.array(img)
                # 使用 question_regions + hw_boxes 确定书写区域
                page_question_regions = question_regions.get(page_idx, [])
                if not page_question_regions:
                    logger.warning("第 %d 页: 未检测到题目区域", page_idx + 1)
                    continue
                page_results = recognize_page_level(
                    original_np, page_question_regions, page_idx,
                    handwriting_boxes=hw_boxes,
                )
            else:
                if not hw_boxes:
                    logger.warning("第 %d 页: 未检测到手写区域", page_idx + 1)
                    continue

                if RECOGNITION_ENGINE == "qwen_vl":
                    from core.qwen_vl_recognizer import recognize_with_qwen_vl
                    page_results = recognize_with_qwen_vl(
                        processed_np, hw_boxes, page_idx,
                    )
                elif RECOGNITION_ENGINE == "dual":
                    from concurrent.futures import ThreadPoolExecutor
                    from core.qwen_vl_recognizer import recognize_with_qwen_vl
                    from core.recognition_merger import merge_recognition_results

                    with ThreadPoolExecutor(max_workers=2) as pool:
                        paddle_future = pool.submit(
                            recognize_handwriting,
                            processed_np, hw_boxes, ocr_records, page_idx,
                        )
                        qwen_future = pool.submit(
                            recognize_with_qwen_vl,
                            processed_np, hw_boxes, page_idx,
                        )
                        paddle_results = paddle_future.result()
                        qwen_results = qwen_future.result()

                    page_results = merge_recognition_results(paddle_results, qwen_results)
                else:
                    # "paddle" — 原有逻辑
                    page_results = recognize_handwriting(
                        processed_np, hw_boxes, ocr_records, page_idx,
                    )
            all_results.extend(page_results)

        doc.close()
        return all_results, question_regions

    def _grade_results(
        self,
        all_results: list[dict],
        answers: list[str],
        question_regions: dict[int, list[dict]],
        t_start: float,
    ) -> tuple[list[dict], dict, float, float, float, bool]:
        """比对评分 + 按题汇总 + 可观测性指标计算。

        Returns:
            (graded, question_summary, avg_conf, low_conf_ratio, process_time, needs_review)
        """
        from core.grader import grade, summarize_by_question

        # 按页分组（每页 = 一个学生的独立答卷）
        page_answers: dict[int, list[dict]] = defaultdict(list)
        for r in all_results:
            page_answers[r["page"]].append(r)

        graded = []
        for page_idx in sorted(page_answers.keys()):
            page_graded = grade(page_answers[page_idx], answers)
            graded.extend(page_graded)
            logger.info("第 %d 页: 手写 %d 字 → 对齐 %d 字",
                         page_idx + 1, len(page_answers[page_idx]), len(page_graded))

        logger.info("总对齐结果: %d 字（%d 页）", len(graded), len(page_answers))

        question_summary = summarize_by_question(graded, question_regions)

        # 可观测性指标
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

        return graded, question_summary, avg_conf, low_conf_ratio, process_time, needs_review

    def _save_and_output(
        self,
        path: Path,
        graded: list[dict],
        question_summary: dict,
        class_name: str | None,
        date_str: str | None,
        seq: str | None,
        avg_conf: float,
        low_conf_ratio: float,
        process_time: float,
        needs_review: bool,
    ) -> None:
        """保存数据库 + 生成批注 PDF + 打印 + 归档。"""
        from db.crud import save_task

        # 4. 保存到数据库
        try:
            save_task(
                str(path), graded,
                class_name=class_name,
                date_str=date_str,
                seq=seq,
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
        annotated_path = annotate(str(path), graded, str(output_dir),
                                  question_summary=question_summary)
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
