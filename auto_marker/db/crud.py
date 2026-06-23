"""
数据操作 — CRUD 封装。
"""

import logging
from contextlib import contextmanager
from pathlib import Path

from db.models import Answer, ReviewCorrection, Task, init_db

logger = logging.getLogger("db")

# 全局 Session，lazy 初始化
_session_factory = None


def _get_session():
    global _session_factory
    if _session_factory is None:
        db_path = Path(__file__).resolve().parent.parent / "data" / "marker.db"
        db_path.parent.mkdir(parents=True, exist_ok=True)
        _session_factory = init_db(str(db_path))
        logger.info("数据库初始化: %s", db_path)
    return _session_factory()


@contextmanager
def _session_scope():
    """Session 生命周期 context manager — 自动 commit/rollback/close。"""
    session = _get_session()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def save_task(pdf_path: str, graded_results: list[dict],
              class_name: str | None = None,
              date_str: str | None = None,
              seq: str | None = None,
              annotated_pdf: str | None = None,
              avg_confidence: float | None = None,
              low_conf_ratio: float | None = None,
              process_time: float | None = None,
              needs_review: bool = False) -> int:
    """保存批改任务到数据库。

    元数据（class_name/date_str/seq）由调用方解析后传入，
    避免数据层反向依赖业务模块。
    """
    path = Path(pdf_path)

    total = len(graded_results)
    correct = sum(1 for r in graded_results if r["status"] == "correct")
    uncertain = sum(1 for r in graded_results if r["status"] == "uncertain")
    wrong = sum(1 for r in graded_results if r["status"] == "wrong")

    try:
        with _session_scope() as session:
            task = Task(
                filename=path.name,
                class_name=class_name,
                date_str=date_str,
                seq=seq,
                total_chars=total,
                correct_count=correct,
                uncertain_count=uncertain,
                wrong_count=wrong,
                result_json=graded_results,
                annotated_pdf=annotated_pdf,
                avg_confidence=avg_confidence,
                low_conf_ratio=low_conf_ratio,
                process_time=process_time,
                needs_review=needs_review,
            )
            session.add(task)
            session.flush()  # 获取 task.id
            task_id = task.id
            logger.info("任务 #%d 已保存: %s (需复核=%s)", task_id, path.name, needs_review)
            return task_id
    except Exception as e:
        logger.error("保存任务失败: %s", e)
        raise


def get_tasks(limit: int = 50) -> list[Task]:
    """获取最近任务列表。"""
    with _session_scope() as session:
        return session.query(Task).order_by(Task.id.desc()).limit(limit).all()


def get_task(task_id: int) -> Task | None:
    """获取单个任务。"""
    with _session_scope() as session:
        return session.query(Task).filter(Task.id == task_id).first()


def save_answer(class_name: str, date_str: str, content: str) -> int:
    """保存标准答案。"""
    try:
        with _session_scope() as session:
            ans = Answer(class_name=class_name, date_str=date_str, content=content)
            session.add(ans)
            session.flush()
            return ans.id
    except Exception as e:
        logger.error("保存答案失败: %s", e)
        raise


def get_answer(class_name: str | None = None, date_str: str | None = None) -> str | None:
    """获取答案内容。"""
    with _session_scope() as session:
        q = session.query(Answer)
        if class_name:
            q = q.filter(Answer.class_name == class_name)
        if date_str:
            q = q.filter(Answer.date_str == date_str)
        ans = q.order_by(Answer.id.desc()).first()
        return ans.content if ans else None


def get_all_answers() -> list[Answer]:
    """获取所有答案记录。"""
    with _session_scope() as session:
        return session.query(Answer).order_by(Answer.id.desc()).all()


def delete_answer(answer_id: int) -> bool:
    """删除指定答案。"""
    try:
        with _session_scope() as session:
            ans = session.query(Answer).filter(Answer.id == answer_id).first()
            if ans:
                session.delete(ans)
                return True
            return False
    except Exception as e:
        logger.error("删除答案失败: %s", e)
        return False


# ============================================================
#  人工复核相关 CRUD
# ============================================================

def update_task_result(task_id: int, updated_results: list[dict],
                       review_done: bool = True) -> bool:
    """更新任务的批改结果（人工复核后），并重新计算统计。"""
    try:
        with _session_scope() as session:
            task = session.query(Task).filter(Task.id == task_id).first()
            if not task:
                return False

            task.result_json = updated_results
            task.correct_count = sum(1 for r in updated_results if r["status"] == "correct")
            task.uncertain_count = sum(1 for r in updated_results if r["status"] == "uncertain")
            task.wrong_count = sum(1 for r in updated_results if r["status"] == "wrong")
            task.review_done = review_done
            logger.info("任务 #%d 复核结果已更新", task_id)
            return True
    except Exception as e:
        logger.error("更新任务结果失败: %s", e)
        return False


def save_correction(task_id: int, char_index: int,
                    original_char: str, original_status: str,
                    corrected_char: str, corrected_status: str,
                    bbox_pixel: dict | None = None,
                    confidence: float | None = None) -> int:
    """保存单条人工复核修正记录。"""
    try:
        with _session_scope() as session:
            corr = ReviewCorrection(
                task_id=task_id,
                char_index=char_index,
                original_char=original_char,
                original_status=original_status,
                corrected_char=corrected_char,
                corrected_status=corrected_status,
                bbox_pixel=bbox_pixel,
                confidence=confidence,
            )
            session.add(corr)
            session.flush()
            return corr.id
    except Exception as e:
        logger.error("保存修正记录失败: %s", e)
        raise


def get_corrections(task_id: int) -> list[ReviewCorrection]:
    """获取某任务的所有复核修正记录。"""
    with _session_scope() as session:
        return (
            session.query(ReviewCorrection)
            .filter(ReviewCorrection.task_id == task_id)
            .order_by(ReviewCorrection.char_index)
            .all()
        )


def get_all_corrections(limit: int = 1000) -> list[ReviewCorrection]:
    """获取所有修正记录（用于导出训练集）。"""
    with _session_scope() as session:
        return (
            session.query(ReviewCorrection)
            .order_by(ReviewCorrection.id.desc())
            .limit(limit)
            .all()
        )
