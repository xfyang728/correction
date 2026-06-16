"""
数据操作 — CRUD 封装。
"""

import json
import logging
from pathlib import Path

from db.models import init_db, Task, Answer

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


def save_task(pdf_path: str, graded_results: list[dict],
              annotated_pdf: str | None = None) -> int:
    """保存批改任务到数据库。"""
    from monitor.processor import PATTERN
    import re

    path = Path(pdf_path)
    m = PATTERN.search(path.name)
    class_name = m.group(1) if m else None
    date_str = m.group(2) if m else None
    seq = m.group(3) if m else None

    total = len(graded_results)
    correct = sum(1 for r in graded_results if r["status"] == "correct")
    uncertain = sum(1 for r in graded_results if r["status"] == "uncertain")
    wrong = sum(1 for r in graded_results if r["status"] == "wrong")

    session = _get_session()
    try:
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
        )
        session.add(task)
        session.commit()
        task_id = task.id
        logger.info("任务 #%d 已保存: %s", task_id, path.name)
        return task_id
    except Exception as e:
        session.rollback()
        logger.error("保存任务失败: %s", e)
        raise
    finally:
        session.close()


def get_tasks(limit: int = 50) -> list[Task]:
    """获取最近任务列表。"""
    session = _get_session()
    try:
        return session.query(Task).order_by(Task.id.desc()).limit(limit).all()
    finally:
        session.close()


def get_task(task_id: int) -> Task | None:
    """获取单个任务。"""
    session = _get_session()
    try:
        return session.query(Task).filter(Task.id == task_id).first()
    finally:
        session.close()


def save_answer(class_name: str, date_str: str, content: str) -> int:
    """保存标准答案。"""
    session = _get_session()
    try:
        ans = Answer(class_name=class_name, date_str=date_str, content=content)
        session.add(ans)
        session.commit()
        return ans.id
    except Exception as e:
        session.rollback()
        logger.error("保存答案失败: %s", e)
        raise
    finally:
        session.close()


def get_answer(class_name: str | None = None, date_str: str | None = None) -> str | None:
    """获取答案内容。"""
    session = _get_session()
    try:
        q = session.query(Answer)
        if class_name:
            q = q.filter(Answer.class_name == class_name)
        if date_str:
            q = q.filter(Answer.date_str == date_str)
        ans = q.order_by(Answer.id.desc()).first()
        return ans.content if ans else None
    finally:
        session.close()


def get_all_answers() -> list[Answer]:
    """获取所有答案记录。"""
    session = _get_session()
    try:
        return session.query(Answer).order_by(Answer.id.desc()).all()
    finally:
        session.close()


def delete_answer(answer_id: int) -> bool:
    """删除指定答案。"""
    session = _get_session()
    try:
        ans = session.query(Answer).filter(Answer.id == answer_id).first()
        if ans:
            session.delete(ans)
            session.commit()
            return True
        return False
    except Exception as e:
        session.rollback()
        logger.error("删除答案失败: %s", e)
        return False
    finally:
        session.close()
