"""任务管理服务 — 封装任务 CRUD 业务逻辑。"""

import logging

logger = logging.getLogger("task_service")


class TaskService:
    """任务管理服务。"""

    @staticmethod
    def get_tasks(limit: int = 50) -> list:
        """获取最近任务列表。"""
        from db.crud import get_tasks as _get_tasks
        return _get_tasks(limit)

    @staticmethod
    def get_task(task_id: int):
        """获取单个任务。"""
        from db.crud import get_task as _get_task
        return _get_task(task_id)

    @staticmethod
    def save_correction(task_id: int, char_index: int,
                        original_char: str, original_status: str,
                        corrected_char: str, corrected_status: str,
                        bbox_pixel: dict | None = None,
                        confidence: float | None = None) -> int:
        """保存单条人工复核修正记录。"""
        from db.crud import save_correction as _save
        return _save(task_id, char_index, original_char, original_status,
                     corrected_char, corrected_status, bbox_pixel, confidence)

    @staticmethod
    def update_task_result(task_id: int, updated_results: list[dict],
                           review_done: bool = True) -> bool:
        """更新任务的批改结果（人工复核后）。"""
        from db.crud import update_task_result as _update
        return _update(task_id, updated_results, review_done)
