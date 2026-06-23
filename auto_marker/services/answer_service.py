"""答案管理服务 — 封装答案 CRUD 业务逻辑。"""

import logging

logger = logging.getLogger("answer_service")


class AnswerService:
    """答案管理服务。"""

    @staticmethod
    def get_answer(class_name: str | None, date_str: str | None) -> str | None:
        """获取答案内容。"""
        from db.crud import get_answer as _get_answer
        return _get_answer(class_name, date_str)

    @staticmethod
    def save_answer(class_name: str, date_str: str, content: str) -> int:
        """保存标准答案。"""
        from db.crud import save_answer as _save_answer
        return _save_answer(class_name, date_str, content)

    @staticmethod
    def get_all_answers() -> list:
        """获取所有答案记录。"""
        from db.crud import get_all_answers as _get_all
        return _get_all()

    @staticmethod
    def delete_answer(answer_id: int) -> bool:
        """删除指定答案。"""
        from db.crud import delete_answer as _delete
        return _delete(answer_id)
