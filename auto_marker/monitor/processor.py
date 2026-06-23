"""流水线调度 — 向后兼容入口。

实际业务逻辑已迁移至 services/pipeline_service.py。
本模块保留模块级函数供 watcher.py / run_test.py 等现有调用方使用。
"""

from services.pipeline_service import PATTERN, PipelineService  # noqa: F401


def process_pdf(pdf_path: str) -> None:
    """完整批改流水线（向后兼容入口）。"""
    PipelineService().process_pdf(pdf_path)


def load_answers(class_name: str, date_str: str) -> list[str]:
    """加载标准答案（向后兼容入口）。"""
    return PipelineService().load_answers(class_name, date_str)
