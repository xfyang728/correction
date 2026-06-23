# P3-18 OCR 缓存接入 + P3-20 Service 层完整重构

## 概述

完成项目差距分析计划中剩余的两项改进：
- **P3-18**：将已有的 OCR 缓存函数（`_load_ocr_cache`/`_save_ocr_cache`）接入 `process_pdf` 的页面循环，缓存 `detect_text` 结果
- **P3-20**：创建 `services/` 层，将 `process_pdf` 单体函数拆分为 `PipelineService`，同时创建 `AnswerService` 和 `TaskService` 封装业务逻辑，`web/app.py` 改为调用 service 层

---

## 当前状态分析

### P3-18 OCR 缓存
- **已完成**：`_file_hash()`、`_load_ocr_cache()`、`_save_ocr_cache()` 函数已定义（processor.py 第 52-81 行）
- **已完成**：`file_hash = _file_hash(path)` 已在 process_pdf 第 149 行调用
- **未完成**：缓存检查/保存逻辑未接入第 170-177 行的 `_retry_ocr(detect_text)` 调用

### P3-20 Service 层
- **现状**：`process_pdf` 是 174 行单体函数（processor.py 第 113-286 行）
- **现状**：`web/app.py` 直接调用 `db.crud` 的 9 个函数（get_tasks, get_task, save_correction, update_task_result, save_answer, get_answer, get_all_answers, delete_answer）
- **现状**：`db/crud.py` 中 `save_task` 反向依赖 `monitor.processor.PATTERN`（循环依赖隐患）
- **现状**：无 `services/` 目录

---

## 实施方案

### Step 1: P3-18 — 接入 OCR 缓存

**文件**：`d:\MyCode\correction\auto_marker\monitor\processor.py`

**改动**：在 `for page_idx in range(num_pages):` 循环中，Step B（文本检测）前后添加缓存逻辑：

```python
# Step B: 阶段一 — 文本检测（PP-OCRv6 det model）
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
```

**缓存粒度**：仅缓存 `detect_text` 的 `(det_boxes, ocr_records)` 结果。版面分析和手写识别不缓存（依赖图像预处理结果，且计算量较小）。

---

### Step 2: P3-20 — 创建 services 层

#### 2.1 创建 `services/__init__.py`

空包初始化文件。

#### 2.2 创建 `services/pipeline_service.py`

从 `monitor/processor.py` 的 `process_pdf` 提取业务逻辑为 `PipelineService` 类：

```python
class PipelineService:
    """批改流水线服务 — 封装从 PDF 到归档的完整流程。"""

    def __init__(self):
        self._ocr_dpi = 200
        self._low_confidence_threshold = 0.60
        self._review_trigger_ratio = 0.30

    def process_pdf(self, pdf_path: str) -> None:
        """完整批改流水线入口。"""
        # 原有 process_pdf 的全部逻辑移入此处

    def _parse_metadata(self, path: Path) -> tuple[str, str, str] | None:
        """从文件名解析元数据 (class_name, date_str, seq)。"""

    def _run_ocr_pipeline(self, doc, file_hash: str) -> tuple[list[dict], dict[int, list[dict]], list[tuple[int, int]]]:
        """三阶段流水线：渲染 → 预处理 → 文本检测 → 版面分析 → 手写识别。"""

    def _grade_results(self, all_results: list[dict], answers: list[str]) -> tuple[list[dict], dict]:
        """比对评分 + 按题汇总 + 可观测性指标。"""

    def _save_and_output(self, path: Path, graded: list[dict], question_summary: dict,
                         avg_conf: float, low_conf_ratio: float,
                         process_time: float, needs_review: bool) -> None:
        """保存数据库 + 生成批注 PDF + 打印 + 归档。"""
```

**关键设计决策**：
- `process_pdf` 保留为模块级函数（向后兼容 `watcher.py` 和 `run_test.py` 的调用），内部委托给 `PipelineService().process_pdf()`
- `PATTERN` 正则移到 `pipeline_service.py`，消除 `db/crud.py` 对 `monitor.processor` 的反向依赖
- 缓存函数（`_file_hash`/`_load_ocr_cache`/`_save_ocr_cache`）和重试函数（`_retry_ocr`）保留在 `pipeline_service.py` 中作为私有函数
- `load_answers` 移到 `pipeline_service.py`

#### 2.3 创建 `services/answer_service.py`

封装答案相关的业务逻辑：

```python
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
```

#### 2.4 创建 `services/task_service.py`

封装任务相关的业务逻辑：

```python
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
```

#### 2.5 修改 `web/app.py`

将直接调用 `db.crud` 改为调用 `services` 层：

```python
# 原来：
from db.crud import (delete_answer, get_all_answers, get_answer, get_task, get_tasks,
                     save_answer, save_correction, update_task_result)

# 改为：
from services.answer_service import AnswerService
from services.task_service import TaskService
```

所有调用点相应替换，例如：
- `get_tasks(limit=500)` → `TaskService.get_tasks(limit=500)`
- `save_answer(cls_name, date_str, ...)` → `AnswerService.save_answer(cls_name, date_str, ...)`

#### 2.6 修改 `db/crud.py`

消除 `save_task` 对 `monitor.processor.PATTERN` 的反向依赖：
- 将 `PATTERN` 正则移到 `services/pipeline_service.py`
- `save_task` 不再从 `monitor.processor` 导入 `PATTERN`，改为接受 `class_name`/`date_str`/`seq` 作为参数（已经是这样，但内部重新解析了文件名）
- 实际上 `save_task` 已经接受 `pdf_path` 并自行解析元数据。改为直接接受已解析的元数据参数

**改动**：`save_task` 签名改为：
```python
def save_task(pdf_path: str, graded_results: list[dict],
              class_name: str | None = None,
              date_str: str | None = None,
              seq: str | None = None,
              annotated_pdf: str | None = None,
              avg_confidence: float | None = None,
              low_conf_ratio: float | None = None,
              process_time: float | None = None,
              needs_review: bool = False) -> int:
```

移除 `from monitor.processor import PATTERN` 和内部文件名解析逻辑。调用方（`PipelineService._save_and_output`）已解析好元数据，直接传入。

#### 2.7 修改 `monitor/processor.py`

简化为薄包装层：

```python
"""流水线调度 — 向后兼容入口。"""

from services.pipeline_service import PipelineService

# 保留模块级函数供 watcher.py / run_test.py 调用
PATTERN = PipelineService.PATTERN

def process_pdf(pdf_path: str) -> None:
    """完整批改流水线（向后兼容入口）。"""
    PipelineService().process_pdf(pdf_path)

def load_answers(class_name: str, date_str: str) -> list[str]:
    """加载标准答案（向后兼容入口）。"""
    return PipelineService().load_answers(class_name, date_str)
```

---

## 文件变更清单

| 操作 | 文件 | 说明 |
|------|------|------|
| 新建 | `services/__init__.py` | 空包初始化 |
| 新建 | `services/pipeline_service.py` | PipelineService 类，从 processor.py 提取 |
| 新建 | `services/answer_service.py` | AnswerService 类，封装答案 CRUD |
| 新建 | `services/task_service.py` | TaskService 类，封装任务 CRUD |
| 修改 | `monitor/processor.py` | 简化为薄包装层，保留 process_pdf 入口 |
| 修改 | `db/crud.py` | save_task 移除对 monitor.processor 的反向依赖 |
| 修改 | `web/app.py` | 改为调用 services 层而非直接调 db.crud |

---

## 实施顺序

1. **Step 1**：P3-18 OCR 缓存接入（processor.py 第 170-177 行）
2. **Step 2.1**：创建 `services/__init__.py`
3. **Step 2.2**：创建 `services/pipeline_service.py`（从 processor.py 提取逻辑）
4. **Step 2.3**：创建 `services/answer_service.py`
5. **Step 2.4**：创建 `services/task_service.py`
6. **Step 2.5**：修改 `db/crud.py`（移除反向依赖）
7. **Step 2.6**：修改 `monitor/processor.py`（简化为薄包装层）
8. **Step 2.7**：修改 `web/app.py`（改用 services 层）
9. **验证**：运行 `ruff check .` + `pytest` 确认无回归

---

## 假设与决策

1. **缓存粒度**：仅缓存 `detect_text` 结果（用户选择），版面分析和手写识别不缓存
2. **Service 层模式**：使用静态方法类（而非实例方法），因为当前无状态需要跨请求保持
3. **向后兼容**：`monitor/processor.py` 保留 `process_pdf` 和 `PATTERN` 的模块级入口，确保 `watcher.py` 和 `run_test.py` 无需修改
4. **PATTERN 归属**：正则移到 `PipelineService.PATTERN`，processor.py 通过 `PipelineService.PATTERN` 重导出
5. **save_task 签名变更**：不再内部解析文件名，改为接受已解析的元数据参数，消除循环依赖

---

## 验证步骤

1. `ruff check .` — 无 lint 错误
2. `pytest tests/ -v` — 24 个现有测试全部通过
3. 手动检查 `web/app.py` 所有调用点已替换为 service 层
4. 确认 `db/crud.py` 不再 `from monitor.processor import PATTERN`
5. 确认 `monitor/processor.py` 的 `process_pdf` 仍可被 `watcher.py` 和 `run_test.py` 正常调用
