# 项目差距分析与改进计划

## 概述

对 `auto_marker` 项目进行全面代码审查，对照工程最佳实践，识别出 **20 项差距**，按优先级分为 P0（正确性Bug）、P1（可靠性）、P2（工程质量）、P3（架构与安全）四个层级。每项给出具体问题位置、影响分析和修复方案。

---

## P0 — 正确性 Bug（影响运行，应立即修复）

### 1. Web App `page_all_results` 变量在赋值前被引用

- **文件**: [web/app.py](file:///d:\MyCode\correction\auto_marker\web\app.py#L334)
- **问题**: 第 334 行引用 `page_all_results` 进行按题分组统计，但该变量在第 375 行才赋值（`page_all_results = page_results`）。当任务有结果时，进入按题统计代码块会抛出 `NameError`。
- **影响**: 任务详情页在有批改结果时直接崩溃，无法查看按题统计。
- **修复方案**: 将 `page_all_results = page_results` 的赋值移到按题统计代码块之前（第 333 行之前），确保先保存全量再过滤。

```python
# 修复后的顺序
page_results = [r for r in results if r["page"] == page_sel]
page_all_results = page_results  # 先保存全量
if status_filter != "全部":
    page_results = [r for r in page_results if r["status"] == status_filter]
# ... 然后再用 page_all_results 做按题统计
```

### 2. OCR 单例初始化无线程锁（竞态条件）

- **文件**: [core/text_detector.py](file:///d:\MyCode\correction\auto_marker\core\text_detector.py#L20-L28)、[core/handwriting_recognizer.py](file:///d:\MyCode\correction\auto_marker\core\handwriting_recognizer.py#L23-L29)
- **问题**: `_get_detector()` 和 `_get_sub_region_ocr()` 使用全局变量 + None 检查实现单例，但无线程锁。`watcher.py` 使用 `ThreadPoolExecutor(max_workers=4)` 并发处理 PDF，多个线程可能同时通过 None 检查，导致 PaddleOCR 被初始化多次（每次加载耗时数秒 + 占用数 GB 内存）。
- **影响**: 高并发时内存暴涨、模型重复加载、潜在段错误。
- **修复方案**: 加 `threading.Lock()` 保护单例初始化。

```python
import threading
_det_instance = None
_det_lock = threading.Lock()

def _get_detector():
    global _det_instance
    if _det_instance is None:
        with _det_lock:
            if _det_instance is None:  # double-check
                from paddleocr import PaddleOCR
                _det_instance = PaddleOCR(lang='ch')
    return _det_instance
```

### 3. Printer 模块 `printer_name` 参数被完全忽略

- **文件**: [core/printer.py](file:///d:\MyCode\correction\auto_marker\core\printer.py#L34-L42)
- **问题**: `if printer_name:` 和 `else:` 两个分支的命令完全相同，`printer_name` 参数从未被使用。config.ini 中 `[printer] name` 字段也从未被读取传入。
- **影响**: 用户在配置中指定打印机名称无效，始终使用系统默认打印机。
- **修复方案**: 使用 `SumatraPDF` 或 `/Printer` 参数传递打印机名称，或回退到 Windows `win32print` API。

```python
if printer_name:
    cmd = ["SumatraPDF", "-print-to", printer_name, "-silent", str(path)]
else:
    cmd = ["SumatraPDF", "-print-to-default", "-silent", str(path)]
```

### 4. Image Processor 日志显示错误尺寸

- **文件**: [core/image_processor.py](file:///d:\MyCode\correction\auto_marker\core\image_processor.py#L51)
- **问题**: 第 51 行 `logger.info("低分辨率图像 %dx%d → 上采样到 %dx%d", w_orig := w, h_orig := h, new_w, new_h)` — 此处 `w` 和 `h` 已在第 49 行被 resize 更新为新尺寸，walrus 操作符 `w_orig := w` 只是把已更新的 `w` 赋给 `w_orig`，日志显示的"原始尺寸"实际是新尺寸。
- **影响**: 日志误导调试，不影响功能。
- **修复方案**: 在 resize 之前保存原始尺寸。

```python
h_orig, w_orig = gray.shape[:2]
if min(h_orig, w_orig) < LOW_RES_THRESHOLD:
    scale = LOW_RES_THRESHOLD / min(h_orig, w_orig)
    new_w, new_h = int(w_orig * scale), int(h_orig * scale)
    gray = cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    logger.info("低分辨率图像 %dx%d → 上采样到 %dx%d", w_orig, h_orig, new_w, new_h)
```

### 5. 子区域 OCR 重复创建 PaddleOCR 实例

- **文件**: [core/handwriting_recognizer.py](file:///d:\MyCode\correction\auto_marker\core\handwriting_recognizer.py#L23-L29)
- **问题**: `_get_sub_region_ocr()` 创建了一个全新的 `PaddleOCR(lang='ch')` 实例，与 `text_detector.py` 中的 `_det_instance` 完全独立。两个实例各占数 GB 内存，且初始化耗时数秒。
- **影响**: 内存浪费翻倍，首次子区域识别时阻塞数秒等待模型加载。
- **修复方案**: 复用 `text_detector._get_detector()` 的单例，删除独立的子区域 OCR 实例。

```python
def _get_sub_region_ocr():
    from core.text_detector import _get_detector
    return _get_detector()  # 复用同一实例
```

---

## P1 — 可靠性与健壮性

### 6. config.ini 配置与代码完全脱节

- **文件**: [config.ini](file:///d:\MyCode\correction\auto_marker\config.ini#L15-L18)、[core/text_detector.py](file:///d:\MyCode\correction\auto_marker\core\text_detector.py#L26)
- **问题**: `config.ini` 的 `[ocr]` section 定义了 `model`、`use_gpu`、`confidence_threshold`，但 `text_detector.py` 硬编码 `PaddleOCR(lang='ch')`，从未读取配置。`[grader]` 的 `green_threshold`/`orange_threshold` 同样未被 `grader.py` 读取（grader 使用函数参数默认值）。Web UI 允许修改配置但实际不生效。
- **影响**: 用户通过 Web UI 修改 OCR 模型、GPU、阈值等配置后完全不生效，产生误导。
- **修复方案**: 创建 `core/config.py` 集中读取 config.ini，各模块从配置对象获取参数。

```python
# core/config.py
import configparser
from pathlib import Path

_cfg = None
def get_config():
    global _cfg
    if _cfg is None:
        _cfg = configparser.ConfigParser()
        _cfg.read(Path(__file__).resolve().parent.parent / "config.ini", encoding="utf-8")
    return _cfg
```

### 7. 死代码：ocr_engine.py 未被使用

- **文件**: [core/ocr_engine.py](file:///d:\MyCode\correction\auto_marker\core\ocr_engine.py)
- **问题**: 该文件 204 行，提供 `ocr_image()` 和 `ocr_pdf()` 接口，但已被三阶段流水线（text_detector + layout_analyzer + handwriting_recognizer）完全取代。`processor.py` 不再调用它。其中的 `_is_chinese_char` 与 `handwriting_recognizer.py` 重复定义。
- **影响**: 代码维护负担，混淆贡献者，`_is_chinese_char` 等函数两处定义可能不一致。
- **修复方案**: 删除 `ocr_engine.py`，确认无其他模块 import 后移除。

### 8. 死代码：extract_student_answers 兼容函数

- **文件**: [core/layout_analyzer.py](file:///d:\MyCode\correction\auto_marker\core\layout_analyzer.py#L540-L551)
- **问题**: `extract_student_answers()` 返回空列表并打印警告，是旧接口的过渡兼容层。`processor.py` 已使用新三阶段流水线，不再调用此函数。
- **修复方案**: 直接删除该函数。

### 9. process_pdf 无重试机制

- **文件**: [monitor/processor.py](file:///d:\MyCode\correction\auto_marker\monitor\processor.py#L59-L146)
- **问题**: OCR predict() 偶发失败（GPU OOM、模型加载超时等）时，整个任务直接失败并归档到 `failed` 目录，无重试。用户需手动重新上传。
- **影响**: 偶发失败导致用户体验差，需人工干预。
- **修复方案**: 对 OCR 调用添加 1-2 次重试，仅对可重试异常（超时、OOM）重试，对确定性错误不重试。

```python
import time
def _retry(fn, max_retries=2, delay=2.0):
    for attempt in range(max_retries + 1):
        try:
            return fn()
        except Exception as e:
            if attempt == max_retries:
                raise
            logger.warning("第 %d 次重试: %s", attempt + 1, e)
            time.sleep(delay)
```

### 10. 数据库 Session 管理缺乏 Context Manager

- **文件**: [db/crud.py](file:///d:\MyCode\correction\auto_marker\db\crud.py#L17-L24)
- **问题**: 每个 CRUD 函数手动 `session = _get_session()` + `try/finally: session.close()`。如果中间逻辑抛出未捕获异常（如 JSON 序列化失败），rollback 可能未执行。重复样板代码 ~15 处。
- **影响**: 异常路径下可能连接泄漏；代码冗余。
- **修复方案**: 使用 contextmanager 封装 session 生命周期。

```python
from contextlib import contextmanager

@contextmanager
def _session_scope():
    session = _get_session()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()

# 使用
def get_tasks(limit=50):
    with _session_scope() as session:
        return session.query(Task).order_by(Task.id.desc()).limit(limit).all()
```

---

## P2 — 工程质量

### 11. 零测试覆盖

- **现状**: 无 `pytest`，无 `tests/` 目录，仅有手动调试脚本 `_debug_ocr.py` 和端到端 `run_test.py`。核心算法（DP 对齐、题号正则匹配、列检测、手写分配）无任何单元测试。
- **影响**: 每次修改 layout_analyzer 的复杂逻辑（列感知、跨列纠正）后，只能靠手动运行调试脚本验证，回归风险高。
- **修复方案**:
  - 添加 `pytest` 到 requirements.txt
  - 创建 `tests/` 目录
  - 为 `grader._align()` 编写单元测试（纯函数，易测试）
  - 为 `layout_analyzer._question_match()` 编写正则匹配测试
  - 为 `layout_analyzer._split_into_columns()` 编写列检测测试
  - 为 `layout_analyzer._build_column_intervals()` 编写区间构建测试
  - 目标：核心算法函数测试覆盖率 > 80%

### 12. 魔法数字散落各处

- **文件**: [core/layout_analyzer.py](file:///d:\MyCode\correction\auto_marker\core\layout_analyzer.py#L32-L33)、[core/handwriting_recognizer.py](file:///d:\MyCode\correction\auto_marker\core\handwriting_recognizer.py#L200)
- **问题**: 大量阈值硬编码：
  - `HANDWRITTEN_HEIGHT_MIN = 90`、`PINYIN_HEIGHT_MAX = 55`
  - `_SUB_REGION_HEIGHT_MIN = 130`、`_SUB_REGION_RATIO = 0.55`
  - `_COLUMN_GAP_THRESHOLD = 150`、`_Y_PROXIMITY_THRESHOLD = 30`
  - `mark_r = max(min(mark_r, 18), 8)`（pdf_annotator.py）
  - `OCR_DPI = 200`、`BLUR_THRESHOLD = 60`
- **影响**: 调参需改代码、重新部署；不同 DPI 场景下阈值不可配置。
- **修复方案**: 将所有阈值集中到 `config.ini` 的对应 section，通过 `core/config.py` 读取。

### 13. 工具函数重复定义

- **问题**:
  - `_is_chinese_char`: 定义在 `ocr_engine.py`（待删）和 `handwriting_recognizer.py`
  - `_bbox_center` / `_bbox_center_x` / `_bbox_center_y`: 定义在 `handwriting_recognizer.py` 和 `layout_analyzer.py`
  - `_bbox_h`: 定义在 `layout_analyzer.py`
- **修复方案**: 创建 `core/utils.py`，集中 bbox 操作和字符判断工具函数，各模块 import。

### 14. 无 Linting / 类型检查配置

- **现状**: 无 `.flake8`、`pyproject.toml`、`ruff.toml`、`mypy.ini`。代码中部分函数有类型注解（如 `-> list[dict]`），但不完整且无检查。
- **修复方案**:
  - 添加 `pyproject.toml`，配置 `ruff`（替代 flake8 + isort）
  - 添加 `mypy` 配置（渐进式，先 strict 可选）
  - 添加 pre-commit hook

```toml
# pyproject.toml
[tool.ruff]
line-length = 100
target-version = "py312"

[tool.ruff.lint]
select = ["E", "F", "W", "I", "UP", "B"]
```

### 15. 无 pyproject.toml 项目元数据

- **现状**: 仅有 `requirements.txt`，无 `pyproject.toml`。项目无正式包名、版本、入口点定义。
- **修复方案**: 创建 `pyproject.toml`，定义项目元数据、依赖、工具配置。

---

## P3 — 架构与安全

### 16. 安全：FTP 匿名登录 + Web 无认证

- **文件**: [config.ini](file:///d:\MyCode\correction\auto_marker\config.ini#L5)、[ftp_server.py](file:///d:\MyCode\correction\auto_marker\ftp_server.py)
- **问题**: FTP 默认 `anon = yes`（匿名登录），Web 界面无任何认证。任何同网络用户可访问管理界面、修改答案、查看学生成绩。
- **修复方案**:
  - FTP: 默认关闭匿名，添加用户名密码认证
  - Web: 添加 Streamlit `st.secrets` 密码保护或 OAuth

### 17. 无结构化日志

- **文件**: [monitor/watcher.py](file:///d:\MyCode\correction\auto_marker\monitor\watcher.py#L98-L123)
- **问题**: 日志为纯文本格式，不利于机器解析和监控告警。处理耗时、置信度等指标散落在日志文本中。
- **修复方案**: 添加 JSON 格式日志 handler（可选），或将关键指标写入数据库的 metrics 表。

### 18. 无 OCR 结果缓存

- **文件**: [monitor/processor.py](file:///d:\MyCode\correction\auto_marker\monitor\processor.py#L115)
- **问题**: 同一 PDF 重复处理（如调试时多次运行）每次都重新 OCR，耗时数十秒。
- **修复方案**: 以 PDF 文件 hash 为 key 缓存 OCR 结果到本地磁盘或数据库。

### 19. Watcher 无优雅关闭

- **文件**: [monitor/watcher.py](file:///d:\MyCode\correction\auto_marker\monitor\watcher.py#L148-L156)
- **问题**: `KeyboardInterrupt` 后直接 `executor.shutdown(wait=False)`，在途任务被丢弃。已开始处理但未完成的 PDF 可能产生不完整的批注文件。
- **修复方案**: `shutdown(wait=True, cancel_futures=False)` 等待在途任务完成，或记录未完成任务列表。

### 20. Web App 直接调用 CRUD，无 Service 层

- **文件**: [web/app.py](file:///d:\MyCode\correction\auto_marker\web\app.py#L17-L27)
- **问题**: Web 层直接 import 并调用 `db.crud` 的函数，业务逻辑（如复核后重新统计、答案去空格处理）散落在 UI 代码中。
- **修复方案**: 抽取 `services/` 层封装业务逻辑，Web 层只调 service。此项为架构优化，优先级最低。

---

## 实施优先级总结

| 优先级 | 编号 | 问题 | 工作量 |
|--------|------|------|--------|
| **P0** | 1 | web app NameError | 极小 |
| **P0** | 2 | OCR 单例线程不安全 | 小 |
| **P0** | 3 | printer_name 被忽略 | 小 |
| **P0** | 4 | image_processor 日志bug | 极小 |
| **P0** | 5 | 子区域 OCR 重复实例 | 小 |
| **P1** | 6 | config 脱节 | 中 |
| **P1** | 7 | 删除 ocr_engine.py | 极小 |
| **P1** | 8 | 删除兼容函数 | 极小 |
| **P1** | 9 | OCR 重试机制 | 小 |
| **P1** | 10 | Session context manager | 小 |
| **P2** | 11 | 单元测试框架 | 中 |
| **P2** | 12 | 魔法数字集中配置 | 中 |
| **P2** | 13 | 工具函数去重 | 小 |
| **P2** | 14 | Linting/类型检查 | 小 |
| **P2** | 15 | pyproject.toml | 小 |
| **P3** | 16 | 安全认证 | 中 |
| **P3** | 17 | 结构化日志 | 中 |
| **P3** | 18 | OCR 缓存 | 中 |
| **P3** | 19 | 优雅关闭 | 小 |
| **P3** | 20 | Service 层抽象 | 大 |

## 验证方式

- P0 修复后：运行 `python run_test.py --pdf 301_2026-06-18_003.pdf` 确认流水线正常，Web UI 任务详情页可正常打开
- P1 修复后：修改 config.ini 中 OCR 参数，确认生效；手动 kill OCR 进程确认重试机制工作
- P2 修复后：`pytest tests/ -v` 全部通过；`ruff check .` 无 error
- P3 修复后：匿名 FTP 登录被拒；Web 需密码才能访问
