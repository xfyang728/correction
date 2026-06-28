# 默写批改台 — 规格说明 (spec.md)

> 统一规格说明，整合架构设计、工程约定和经验教训。AI 协作时以此为唯一事实来源。

## 1. 项目概述

半自动默写批改系统：复印机扫描 → FTP 上传 → 自动批改 → 打印输出。老师仅需放纸、按按钮、取卷子。

**技术栈**：Python 3.12+ / PaddleOCR (PP-OCRv6) / Qwen3-VL (llama.cpp) / SQLAlchemy + SQLite / Streamlit / pdfplumber + reportlab + PyPDF2

## 2. 目录结构

```
auto_marker/
├── core/                    # 批改核心引擎
│   ├── image_processor.py   # 图像预处理（去噪/CLAHE/倾斜校正）
│   ├── text_detector.py     # 文本检测（PP-OCRv6 det+rec）
│   ├── layout_analyzer.py   # 版面分析（bbox 高度规则分类 + 题号检测）
│   ├── handwriting_recognizer.py  # PaddleOCR 手写识别
│   ├── qwen_vl_recognizer.py      # Qwen3-VL 整页识别 + 坐标映射
│   ├── recognition_merger.py      # 双路结果合并
│   ├── recognition_config.py      # 识别引擎配置
│   ├── grader.py            # Needleman-Wunsch DP 比对 + 置信度门控
│   ├── pdf_annotator.py     # PDF 批注（标记模式 / 文字渲染模式）
│   ├── printer.py           # Windows 打印
│   └── utils.py             # 工具函数
├── services/                # 服务层（核心业务逻辑）
│   ├── pipeline_service.py  # 批改流水线编排
│   ├── answer_service.py    # 答案管理
│   └── task_service.py      # 任务管理
├── monitor/                 # 文件监控
│   ├── watcher.py           # watchdog 监听 incoming 目录
│   └── processor.py         # 薄包装，委托 PipelineService
├── db/                      # 数据层
│   ├── models.py            # SQLAlchemy ORM（Task/Answer/ReviewCorrection）
│   └── crud.py              # CRUD 操作
├── web/                     # Streamlit 管理界面
│   └── app.py               # 4 Tab：统计/任务/答案/配置
├── tools/                   # 调试与可视化工具
├── tests/                   # 单元测试
├── data/                    # 运行时数据
│   ├── incoming/            # FTP 接收目录
│   ├── output/              # 批注 PDF 输出
│   ├── backup/              # 已处理 PDF 归档
│   ├── failed/              # 处理失败 PDF
│   ├── cache/               # OCR 结果缓存
│   ├── test_samples/        # 测试 PDF 样本
│   └── marker.db            # SQLite 数据库
├── model/                   # PaddleOCR 模型文件
├── run_test.py              # CLI 测试入口
├── ftp_server.py            # FTP 服务器
└── config.ini               # 配置文件
```

## 3. 数据模型

### Task（批改任务）

| 字段 | 类型 | 说明 |
|------|------|------|
| id | Integer PK | 自增主键 |
| filename | String(255) | 原始 PDF 文件名 |
| class_name / date_str / seq | String | 文件名解析的元数据 |
| total_chars / correct_count / uncertain_count / wrong_count | Integer | 统计汇总 |
| result_json | JSON | 完整逐字批改结果（含 question_idx, bbox_pixel, status, confidence） |
| annotated_pdf | String(500) | 批注后 PDF 路径 |
| avg_confidence | Float | OCR 平均置信度 |
| low_conf_ratio | Float | 低置信度字占比 |
| process_time | Float | 处理耗时（秒） |
| needs_review | Boolean | 低置信度占比 > 30% 自动标记 |
| review_done | Boolean | 教师已完成复核 |
| created_at | DateTime | 创建时间 |

### Answer（标准答案）
按 `(class_name, date_str)` 存档，支持增删改查。

### ReviewCorrection（人工复核修正）
关联 Task，记录原始/修正字符和状态，积累为微调训练集。

**关键约束**：`sessionmaker` 配置 `expire_on_commit=False`，确保 Task 对象在 session 关闭后仍可访问。

## 4. 核心流水线

```
文件名解析 → 加载答案 → [逐页: 渲染→预处理→文本检测→版面分析→手写识别]
→ DP 比对评分 → 按题汇总 → 生成批注 PDF → 保存数据库 → 打印 → 归档
```

### 4.1 文件名契约
`{班级}_{日期}_{序号}.pdf`，如 `301_2026-06-18_003.pdf`。多页 PDF 每页是一个独立学生的答卷。

### 4.2 渲染
PyMuPDF @ 200 DPI → RGB 图像。OCR_DPI = 200。

### 4.3 图像预处理
灰度化 → 低分辨率上采样 → 中值滤波 → CLAHE(clipLimit=3.0) → 倾斜校正(阈值 0.3°) → 模糊检测(方差 < 60 警告)

### 4.4 文本检测
PP-OCRv6 Medium，单次 `predict()` 同时返回 det_boxes 和 ocr_records。
OCR 结果缓存仅存储 `(det_boxes, ocr_records)`，版面分析和手写识别结果不缓存。

### 4.5 版面分析
bbox 高度规则分类（@200 DPI）：
- h ≥ 70px → handwriting
- h ≤ 50px → pinyin_hint
- 其余 → printed_question

题号检测正则：`[（(](\d+)[)）]` | `[①②③④⑤⑥⑦⑧⑨⑩]` | `(\d+)[.、]`

q_idx 偏移量：
- 括号题号 `(1)` → 0-99
- 点号题号 `1.` → 100-199
- 带圈数字 `①` → 200-209

### 4.6 手写识别

**四种引擎**（由 `RECOGNITION_ENGINE` 配置）：
- `paddle` — PaddleOCR bbox 匹配 + 等宽切分
- `qwen_vl` — Qwen3-VL 逐框识别
- `dual` — 双路并行合并
- `page_level`（当前默认）— Qwen3-VL 整页识别，输出逐字归一化 bbox

**坐标映射**（page_level 模式核心）：
1. 模型输出归一化坐标 [0-1]
2. `_norm_to_pixel_bbox()` 转像素坐标
3. Y 坐标修正：模型 y 系统性偏移，用 `_clamp_chars_y_to_region()` 替换为手写框 y 范围
4. X 坐标修正（双重触发）：用 `_rescale_chars_x_to_region()` 检测模型 x 跨度偏大（>1.3×目标）或整体偏移（>80px），线性映射到 marker_text+answer_text 推导的目标范围；偏差小时保留模型 x（保留逐字相对间距）
5. 圈号回退匹配：① → q_idx=200 不存在时，通过标记文本中的 ① 字符查找对应区域
6. 文本回退匹配：无题号格式的 q_marker（如"下联"）通过标记文本内容匹配

**文字渲染模式**（`RENDER_TEXT_MODE=True`）：
- 将识别文字直接渲染到 bbox 位置（覆盖原文）
- 颜色编码：正确=绿，错误=红，存疑=橙
- 透明度 0.7，字号自适应 bbox 高度（8-24pt）
- 应用 y-clamp（模型 y 系统性偏移）+ 条件性 x-rescale（双重触发：跨度>1.3×或偏移>80px）

### 4.7 比对评分
Needleman-Wunsch DP 全局对齐 + 置信度门控：
- 匹配 & conf ≥ 0.85 → correct（绿）
- 匹配 & 0.60 ≤ conf < 0.85 → uncertain（橙）
- 不匹配 & conf < 0.60 → uncertain（橙）
- 不匹配 & conf ≥ 0.60 → wrong（红）

### 4.8 PDF 批注
坐标映射：`x_page = x_pixel / img_w * page_width`，`y_page = page_height - y_pixel / img_h * page_height`

三种模式：
- 逐字模式：每字独立画勾/圈/三角
- 按题模式：全对题画大绿✓，错误字保留标记
- 文字渲染模式：识别文字覆盖原位置

per_char 题型必须逐字标记，即使全对也不用单个大勾。

## 5. 硬约束（不可违反）

1. OCR 缓存只存 `(det_boxes, ocr_records)`，不缓存版面分析和手写识别结果
2. Task 对象在 session 关闭后必须可访问（`expire_on_commit=False`）
3. per_char 题型逐字标记，全对也不用单个大勾
4. Qwen3-VL 识别必须使用原始 PDF 图像，不做预处理
5. 答案解析必须顺序处理，不用字典（防止答案覆盖）
6. 标记坐标：勾用 bbox 右下角，圈/三角用中心坐标

## 6. 工程约定

- 服务层类（PipelineService, AnswerService, TaskService）处理核心业务逻辑；processor.py 是薄包装
- 题型在 layout_analyzer.py 确定，经 grader.py 传播到 pdf_annotator.py
- 文字渲染颜色：正确=绿，错误=红，存疑=橙
- 文字透明度 0.7，字号自适应 bbox 高度（8-24pt）
- 中文字体自动注册（微软雅黑/黑体/宋体）
- 代码变更影响定位时，必须清除 OCR 缓存和旧批注 PDF

## 7. 已知问题与经验教训

- Qwen3-VL y 坐标系统性偏移：多题共享同一 y 值，必须用 `_clamp_chars_y_to_region` 修正
- Qwen3-VL x 坐标双重失效：跨度偏大（双栏当单栏）或整体偏移（平移但跨度正常），用 `_rescale_chars_x_to_region` 双重触发线性映射修正
- Qwen3-VL 可能将 ①② 误读为 (1)(2)，导致 q_idx 不匹配 → 圈号回退匹配
- Qwen3-VL 可能合并拆分题答案（如 (7)(8)）→ 需要后处理拆分
- Qwen3-VL 手写区域含印刷体时会过度识别 → 需精确裁剪
- 旧 OCR 缓存或批注 PDF 会导致持续坐标偏移 → 代码变更后清除缓存

## 8. 测试策略

### 单元测试（pytest）
- `test_grader.py` — DP 对齐 + 置信度门控
- `test_layout_analyzer.py` — bbox 分类 + 题号检测
- `test_qwen_vl_recognizer.py`（待添加）— 坐标映射 + y-clamp + q_idx 回退匹配

### 集成测试
- `run_test.py` — 端到端：PDF → 识别 → 批改 → 批注
- 测试样本：`data/test_samples/` 目录下的 5 个 PDF

### 验证工具（tools/ 目录）
- `_visualize_annotated.py` — 渲染批注 PDF 为 PNG
- `_debug_chars.py` — 数据库坐标检查
- `visualize_questions.py` — 题目区域可视化

## 9. 配置项

| 配置 | 文件 | 当前值 | 说明 |
|------|------|--------|------|
| RECOGNITION_ENGINE | recognition_config.py | `page_level` | 识别引擎选择 |
| RENDER_TEXT_MODE | recognition_config.py | `True` | 文字渲染模式开关 |
| QWEN_VL_USE_DESKEW | recognition_config.py | `False` | VL 路径倾斜校正 |
| OCR_DPI | pipeline_service.py | `200` | 渲染 DPI |
| LOW_CONFIDENCE_THRESHOLD | pipeline_service.py | `0.60` | 低置信度阈值 |
| REVIEW_TRIGGER_RATIO | pipeline_service.py | `0.30` | 需复核触发比例 |
