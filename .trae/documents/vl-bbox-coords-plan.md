# 计划：Qwen3-VL 输出逐字归一化 bbox 坐标

## Context（背景）

当前 `recognize_page_level()` 用"等宽切分"生成假的逐字 bbox：估算 x 范围后按字符数均分，y 直接取整题区间。这导致标记位置系统性偏移——双栏布局时手写框匹配到错误栏、y 覆盖整行而非单字、x 范围靠启发式。

目标：让 Qwen3-VL 直接输出每个手写汉字的归一化 bbox `[0-1]`，recognizer 转成像素坐标后填充现有 per-char dict 契约。grader/annotator 完全不改——真实坐标只会让它们更准。

用户已确认：**逐字符坐标粒度** + **归一化 0-1 格式**。

## 核心决策

1. **移除模型的"批改输出"职责**：新 prompt 只要求识别+坐标，批改仍由 `grader.py` 的 Needleman-Wunsch 完成。省 token、简化解析。
2. **标准答案动态注入**：新增 `load_answers_raw()` 返回原始多行文本，透传到 prompt。当前 prompt 硬编码了 8 道诗句题答案，换作业就失效。
3. **JSON 按题嵌套**：`[{"q":"（1）","chars":[{"c":"随","bbox":[0.12,0.45,0.16,0.52]},...]}]`，省 token 且天然分组。
4. **按题全有/全无回退**：某题所有字都有合法 bbox → 用模型坐标；任一缺失/越界 → 该题整题回退等宽切分。保证可预测。
5. **q_idx 匹配复用 `layout_analyzer._question_match`**：与版面分析同源，避免让 4B 模型自己判断 q_format。匹配失败时用首字 bbox 中心 y 落入哪个 region 做兜底。

## 改动文件

### 1. `core/recognition_config.py`
- 用新模板替换 `QWEN_VL_PAGE_LEVEL_PROMPT`，标准答案段改为 `{answers_text}` 占位符。
- 新增 `QWEN_VL_PAGE_LEVEL_MAX_TOKENS = 4096`（当前 1024 不够 8 题×7字×坐标）。
- 新增 `QWEN_VL_PAGE_LEVEL_TIMEOUT = 120`（4B 模型生成 2000+ token 需要更久，不影响单框识别路径的 30s）。

### 2. `core/qwen_vl_recognizer.py`（主战场）

**重构现有逻辑为 helper（零行为变化，供回退路径复用）：**
- `_estimate_question_x_range(region, handwriting_boxes, img_w) -> (x_start, x_end)`：对应现 474-502 行的列过滤 x 范围估算。
- `_equal_width_split(chars, x_start, x_end, y_start, y_end, page_idx, q_idx, img_w, img_h) -> list[dict]`：对应现 504-528 行。

**新增函数：**
- `_build_page_level_prompt(answers_text) -> str`：模板 + answers_text 拼装，空时回退配置默认答案。
- `_parse_page_level_json(response) -> list[dict] | None`：去 markdown 围栏→截取 `[`..`]`→`json.loads`→校验 bbox 长度4/值域（>1.5 视为 0-1000 量纲，除以1000）→返回 `[{"q_marker","chars":[{"char","bbox_norm"}]}]`，过滤非中文。失败返回 None。
- `_norm_to_pixel_bbox(bbox_norm, img_w, img_h) -> tuple`：`clamp(x,0,1)*img_size`，保证 x0<x1/y0<y1。
- `_question_marker_to_q_idx(marker) -> int | None`：`from core.layout_analyzer import _question_match` 并包装。
- `_match_q_idx_by_bbox(char_bbox_pixel, region_by_qidx, img_w) -> int | None`：首字 bbox 中心 y 匹配 region，x 中心与 marker 列同侧过滤。

**重写 `recognize_page_level()`（379-544行）：**
- 签名加 `answers_text=""` 参数。
- `_call_qwen_vl_page_level(img, answers_text)` 用新 max_tokens/timeout。
- `parsed = _parse_page_level_json(response)`。
- parsed 成功：每题 → `_question_marker_to_q_idx`（失败用 `_match_q_idx_by_bbox`）→ 该题字全有合法 bbox 则 `_norm_to_pixel_bbox` 生成 per-char dict，否则该题回退 `_equal_width_split`。
- parsed 为 None：走旧路径 `_parse_page_level_response` + `_answer_to_q_idx` + `_equal_width_split`（零回归）。
- 保留 region_by_qidx 构建、未匹配统计日志。
- 输出 dict 契约不变：`{page, bbox_pixel, char, confidence, question_idx, img_pixel_w, img_pixel_h, engine}`。

### 3. `services/pipeline_service.py`
- 新增 `load_answers_raw(class_name, date_str) -> str | None`：返回数据库/answers.txt 的原始多行文本。`load_answers` 内部调用它再扁平化。
- `process_pdf`：取 `answers_raw`，传入 `_run_ocr_pipeline`。
- `_run_ocr_pipeline` 签名加 `answers_raw`，透传给 `recognize_page_level(..., answers_text=answers_raw)`。

### 4. `core/layout_analyzer.py`
- 无需改动。`_question_match`（51-79行）被 `qwen_vl_recognizer` 直接 import，无循环依赖。

### 5. 不改动的文件
- `core/grader.py`：契约不变，真实 bbox 让 `summarize_by_question` 的 `answer_bbox`（取 `items[-1]["bbox_pixel"]`）更准。
- `core/pdf_annotator.py`：契约不变，`_pixel_to_page` 已有越界校验，真实坐标经 clamp 后必过。per_char 题型反而更受益于真实逐字 bbox。

## 回退与容错

| 失败场景 | 处理 |
|---|---|
| API 失败/空响应 | 返回 `[]`（同现状） |
| JSON 解析失败 | 回退旧文本解析器 + 等宽切分（零回归） |
| 某题 q 缺失或 _question_match 返回 None | 首字 bbox 中心 y 匹配 region；仍失败跳过 |
| 某题部分字缺 bbox/越界 | 该题整题回退等宽切分 |
| 坐标 >1.5（疑似 0-1000） | 整体除以 1000 再 clamp |
| 坐标 x0>x1 或 y0>y1 | 交换端点 |

## 验证

1. `python -c "import core.qwen_vl_recognizer"` 确认编译通过。
2. 跑 `python run_test.py --pdf 301_2026-06-18_003.pdf`，检查日志：
   - JSON 解析成功题数 vs 回退题数
   - 未匹配区域数
   - 正确/错误字数（应与之前 44 全对一致）
3. 跑 `python _visualize_annotated.py`，肉眼对比 `data/output/301_2026-06-18_003_annotated_page1.png`：
   - (1)-(8) 绿勾应紧贴各字右侧（对比 Qwen 直接批改图）
   - 质朴/绚丽若被模型输出坐标也应标记
4. 若 4B 模型坐标精度差，日志会显示回退题数，可据此评估是否需要降级到答案级坐标方案。
