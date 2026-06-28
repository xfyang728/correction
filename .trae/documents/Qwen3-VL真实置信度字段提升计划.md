# Qwen3-VL 真实置信度字段提升计划

## 摘要

基于文章《Qwen3-VL能否替代人工标注？图像语义理解部署实操手册》的方法对比评估，本项目当前 Qwen3-VL 路径使用**伪置信度**（0.85/0.90/0.70 规则估算），导致 `green_threshold=0.85` 置信度门控实质失效，无法捕捉"模型高置信但答错"的情况。本计划让 Qwen3-VL 在 prompt 中输出逐字 `conf` 字段，替换伪置信度，使置信度门控真正生效，让复核触发反映真实模型不确定性。

## 当前状态分析（基于 Phase 1 探索）

### 伪置信度问题

**当前 `_estimate_confidence`（[qwen_vl_recognizer.py](file:///d:/MyCode/correction/auto_marker/core/qwen_vl_recognizer.py) L1127-1145）输出规则**：
- invalid → 0.50
- 无标准答案参照 → 0.85
- 字数匹配标准答案 → 0.90
- 字数不匹配 → 0.70

**导致的问题**：
1. `green_threshold=0.85`（[grader.py](file:///d:/MyCode/correction/auto_marker/core/grader.py) L15）实质退化为"字数匹配即 correct"，无法区分模型真实不确定性
2. `low_conf_ratio`（[pipeline_service.py](file:///d:/MyCode/correction/auto_marker/services/pipeline_service.py) L349-352）只在 invalid(0.50) 或字数不匹配(0.70) 时升高，复核触发退化为"字数不匹配比例 >30%"的代理指标
3. 无法捕捉"模型高置信但答错"的危险情况（如模型把"随"识为"随"但实际写的是"谁"）

### 文章方法对比

| 维度 | 文章方法 | 本项目现状 | 提升方向 |
|------|---------|-----------|---------|
| confidence 来源 | 模型真实输出 | 伪置信度（规则估算） | prompt 加 `conf` 字段 |
| 置信度门控 | `confidence<0.85` 路由人工 | `low_conf_ratio>30%` 事后标记 | 门控实质生效 |
| prompt 设计 | JSON + confidence | JSON 无 confidence | 增加 `conf` 字段 |

### 关键文件路径

- [recognition_config.py](file:///d:/MyCode/correction/auto_marker/core/recognition_config.py) L21-35 — prompt 模板
- [qwen_vl_recognizer.py](file:///d:/MyCode/correction/auto_marker/core/qwen_vl_recognizer.py) L241-338 — `_parse_page_level_json`
- [qwen_vl_recognizer.py](file:///d:/MyCode/correction/auto_marker/core/qwen_vl_recognizer.py) L1127-1145 — `_estimate_confidence`
- [qwen_vl_recognizer.py](file:///d:/MyCode/correction/auto_marker/core/qwen_vl_recognizer.py) L1429-1439 — model_json 路径（路径 A）
- [qwen_vl_recognizer.py](file:///d:/MyCode/correction/auto_marker/core/qwen_vl_recognizer.py) L1212-1222 — model_merged_split 路径（路径 B）
- [grader.py](file:///d:/MyCode/correction/auto_marker/core/grader.py) L13-16 — 置信度门控阈值
- [pipeline_service.py](file:///d:/MyCode/correction/auto_marker/services/pipeline_service.py) L28-30, L349-352 — 复核触发逻辑

## 拟议改动

### 改动 1：prompt 模板增加 `conf` 字段

**文件**：[recognition_config.py](file:///d:/MyCode/correction/auto_marker/core/recognition_config.py) L21-35

**做什么**：在 `QWEN_VL_PAGE_LEVEL_PROMPT_TEMPLATE` 的 few-shot 示例和字段说明中加入 `conf` 字段。

**为什么**：让 Qwen3-VL 为每个识别的汉字输出真实置信度，替换伪置信度。

**怎么做**：
- 在字段说明中增加：`"conf" 是该汉字的识别置信度（0.0~1.0），低于 0.7 表示模糊/不确定`
- few-shot 示例更新为：`{"c":"随","bbox":[0.12,0.45,0.16,0.52],"conf":0.95}`
- 强调：`conf` 必须反映模型对该字识别的真实把握程度，不要恒输出高值

**改动后 prompt 模板**：
```
你是手写文字识别助手。图片是一张学生作业扫描页。请完成：
1. 找到每道题的学生手写答案（忽略印刷题目、拼音提示、涂涂画画）。
2. 逐个识别学生"实际书写"的汉字——学生可能写错，必须如实转录，不得纠正或照抄标准答案。
3. 对每个手写汉字，给出它在整页图像中的归一化包围盒 [x0,y0,x1,y1]，
   坐标范围 0~1，左上角(0,0)右下角(1,1)。
4. 对每个汉字给出识别置信度 conf（0.0~1.0）：笔画清晰→0.9+，模糊/涂改→0.7~0.9，难以辨认→<0.7。

只输出 JSON 数组，不要任何解释、评分或多余文字。格式：
[
  {"q":"（1）","chars":[{"c":"随","bbox":[0.12,0.45,0.16,0.52],"conf":0.95},{"c":"君","bbox":[0.16,0.45,0.20,0.52],"conf":0.92}]},
  {"q":"①","chars":[{"c":"质","bbox":[0.31,0.60,0.34,0.67],"conf":0.88}]}
]
其中 "q" 是题号原文，"c" 是识别汉字，"bbox" 是归一化坐标，"conf" 是识别置信度。未作答则 "chars" 为空数组。坐标必须是 0~1 的小数。conf 必须反映真实把握程度，不要恒输出高值。

参考标准答案（仅供识别辅助，务必转录学生实际书写）：
{answers_text}
```

### 改动 2：`_parse_page_level_json` 解析 `conf` 字段

**文件**：[qwen_vl_recognizer.py](file:///d:/MyCode/correction/auto_marker/core/qwen_vl_recognizer.py) L241-338

**做什么**：在解析 `chars[].c` 和 `chars[].bbox` 的同时，解析 `chars[].conf` 字段。

**为什么**：提取模型输出的真实置信度，供后续使用。

**怎么做**：
- 在 L289-328 的 `parsed_chars` 构建循环中，读取 `ch_item.get("conf")`
- 类型检查：必须是数字，值域 [0.0, 1.0]，越界则 clamp
- 缺失或类型错误时：标记该字 `conf=None`（后续回退到伪置信度）
- 在 `parsed_chars.append` 中增加 `"conf": conf_value` 字段

**关键代码片段**（在现有 L327-328 之前插入）：
```python
conf_raw = ch_item.get("conf")
if conf_raw is None:
    conf_value = None  # 标记缺失，后续回退
else:
    try:
        conf_value = float(conf_raw)
        conf_value = max(0.0, min(1.0, conf_value))  # clamp
    except (TypeError, ValueError):
        conf_value = None
```

并在 `parsed_chars.append({"char": c, "bbox_norm": (x0, y0, x1, y1)})` 中增加 `"conf": conf_value`。

### 改动 3：model_json 和 model_merged_split 路径使用模型 conf

**文件**：[qwen_vl_recognizer.py](file:///d:/MyCode/correction/auto_marker/core/qwen_vl_recognizer.py)

**做什么**：在路径 A（L1429-1439）和路径 B（L1212-1222）中，使用模型输出的 `conf` 字段，缺失时回退到 `_estimate_confidence`。

**为什么**：让真实置信度流经主路径，同时保持向后兼容（模型不输出时回退）。

**怎么做**：
- 路径 A（L1429-1439）：`"confidence": ci.get("conf") if ci.get("conf") is not None else conf`（`conf` 是现有 `_estimate_confidence` 计算值）
- 路径 B（L1212-1222）：同样使用 `ci.get("conf")` 回退到 `conf`

**回退策略**：模型未输出 `conf` 字段时，回退到现有 `_estimate_confidence` 伪置信度，保证向后兼容。

### 改动 4：fallback 路径保留伪置信度

**文件**：[qwen_vl_recognizer.py](file:///d:/MyCode/correction/auto_marker/core/qwen_vl_recognizer.py) L1459-1462, L1570-1572

**做什么**：路径 C（fallback_invalid）和路径 D（fallback_text_parse）不来自模型，继续使用伪置信度。

**为什么**：fallback 路径本身不来自模型，无真实置信度可言，保持现有行为。

**怎么做**：无代码改动，仅文档说明。

### 改动 5：单元测试

**文件**：[test_qwen_vl_recognizer.py](file:///d:/MyCode/correction/auto_marker/tests/test_qwen_vl_recognizer.py)

**做什么**：添加 4 个单元测试验证真实置信度字段。

**测试用例**：
1. `test_prompt_template_includes_conf_field`：验证 prompt 模板含 `conf` 字段说明和示例
2. `test_parse_json_extracts_conf`：验证 `_parse_page_level_json` 正确解析 `conf` 字段
3. `test_parse_json_conf_clamp`：验证 `conf` 越界值（如 1.5、-0.3）被 clamp 到 [0,1]
4. `test_parse_json_conf_missing_fallback`：验证 `conf` 缺失时返回 `None`，后续回退到伪置信度

## 假设与决策

### 假设
1. Qwen3-VL-4B-Instruct 能在 JSON 输出中可靠地生成 `conf` 字段（需 E2E 验证）
2. 模型输出的 `conf` 能反映真实识别难度（笔画清晰→高，模糊→低）
3. 增加 `conf` 字段对耗时影响可忽略（每字增加 ~5 token）

### 决策
1. **字段名**：用 `conf`（与 `c`/`bbox` 简洁风格一致），解析后映射到 `confidence`
2. **回退策略**：模型未输出 `conf` 时回退到 `_estimate_confidence` 伪置信度，保证向后兼容
3. **不删除 `_estimate_confidence`**：保留作为 fallback 路径和模型缺失时的回退
4. **不调整阈值**：`green_threshold=0.85` / `orange_threshold=0.60` 保持不变，让真实置信度在现有阈值下生效

### 风险
1. **模型可能恒输出高 conf**：如果 Qwen3-VL 倾向于输出 0.9+，置信度门控仍会失效。需 E2E 验证后决定是否调整 prompt 措辞
2. **耗时增加**：输出 token 数增加，可能影响单页耗时。需对比改动前后耗时
3. **旧缓存不兼容**：OCR 缓存中的记录无 `conf` 字段，需清缓存重新识别

## 验证步骤

### 步骤 1：单元测试
- 运行 `tests/test_qwen_vl_recognizer.py`，确认新增 4 个测试通过 + 现有 60 个测试不回归
- 命令：`cd d:\MyCode\correction\auto_marker && python -m pytest tests/test_qwen_vl_recognizer.py -v`

### 步骤 2：回归测试
- 运行 `tests/test_grader.py` 和 `tests/test_layout_analyzer.py`，确认无回归
- 命令：`cd d:\MyCode\correction\auto_marker && python -m pytest tests/test_grader.py tests/test_layout_analyzer.py -v`

### 步骤 3：清缓存 + E2E 测试
- 清除 `data/cache/` 下的 OCR 缓存（避免旧无 conf 记录影响）
- 运行 `tests/test_e2e_annotation.py`，确认 19 个 E2E 测试通过
- 命令：`cd d:\MyCode\correction\auto_marker && python -m pytest tests/test_e2e_annotation.py -v`

### 步骤 4：验证工具检查置信度分布
- 运行 `tools/_verify_annotation_position.py`，检查输出中置信度是否为模型真实输出（非恒 0.85/0.90）
- 期望：置信度值多样化，反映不同字的识别难度

### 步骤 5：手动验证
- 检查日志 `pipeline_service.py` 输出的 `low_conf_ratio` 是否反映真实不确定性
- 确认 `needs_review` 标记是否在真实低置信度时触发
