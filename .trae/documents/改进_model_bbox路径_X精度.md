# 改进 model bbox 路径的 X 精度

## 摘要

当前 Qwen3-VL 识别流水线中，model bbox 路径（`_rescale_chars_x_to_region`）仅处理"模型 x 跨度 > 1.3× 手写框跨度"一种失效模式，且调用时未传 `answer_text`，导致等宽切分路径已验证有效的 marker_text 比例裁剪逻辑（Step 2）在 model bbox 路径完全未启用。整体 X 误差 avg≈144px，model bbox 路径是主要误差源。

本方案在 `_rescale_chars_x_to_region` 中启用 marker_text 比例裁剪以获得更精确的目标 x 范围，并新增"偏移超阈"触发条件，覆盖"整体平移但跨度正常"的失效模式。重缩放时保留模型逐字相对间距（线性映射）。

## 当前状态分析

### model bbox 路径的 X 坐标处理链路

```
模型 bbox_norm [0,1]
  → _rescale_chars_x_to_region（仅当跨度>1.3倍时线性映射 x，否则不动）★ 改进点
  → _clamp_chars_y_to_region（只改 y）
  → _norm_to_pixel_bbox（×img_w, ×img_h）
  → bbox_pixel
  → pdf_annotator._pixel_to_page（/img_w*page_w, page_h - /img_h*page_h）
  → PDF 页面坐标
```

### `_rescale_chars_x_to_region` 现状（qwen_vl_recognizer.py L796-874）

- **触发条件**：`model_span_px > hw_span_px * 1.3`（仅跨度偏大）
- **目标范围来源**：`_estimate_question_x_range(region, handwriting_boxes, img_w)` — **未传 `answer_text`**，走 Step 3（窄 marker 回退）或 Step 4（手写框 x 范围），精度较低
- **失效模式覆盖**：
  - ✅ 模型 x 跨度偏大（双栏当单栏）
  - ❌ 模型 x 整体平移但跨度正常
  - ❌ 未利用 marker_text 比例裁剪（Step 2 未启用）

### 调用点（L1368）

```python
rescaled_items = _rescale_chars_x_to_region(
    char_items, region, img_w, handwriting_boxes)
```

`std_chars` 在 L1349 已可用（`std_answers_by_qidx.get(q_idx, [])`），可构造 `answer_text = "".join(std_chars)`。

### 误差数据（test_coordinate_root_cause.py 历史运行）

| 路径 | X avg 误差 | X max 误差 |
|---|---|---|
| Model Raw vs Golden | 145px | — |
| Pipeline vs Golden（改进前） | 144px | 328px |

等宽切分路径改进后，model bbox 路径成为整体 X 误差的主导因素。

## 提议变更

### 变更 1：修改 `_rescale_chars_x_to_region`（qwen_vl_recognizer.py L796-874）

**改什么**：
1. 新增 `answer_text: str | None = None` 参数，透传给 `_estimate_question_x_range` 启用 Step 2（marker_text 比例裁剪）
2. 新增 `offset_threshold_px: float = 80.0` 参数，新增偏移检测条件
3. 触发条件改为双重判断：`span_exceed OR offset_exceed`
   - `span_exceed = model_span_px > hw_span_px * scale_threshold`（现有）
   - `offset_exceed = abs(model_x0_px - hw_x0) > offset_threshold_px`（新增）
4. 增加除零保护：`model_span_norm <= 0` 时跳过该行

**为什么**：
- 传 `answer_text` 启用 Step 2，让目标 x 范围更精确（等宽切分路径已验证：q_idx=0 误差从 -272px 改善到 -40px）
- 新增偏移检测覆盖"整体平移但跨度正常"失效模式
- 线性映射天然同时修正跨度与偏移（当 model_span ≈ hw_span 时，线性映射退化为均匀平移）

**怎么改**（伪代码）：
```python
def _rescale_chars_x_to_region(
    char_items: list[dict],
    region: dict,
    img_w: int,
    handwriting_boxes: list[dict] | None = None,
    scale_threshold: float = 1.3,
    offset_threshold_px: float = 80.0,      # 新增
    answer_text: str | None = None,          # 新增
) -> list[dict]:
    if not char_items:
        return char_items

    # 启用 Step 2：传 answer_text 给 _estimate_question_x_range
    hw_x0, hw_x2 = _estimate_question_x_range(
        region, handwriting_boxes, img_w, answer_text=answer_text)
    if hw_x0 >= hw_x2:
        return char_items

    hw_span_px = hw_x2 - hw_x0

    # 按模型 y 分行（现有逻辑不变）
    rows = ...  # 现有分行逻辑

    rescaled = list(char_items)
    any_rescaled = False
    for row_indices in rows:
        row_items = [char_items[i] for i in row_indices]
        model_x0_norm = min(ci["bbox_norm"][0] for ci in row_items)
        model_x2_norm = max(ci["bbox_norm"][2] for ci in row_items)
        model_span_px = (model_x2_norm - model_x0_norm) * img_w
        model_x0_px = model_x0_norm * img_w

        # 双重判断：跨度或偏移超阈
        span_exceed = model_span_px > hw_span_px * scale_threshold
        offset_exceed = abs(model_x0_px - hw_x0) > offset_threshold_px
        if not (span_exceed or offset_exceed):
            continue

        # 线性映射（保留模型相对间距）— 现有逻辑
        model_span_norm = model_x2_norm - model_x0_norm
        if model_span_norm <= 0:
            continue  # 新增除零保护
        hw_x0_norm = hw_x0 / img_w
        hw_x2_norm = hw_x2 / img_w
        hw_span_norm = hw_x2_norm - hw_x0_norm
        for idx in row_indices:
            ci = char_items[idx]
            x0, y0, x1, y1 = ci["bbox_norm"]
            new_x0 = hw_x0_norm + (x0 - model_x0_norm) / model_span_norm * hw_span_norm
            new_x1 = hw_x0_norm + (x1 - model_x0_norm) / model_span_norm * hw_span_norm
            rescaled[idx] = {**ci, "bbox_norm": (new_x0, y0, new_x1, y1)}
        any_rescaled = True

    if any_rescaled:
        logger.debug(
            "模型 x 已线性映射到目标范围 (%d 字, 目标 x=[%d,%d], 触发: %s)",
            len(char_items), hw_x0, hw_x2,
            "跨度或偏移超阈",
        )
    return rescaled
```

### 变更 2：更新调用点（qwen_vl_recognizer.py L1368-1369）

**改什么**：构造 `answer_text` 并传入

```python
# L1349 已有: std_chars = std_answers_by_qidx.get(q_idx, [])
answer_text = "".join(std_chars) if std_chars else None
rescaled_items = _rescale_chars_x_to_region(
    char_items, region, img_w, handwriting_boxes,
    answer_text=answer_text)
```

**为什么**：`std_chars` 在 L1349 已可用，等宽切分回退路径（L1398）也是这样构造的，保持一致。

### 变更 3：新增单元测试（tests/test_qwen_vl_recognizer.py）

在 `TestEstimateQuestionXRange` 之后新增 `TestRescaleCharsXToRegion` 测试类，覆盖：

| 测试用例 | 场景 | 预期 |
|---|---|---|
| `test_span_exceed_triggers_rescale` | 模型跨度 > 1.3× 目标跨度 | 线性映射到目标范围 |
| `test_offset_exceed_triggers_rescale` | 跨度正常但首字 x 偏移 > 80px | 线性映射（退化为平移） |
| `test_both_ok_preserves_model_x` | 跨度正常且偏移 < 80px | 保留模型 x 原值 |
| `test_answer_text_enables_step2_target` | 传 answer_text，marker_text 含答案 | 目标范围用 Step 2 比例裁剪 |
| `test_no_answer_text_falls_back` | 不传 answer_text | 走 Step 3/4，行为同现状 |
| `test_single_char_skipped` | 单字（model_span_norm=0） | 跳过，不报错 |
| `test_multi_row_rescale` | 跨行题，每行独立判断 | 每行分别映射 |
| `test_empty_chars_returns_empty` | 空输入 | 返回空列表 |

### 变更 4：更新项目规则（.trae/rules/project.md）

**改什么**：更新"X 坐标保留模型值"规则，反映 model bbox 路径现在会主动修正 x 偏移

**当前规则**：
> **X 坐标保留模型值**：x 坐标通常更准，文字渲染模式下不做 x-clamp

**更新为**：
> **X 坐标修正策略**：文字渲染模式下，当模型 x 跨度偏大（>1.3×目标范围）或整体偏移（>80px）时，线性映射到 marker_text+answer_text 推导的目标范围；偏差小时保留模型 x（保留逐字相对间距）

**为什么**：数据表明 model bbox 路径 X 误差 avg≈144px，"x 坐标通常更准"的假设不成立。规则需反映新的修正逻辑，避免后续维护者误以为不应修改 x。

## 假设与决策

1. **线性映射 vs 等宽分布**：用户选择"保留模型相对间距（线性映射）"。理由：模型对字符间距（如标点后空隙）有判断价值，只修正整体偏移和跨度。
2. **触发门槛**：用户选择"双重判断：跨度或偏移超阈"。`offset_threshold_px=80.0`（约为 avg X 误差 144px 的一半，既能捕获系统性偏移，又避免误触发）。
3. **向后兼容**：`answer_text=None` 时行为与现状完全一致（Step 2 不触发，走 Step 3/4）。
4. **`_split_merged_json_chars` 拆分路径不在本次范围**：合并题拆分有独立的 x 处理逻辑，本次只改正常 model bbox 路径。如需改进可后续单独处理。
5. **偏移阈值 80px 的依据**：等宽切分路径改进后 q_idx=0 的残差为 -40px（可接受），q_idx=5 为 +65px（边界）。80px 阈值能捕获大部分系统性偏移而不误触发。
6. **不修改 `_clamp_chars_y_to_region`**：Y 路径已验证有效（avg 46px），本次只改 X。

## 验证步骤

### 1. 单元测试
```bash
cd d:\MyCode\correction\auto_marker
python -m pytest tests/test_qwen_vl_recognizer.py -v
```
预期：原有 40 测试 + 新增 8 测试 = 48 测试全部通过。

### 2. 根因分析测试（需 Qwen3-VL 服务）
```bash
python -m pytest tests/test_coordinate_root_cause.py -v -s
```
预期：Pipeline vs Golden 的 X avg 误差从 ~144px 下降（目标 < 100px）。关注 `test_root_cause_summary` 的 X 统计输出。

### 3. E2E 测试（需 Qwen3-VL 服务）
```bash
python -m pytest tests/test_e2e_annotation.py -v
```
预期：19 测试全部通过。重点关注 `test_x_coordinates_preserved`（首字 x_center 列归属判断）仍通过。

### 4. 清除 OCR 缓存（项目规则要求）
```bash
# 代码变更影响坐标定位时，必须清除 data/cache/ 下的 OCR 缓存
```

### 5. 视觉验证
运行 `tools/_verify_annotation_position.py` 输出字符位置统计，对比 golden 标准的 X 偏差是否整体下降。

## 涉及文件清单

| 文件 | 变更类型 | 行号 |
|---|---|---|
| `auto_marker/core/qwen_vl_recognizer.py` | 修改 `_rescale_chars_x_to_region` | L796-874 |
| `auto_marker/core/qwen_vl_recognizer.py` | 修改调用点传 `answer_text` | L1368-1369 |
| `auto_marker/tests/test_qwen_vl_recognizer.py` | 新增 `TestRescaleCharsXToRegion` 类 | 在 `TestEstimateQuestionXRange` 之后 |
| `.trae/rules/project.md` | 更新 X 坐标规则描述 | "标记坐标映射规则" 章节 |
