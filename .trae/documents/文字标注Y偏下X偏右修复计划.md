# 文字标注 Y 偏下 / X 偏右修复计划

## 概述

流水线生成的 `301_2026-06-18_003_annotated.pdf` 与人工标注 `301_2026-06-18_003-人工标注box.pdf` 相比存在两个系统性偏差：
1. **文字标注位置整体偏下** — 渲染文字位于手写文字下方
2. **第一个字整体偏右** — 每题首字 x 坐标偏大

本计划基于全网搜索（Qwen3-VL 坐标系、PDF CJK 渲染基线、`unicodedata.east_asian_width()` 字符宽度）和代码探索，给出最小改动的精准修复方案。

---

## 现状分析（根因定位）

### Y 偏下根因（双重叠加）

**主因：`pdf_annotator.py` L116/L121 基线公式失效**

[code](file:///d:/MyCode/correction/auto_marker/core/pdf_annotator.py#L106-L134)

```python
font_size = max(min(bh * 0.9, 24), 8)        # L116: 大 bbox 时被截断到 24pt
text_y = min(y0_page, y1_page) + bh * 0.2    # L121: 基线固定在底+20%bh
```

问题推导：
- 当 `bh` 较大（如 80pt）时，`font_size = 24`（被截断），`bh * 0.8` 远小于 `font_size * 0.8`
- `text_y = bbox_bottom + 0.2 * 80 = bbox_bottom + 16` （基线）
- 文字顶部 ≈ `text_y + 0.8 * 24 = bbox_bottom + 35.2`
- 文字位于 bbox 的 20%-44% 区间（下半部），而手写文字实际位于 bbox 中心区域
- 视觉上文字"偏下"

**次因：`qwen_vl_recognizer.py` L939-941 y-clamp replace 丢失逐字 y 相对信息**

[code](file:///d:/MyCode/correction/auto_marker/core/qwen_vl_recognizer.py#L936-L947)

```python
if y1 <= norm_y0 or y0 >= norm_y1:
    # 情况 1：模型 y 完全在 region 外（系统性偏移）→ replace
    clamped.append({**ci, "bbox_norm": (x0, norm_y0, x1, norm_y1)})
```

问题：所有字都被替换为同一个 `(norm_y0, norm_y1)` 范围，丢失了模型输出的逐字 y 差异（虽然 Qwen3-VL 的 y 精度有限，但仍有部分相对信息）。这导致每题所有字共用同一 bh，font_size 截断效应被放大。

### X 偏右根因

**主因：`qwen_vl_recognizer.py` L592-595 等宽假设高估 ASCII 字符宽度**

[code](file:///d:/MyCode/correction/auto_marker/core/qwen_vl_recognizer.py#L586-L607)

```python
total_chars = len(marker_text)          # 如 "(1) 随君直到夜郎西" = 9 字
m_w = marker_bbox[2] - marker_bbox[0]
char_w = m_w / total_chars              # 等宽：m_w / 9
x_start = int(marker_bbox[0] + answer_start * char_w)  # answer_start=4 (随的位置)
```

问题：`marker_text` 混合了 ASCII（如 `"(1) "`）和 CJK（如 `"随君直到夜郎西"`）。CJK 字符实际宽度约为 ASCII 的 2 倍，等宽假设高估了 ASCII 字符的宽度，导致 `answer_start * char_w` 偏大，`x_start` 偏右。

**次因：`_rescale_chars_x_to_region` L894 首字映射到 hw_x0**

[code](file:///d:/MyCode/correction/auto_marker/core/qwen_vl_recognizer.py#L891-L896)

```python
new_x0 = hw_x0_norm + (x0 - model_x0_norm) / model_span_norm * hw_span_norm
```

修复 X 偏右主因后，`hw_x0` 更精确，首字映射自动正确，无需额外改动。

---

## 提议改动

### 改动 1：修复 Y 偏下主因 — `_draw_text_at_bbox` 基线垂直居中

**文件**: [pdf_annotator.py](file:///d:/MyCode/correction/auto_marker/core/pdf_annotator.py#L106-L135)

**改动**: 修改 `text_y` 公式，使文字在 bbox 内垂直居中（基于 `font_size` 动态计算，而非固定 `bh * 0.2`）。

**Why**: 当 `font_size` 被 24pt 上限截断且 `bh` 较大时，固定 20% 偏移使文字偏下；垂直居中公式 `text_y = bbox_bottom + (bh - font_size * ascender_ratio) / 2` 在任何 `bh` 下都能让文字视觉居中。

**How**:

```python
# 修改前 (L116-L121):
font_size = max(min(bh * 0.9, 24), 8)
text_x = min(x0_page, x1_page)
text_y = min(y0_page, y1_page) + bh * 0.2

# 修改后:
font_size = max(min(bh * 0.9, 24), 8)
text_x = min(x0_page, x1_page)
# 垂直居中：基线 = bbox_bottom + (bh - font_size * ascender_ratio) / 2
# 中文字体 ascender ratio ≈ 0.8，文字视觉中心在基线上方 0.4*font_size
# 使文字视觉中心 = bbox 中心
text_y = min(y0_page, y1_page) + (bh - font_size * 0.8) / 2
```

**数学验证**:
- 文字视觉中心 = `text_y + 0.4 * font_size`（基线 + 半字高）
- 代入：`bbox_bottom + (bh - 0.8*font_size)/2 + 0.4*font_size = bbox_bottom + bh/2` ✓
- 与 bbox 中心 `bbox_bottom + bh/2` 重合，垂直居中

### 改动 2：修复 X 偏右主因 — `_estimate_question_x_range` Step 2 按字符宽度比例计算

**文件**: [qwen_vl_recognizer.py](file:///d:/MyCode/correction/auto_marker/core/qwen_vl_recognizer.py#L586-L607)

**改动**: 在 Step 2 中用 `unicodedata.east_asian_width()` 区分 CJK（Fullwidth=2）和 ASCII（Narrow=1），按实际宽度比例计算 `x_start` 和 `x_end`。

**Why**: `marker_text` 如 `"(1) 随君直到夜郎西"` 含 4 个 ASCII 字符和 5 个 CJK 字符。等宽假设 `m_w/9` 高估 ASCII 宽度，导致 `answer_start=4` 对应的 x 偏右。CJK 实际宽度约为 ASCII 的 2 倍，按宽度比例计算可精准定位答案起始。

**How**:

```python
# 新增辅助函数（在 _estimate_question_x_range 上方）:
import unicodedata

def _char_width_weight(c: str) -> int:
    """返回字符显示宽度权重：CJK/全角=2，ASCII/半角=1。

    用于 marker_text 中混合 ASCII（题号 "(1) "）和 CJK（答案 "随君..."）的
    宽度比例计算，避免等宽假设导致 x 偏移。
    """
    w = unicodedata.east_asian_width(c)
    return 2 if w in ('F', 'W', 'A') else 1

# 修改 _estimate_question_x_range Step 2 (L589-L607):
# 修改前:
total_chars = len(marker_text)
m_w = marker_bbox[2] - marker_bbox[0]
char_w = m_w / total_chars
x_start = int(marker_bbox[0] + answer_start * char_w)
answer_chars = len(answer_text)
x_end = int(x_start + answer_chars * char_w)

# 修改后:
# 按字符宽度权重计算（CJK=2, ASCII=1），更准确反映实际显示宽度
total_weight = sum(_char_width_weight(c) for c in marker_text)
m_w = marker_bbox[2] - marker_bbox[0]
weight_per_pixel = m_w / total_weight if total_weight > 0 else 0
answer_start_weight = sum(_char_width_weight(c) for c in marker_text[:answer_start])
x_start = int(marker_bbox[0] + answer_start_weight * weight_per_pixel)
answer_weight = sum(_char_width_weight(c) for c in answer_text)
x_end = int(x_start + answer_weight * weight_per_pixel)
```

**示例验证** (`marker_text = "(1) 随君直到夜郎西"`, `answer_text = "随君直到夜郎西"`, `m_w = 200px`):
- 修改前: `char_w = 200/9 ≈ 22.2`, `x_start = 0 + 4*22.2 = 88.9`
- 修改后: `total_weight = 4*1 + 5*2 = 14`, `weight_per_pixel = 200/14 ≈ 14.3`, `answer_start_weight = 4*1 = 4`, `x_start = 0 + 4*14.3 = 57.1`
- 差异：88.9 - 57.1 = 31.8px（修改后更靠左，修正"偏右"问题）

### 改动 3：修复 Y 偏下次因 — `_clamp_chars_y_to_region` 情况 1 保留模型 y 相对信息

**文件**: [qwen_vl_recognizer.py](file:///d:/MyCode/correction/auto_marker/core/qwen_vl_recognizer.py#L936-L947)

**改动**: 情况 1 (replace) 不再一刀切替换为 `(norm_y0, norm_y1)`，而是按比例映射模型 y 到 region y，保留逐字相对位置。

**Why**: 当前 replace 后所有字共用同一 bh，font_size 截断效应被放大。保留模型 y 相对信息后，不同字的 bh 会有差异，更接近实际手写位置。

**How**:

```python
# 修改前 (L936-L947):
clamped = []
for ci in char_items:
    x0, y0, x1, y1 = ci["bbox_norm"]
    if y1 <= norm_y0 or y0 >= norm_y1:
        # 情况 1：模型 y 完全在 region 外（系统性偏移）→ replace
        clamped.append({**ci, "bbox_norm": (x0, norm_y0, x1, norm_y1)})
    else:
        # 情况 2/3：部分重叠或完全在内 → clamp
        clamped_y0 = max(y0, norm_y0)
        clamped_y1 = min(y1, norm_y1)
        clamped.append({**ci, "bbox_norm": (x0, clamped_y0, x1, clamped_y1)})
return clamped

# 修改后:
# 预计算模型 y 跨度（用于判断是否有逐字相对信息）
model_y_min = min(ci["bbox_norm"][1] for ci in char_items)
model_y_max = max(ci["bbox_norm"][3] for ci in char_items)
model_y_span = model_y_max - model_y_min
region_y_span = norm_y1 - norm_y0

clamped = []
for ci in char_items:
    x0, y0, x1, y1 = ci["bbox_norm"]
    if y1 <= norm_y0 or y0 >= norm_y1:
        # 情况 1：模型 y 完全在 region 外（系统性偏移）
        if model_y_span > 0.01 and region_y_span > 0:
            # 1a: 模型有逐字 y 相对信息 → 按比例映射到 region
            # 保留字的相对高低位置，bh 更贴近实际
            new_y0 = norm_y0 + (y0 - model_y_min) / model_y_span * region_y_span
            new_y1 = norm_y0 + (y1 - model_y_min) / model_y_span * region_y_span
            # 保证 y0 < y1 且在 region 内
            new_y0 = max(norm_y0, min(norm_y1, new_y0))
            new_y1 = max(norm_y0, min(norm_y1, new_y1))
            if new_y1 <= new_y0:
                new_y0, new_y1 = norm_y0, norm_y1
            clamped.append({**ci, "bbox_norm": (x0, new_y0, x1, new_y1)})
        else:
            # 1b: 模型 y 无相对信息（多题共享同一 y）→ 直接 replace
            clamped.append({**ci, "bbox_norm": (x0, norm_y0, x1, norm_y1)})
    else:
        # 情况 2/3：部分重叠或完全在内 → clamp
        clamped_y0 = max(y0, norm_y0)
        clamped_y1 = min(y1, norm_y1)
        clamped.append({**ci, "bbox_norm": (x0, clamped_y0, x1, clamped_y1)})
return clamped
```

**注**: 当模型 y 跨度 < 0.01（多题共享同一 y）时，回退到原 replace 行为，避免引入噪声。

---

## 假设与决策

1. **不修改 `font_size` 截断上限 24pt** — 保留可读性约束，通过垂直居中公式解决偏下问题
2. **不添加 A/B 测试开关** — 改动幅度小且可验证，直接修改现有函数；`Y_CLAMP_ENABLED` 已存在可用于回退
3. **`_rescale_chars_x_to_region` 无需额外改动** — 修复改动 2 后 `hw_x0` 更精确，首字映射自动正确
4. **`unicodedata.east_asian_width()` 返回值处理** — `F`(Fullwidth)/`W`(Wide)/`A`(Ambiguous) 视为 CJK（权重 2），其他视为 ASCII（权重 1）
5. **改动 3 的回退阈值 0.01** — 当模型 y 跨度 < 0.01（归一化值）时视为无相对信息，回退到原 replace 行为

---

## 验证步骤

### 步骤 1: 单元测试

运行现有测试确保无回归：

```bash
cd d:\MyCode\correction\auto_marker
python -m pytest tests/test_qwen_vl_recognizer.py -v
python -m pytest tests/test_grader.py -v
python -m pytest tests/test_layout_analyzer.py -v
```

### 步骤 2: 新增针对性单元测试

在 `tests/test_qwen_vl_recognizer.py` 中新增：

- `TestEstimateQuestionXRange`:
  - `test_mixed_ascii_cjk_marker_text`: 验证 `"(1) 随君直到夜郎西"` 的 x_start 小于等宽假设
  - `test_all_cjk_marker_text`: 全 CJK marker_text 行为不变
  - `test_all_ascii_marker_text`: 全 ASCII marker_text 行为不变
- `TestClampCharsYToRegion`:
  - `test_replace_preserves_relative_y`: 模型 y 有相对信息时按比例映射
  - `test_replace_no_relative_y`: 模型 y 无相对信息时回退到原 replace

在 `tests/test_pdf_annotator.py`（如存在）或新建中新增：
- `TestDrawTextAtBbox`:
  - `test_text_vertical_centering`: 验证 `text_y` 使文字视觉中心 = bbox 中心
  - `test_font_size_capped`: bh 较大时 font_size=24，text_y 仍能垂直居中

### 步骤 3: 清除 OCR 缓存并重新生成

```bash
# 清除缓存（坐标变更影响定位，必须清除旧缓存）
Remove-Item d:\MyCode\correction\auto_marker\data\cache\* -Recurse -Force
```

### 步骤 4: E2E 测试 + 视觉对比

```bash
cd d:\MyCode\correction\auto_marker
python -m pytest tests/test_e2e_annotation.py -v
python tools/_verify_annotation_position.py
```

### 步骤 5: 人工视觉验证

对比以下两个文件：
- 流水线输出: `d:\MyCode\correction\auto_marker\data\output\301_2026-06-18_003_annotated.pdf`
- 人工标注: `d:\MyCode\correction\auto_marker\data\test_samples\301_2026-06-18_003-人工标注box.pdf`

预期改善：
- 文字标注垂直居中于手写框，不再偏下
- 每题首字 x 坐标准确对齐答案起始位置，不再偏右

### 步骤 6: 坐标精度统计

运行 `tools/_verify_annotation_position.py`，对比修改前后的字符位置统计：
- Y 范围应更紧凑（改动 3 保留逐字 y 差异）
- X 范围首字应更靠左（改动 2 修正 ASCII 宽度高估）

---

## 影响范围

| 文件 | 改动 | 影响函数 |
|------|------|----------|
| `core/pdf_annotator.py` | 改动 1 | `_draw_text_at_bbox` (L92-L135) |
| `core/qwen_vl_recognizer.py` | 改动 2 | `_estimate_question_x_range` Step 2 (L586-L607) + 新增 `_char_width_weight` |
| `core/qwen_vl_recognizer.py` | 改动 3 | `_clamp_chars_y_to_region` 情况 1 (L936-L947) |
| `tests/test_qwen_vl_recognizer.py` | 新增测试 | `TestEstimateQuestionXRange` + `TestClampCharsYToRegion` 扩展 |
| `tests/test_pdf_annotator.py` | 新增测试 | `TestDrawTextAtBbox` |

**不受影响**:
- `_rescale_chars_x_to_region` — 自动受益于改动 2，无需修改
- `_estimate_question_y_range` — 不修改
- `_split_merged_json_chars` — 不修改
- `grader.py` / `layout_analyzer.py` — 不修改
