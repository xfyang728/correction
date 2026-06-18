# 版面分析增强 + 按题标记 实施计划

## 一、目标

1. **版面分析增强**：检测题目序号（(1)(2)(3)），识别填空横线，区分印刷体题目与手写体答案，按题目对手写答案分组
2. **按题标记**：同一题全部正确 → 在题号旁标一个大绿✓；有错误 → 逐字标红圈/橙三角（保留当前错误标记逻辑）

---

## 二、当前状态分析

### 现有数据流

```
PDF渲染 → OCR(逐页) → 拆分为单字(过滤非中文) → 版面分析(纯中文过滤+分行排序) → DP对齐 → 逐字标记
```

### 关键现状

| 方面 | 现状 | 问题 |
|------|------|------|
| **OCR输出** | `ocr_engine.py` 的 `_split_line_to_chars()` 只保留中文字符，丢弃题目序号 `(1)`、拼音提示 `hú lián` | 题目序号信息丢失，无法做按题分组 |
| **版面分析** | `layout_analyzer.py` 的 `_is_pure_chinese()` 进一步过滤非中文，仅按 `(y, x)` 排序 | 没有题目概念 |
| **比对** | `grader.py` 的 `grade()` 将每页所有手写字与答案平面做DP对齐 | 正确，但不知道哪些字属于同一题 |
| **批注** | `pdf_annotator.py` 对每个字画勾/圈/三角 | 标记密集，不符合老师按题批改习惯 |

### 关键文件依赖关系

```
processor.py
  ├── ocr_image(img, page_idx) → list[dict]        # ocr_engine.py
  ├── extract_student_answers(results, w, h)        # layout_analyzer.py
  ├── grade(ocr_results, answers) → list[dict]      # grader.py
  └── annotate(pdf_path, graded, output_dir)        # pdf_annotator.py
```

---

## 三、变更方案

### 3.1 `core/ocr_engine.py` — 保留行级 OCR 数据

**改动**：`ocr_image()` 除返回逐字结果外，额外返回行级原始数据。

```python
# 当前返回:
return results  # list[dict], 每项一个汉字

# 改为返回:
return results, raw_lines  # (list[dict], list[dict])
```

`raw_lines` 每项格式：
```python
{
    "page": page_idx,
    "text": "(1) hú lián",       # 整行原文
    "bbox_pixel": (x0, y0, x2, y2),
    "confidence": 0.98,
    "type": "printed",           # 由 layout_analyzer 填充
}
```

**为什么这么改**：
- 行级数据包含题目序号 `(1)` 和拼音提示，这些是检测题目边界的关键信息
- 逐字数据保持不变，不影响 grader（仅需微调 processor.py 传参）

### 3.2 `core/layout_analyzer.py` — 新增题目检测 + 分组

新增 3 个函数：

#### `detect_question_regions(raw_lines: list[dict], img_h: int) -> list[dict]`

扫描行级 OCR 数据，识别题目序号：

```
检测逻辑：
  1. 对每行文本匹配正则 r'^\(\d+\)' 或 r'^\d+[.、]'
  2. 取匹配行的 bbox y 范围作为题目区域
  3. 第 N 题区域 = (题号_上边界, 下一个题号_上边界 或 页底)
  4. 也保留该行上其它文字（拼音提示）的区域

返回:
  [
    {"q_idx": 0, "y_start": 100, "y_end": 300, "marker_bbox": (x0,y0,x2,y2)},
    {"q_idx": 1, "y_start": 300, "y_end": 500, "marker_bbox": ...},
    ...
  ]
```

回退策略：如果一行都未匹配到题号模式（`r'^\(\d+\)'`），则：
- **尝试答案结构回退**：将答案按换行符 `\n` 拆分为多组
- 按每组字数等分当前页的手写字（例如答案 3 行共 6 字 → 3 题每题 2 字）
- 这种情况只在 Web 界面答案输入时用换行分隔时生效

#### `assign_to_questions(handwritten_items: list[dict], question_regions: list[dict]) -> dict[int, list[dict]]`

```python
# 对每个手写字，根据其 bbox 中心 y 坐标落到哪个题目区域
# 返回: {0: [item1, item2], 1: [item3, item4], ...}
```

#### 修改 `extract_student_answers()`

```python
def extract_student_answers(ocr_results, raw_lines, img_w, img_h):
    # 1. 用 raw_lines 检测题目区域 (detect_question_regions)
    # 2. 从 ocr_results 过滤出纯汉字项
    # 3. 按 page 分组后，对每页调用 assign_to_questions
    # 4. 返回 (sorted_handwritten_items, question_groups)
    #
    # question_groups: {page_idx: {q_idx: [item_indices]}}
```

### 3.3 `core/grader.py` — 新增按题汇总函数

新增 `summarize_by_question()`：

```python
def summarize_by_question(graded: list[dict],
                          question_groups: dict[int, list[int]]) -> dict:
    """对 graded 结果按题目分组统计。

    question_groups: {q_idx: [graded_indices]}
    返回:
    {q_idx: {
        "total": 2,
        "correct": 2,
        "uncertain": 0,
        "wrong": 0,
        "all_correct": True,
        "marker_bbox": (x0,y0,x2,y2),  # 题号位置（用于标记定位）
    }}
    """
```

### 3.4 `core/pdf_annotator.py` — 支持按题标记模式

**改动**：`annotate()` 新增可选参数 `question_summary`。

```python
def annotate(original_pdf, graded_results, output_dir=None,
             question_summary=None):
```

当 `question_summary` 不为 None 时：

```
每页绘制逻辑：
  for each question in page:
    获取题号 bbox → 题号右侧留白处定位
    if question["all_correct"]:
      画一个大绿 ✓ (比单字勾大 1.5 倍)
    else:
      # 有错误 — 仍然逐字标记（当前逻辑）：
      for each char in question:
        if wrong:  红圈
        if uncertain: 橙三角
        if correct: 不画（减少视觉噪音，且已有题号绿✓）
```

**绿色对号定位**：在题号 bbox 的右侧居中位置绘制，半径 ≈ 单字勾 × 1.5。

### 3.5 `monitor/processor.py` — 串联新数据流

```
修改 process_pdf():

1. OCR 阶段:
   - page_results, page_raw = ocr_image(img_np, page_idx)
   - all_ocr_results.extend(page_results)
   - all_raw_results.extend(page_raw)

2. 版面分析阶段:
   - student_answers, question_groups = extract_student_answers(
       all_ocr_results, all_raw_results, ref_w, ref_h
     )

3. 比对阶段（不变）

4. 按题汇总:
   - q_summary = summarize_by_question(graded, question_groups)

5. 批注阶段:
   - annotate(str(path), graded, str(output_dir), question_summary=q_summary)
```

### 3.6 `web/app.py` — 显示按题统计

在任务详情页新增"按题统计"区域：

```python
# 在每页统计下方，展示该页各题的正确率
for q_idx, q_data in question_summary.items():
    cols = st.columns([1, 3, 1, 1])
    cols[0].write(f"第 {q_idx+1} 题")
    cols[1].progress(q_data["correct"] / q_data["total"])
    cols[2].write(f"{q_data['correct']}/{q_data['total']}")
    cols[3].write("✅" if q_data['all_correct'] else "❌")
```

---

## 四、变更清单汇总

| 文件 | 变更类型 | 内容 |
|------|----------|------|
| `core/ocr_engine.py` | **修改** | `ocr_image()` 额外返回行级 raw_lines |
| `core/layout_analyzer.py` | **重写** | 新增 `detect_question_regions()`、`assign_to_questions()`、修改 `extract_student_answers()` 返回分组信息 |
| `core/grader.py` | **新增函数** | 新增 `summarize_by_question()` |
| `core/pdf_annotator.py` | **修改** | `annotate()` 支持 `question_summary` 参数，实现按题绿✓ + 错误逐字标记 |
| `monitor/processor.py` | **修改** | 串联 raw_lines 和 question_groups 在新数据流中 |
| `web/app.py` | **增强** | 任务详情页展示按题统计 |

---

## 五、边界情况处理

| 场景 | 处理 |
|------|------|
| 单页无题号（自由书写） | question_regions 为空，回退到全页逐字标记（当前行为） |
| 题号检测失败 | 回退到答案结构（换行符分隔）或全页逐字标记 |
| 某题 OCR 全空 | q_summary 中标记为 `all_correct=False`，标红✗ |
| 一题内手写字 row_span 跨区域 | 按中心 y 坐标归属最近的题号区域 |
| 已有历史任务（数据库中 result_json 无 question 字段） | 向后兼容：question_summary=None 时保持逐字标记 |

---

## 六、验证

```bash
# 1. 旧有单页 PDF — 逐字标记不变（无题号检测）
cd d:\MyCode\correction\auto_marker
python run_test.py -p 301_2025-03-20_001.pdf -c 301 -d 2025-03-20

# 2. 答题卡 PDF — 应显示按题绿✓ + 错误逐字红圈
python run_test.py -p 301_2026-06-18_001.pdf -c 301 -d 2026-06-18

# 3. 对比批注 PDF，确认：
#   - 正确题号旁有绿✓（比单字勾大）
#   - 错误字仍有红圈
#   - 正确字不再有单字勾（减少噪音）
```