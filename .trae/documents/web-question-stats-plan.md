# web/app.py 按题统计展示 实施计划

## 一、目标

在任务详情页的"每页统计"下方，新增"按题统计"区域，展示当前页各题的正确率、正确字数/总字数、是否全部正确。

## 二、当前状态分析

### 数据现状

`task.result_json` 的每项已包含 `question_idx` 字段（由之前 `grader.py` 的 `grade()` 函数保留），格式示例：

```json
[
  {
    "page": 0,
    "char": "春",
    "expected": "春",
    "confidence": 0.92,
    "status": "correct",
    "question_idx": 0,
    ...
  },
  ...
]
```

### Web 界面现状

[web/app.py](file:///d:/MyCode/correction/auto_marker/web/app.py) 在任务详情页已有：
- **每页统计**（第 309-326 行）：汇总每页的字数/正确数/正确率
- **逐字浏览**（第 392-410 行）：按页面和状态过滤展示每字的详细信息

**缺失**：按题统计展示。用户无法看到每个题目的正确率。

### 关键设计决策

**不修改数据库**：直接从 `result_json` 中按 `(page, question_idx)` 分组计算统计，避免 DB schema 变更。

## 三、变更方案

### 涉及文件

只修改 **`web/app.py`**，在现有"每页统计"表格下方新增"按题统计"区域。

### 具体改动

#### 位置

在 "📊 每页统计" 的 `st.dataframe` 之后（第 326 行之后），插入新代码块。

#### 代码设计

```python
# ── 每页按题统计 ──
st.subheader("📝 每页按题统计")

# 从当前页的结果中提取 question_idx
q_groups: dict[int, list[dict]] = {}
for r in page_results_all:  # 当前页所有结果（不过滤状态前）
    q_idx = r.get("question_idx")
    if q_idx is not None:
        q_groups.setdefault(q_idx, []).append(r)

if q_groups:
    q_stats_data = []
    for q_idx in sorted(q_groups.keys()):
        items = q_groups[q_idx]
        total = len(items)
        correct = sum(1 for r in items if r["status"] == "correct")
        uncertain = sum(1 for r in items if r["status"] == "uncertain")
        wrong = sum(1 for r in items if r["status"] == "wrong")
        all_correct = correct == total and total > 0
        q_stats_data.append({
            "题号": f"第 {q_idx + 1} 题",
            "字数": total,
            "正确": correct,
            "存疑": uncertain,
            "错误": wrong,
            "正确率": _fmt_pct(correct, total),
            "状态": "✅ 全部正确" if all_correct else "❌ 有误",
        })
    st.dataframe(
        pd.DataFrame(q_stats_data),
        use_container_width=True,
        hide_index=True,
        column_config={
            "正确率": st.column_config.ProgressColumn(
                "正确率",
                format=".0%",
                min_value=0,
                max_value=1,
            ),
        },
    )
else:
    st.caption("该页无按题分组数据（可能无题号识别结果）")
```

**关键说明**：
- 需要 **先保存当前页全量结果**（未经过滤器筛选前），命名为 `page_all_results`，再按状态筛选
- 或者从之前计算每页统计时使用的 `p_results` 中提取——但 `p_results` 现在是在页面循环内计算的，不能直接复用
- 最佳方案：在页面选择后、状态筛选前，先保存 `page_all_results`

#### 需要调整的现有代码

在状态筛选代码（第 329-333 行）之前，添加一行保存全量结果：

```python
# 第 328 行后插入：
page_all_results = page_results  # 保存过滤前的全量结果

# 第 330 行：过滤 status
page_results = [r for r in page_results if r["status"] == status_filter]  # 已存在，不变
```

## 四、变更清单

| 文件 | 变更类型 | 内容 |
|------|----------|------|
| `web/app.py` | **增强** | 第 ~327 行后插入"每页按题统计"区域；第 ~328 行后添加 `page_all_results` 保存逻辑 |

## 五、边界情况处理

| 场景 | 处理 |
|------|------|
| `result_json` 中无 `question_idx` 字段（老数据） | `q_groups` 为空，显示 `st.caption("该页无按题分组数据")` |
| 某题全部正确 | 进度条满格，状态列显示 ✅ |
| 某题有错误/存疑 | 进度条部分填充，状态列显示 ❌ |
| 某题 total=0 | 不可达（有分组则有字数），防御性检查 `correct == total and total > 0` |

## 六、验证步骤

1. 启动 Web 服务：`streamlit run auto_marker/web/app.py`
2. 选择一个已处理的任务（含 question_idx 数据的）
3. 进入任务详情页，确认"每页按题统计"表格正常显示
4. 切换页面，确认按题统计跟随变化
5. 切换到无 question_idx 的老任务，确认显示"该页无按题分组数据"