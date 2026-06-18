# 方案：PP-OCRv6 Medium 升级 + 多学生合订批改

## 一、目标

1. **PP-OCRv5_Server → PP-OCRv6_Medium**：精度 +5.1%，速度 **×5.2**（1.40s/图 vs ~70s/页），34.5M 轻量模型
2. **多页 PDF = 多学生合订**：11 页 PDF 按页分组，每页独立与答案做 DP 对齐
3. **保留现有所有功能**：可观测性、人工复核、统计看板

---

## 二、PP-OCRv6 关键信息

| 项目 | 值 |
|------|-----|
| paddleocr 版本 | ≥ 3.6.0（2026-05-28 发布） |
| paddlepaddle 版本 | ≥ 3.2.1 |
| 模型 | PP-OCRv6_Medium（34.5M，自动下载） |
| API | `predict(input=img)` 与 PP-OCRv5 兼容 |
| 速度 | Intel Xeon CPU 1.40s/图（v5_server 的 **5.2 倍**） |
| 精度 | 检测 +4.9%，识别 +5.1% |

---

## 三、变更清单（5 个文件 + 2 个文档）

### 文件 1：`requirements.txt` — 依赖升级

```diff
- paddlepaddle>=2.5,<3.0
- paddleocr>=2.8,<3.0
+ paddlepaddle>=3.2.1,<4.0
+ paddleocr>=3.6.0,<4.0
```

### 文件 2：`core/ocr_engine.py` — PP-OCRv6 Medium

- paddleocr 3.6.0 初始化和 API 与 3.0 基本一致
- `PaddleOCR(det_model="PP-OCRv6_medium", rec_model="PP-OCRv6_medium")`
- 返回格式兼容：`raw_results` 的 `res.json.get("res")` 结构与 v5 兼容
- `ocr_image()` 接口不变
- 仍然需要 `_get_ocr()` 单例 + `_split_line_to_chars()` 拆分整行

### 文件 3：`core/image_processor.py` — 增强预处理

针对手拍试卷的特定特征优化：

| 手拍试卷问题 | 当前处理 | PP-OCRv6 增强 |
|-------------|---------|---------------|
| 倾斜严重 | 降采样检测，>0.5° 才校正 | **阈值降低到 0.3°**，强制校正 |
| 光照不均 | CLAHE | CLAHE clipLimit 从 2.0 → 3.0 |
| 阴影/折痕 | 中值滤波 3x3 | 中值滤波 3x3（不变，够用） |
| 低分辨率 | 无 | 检测原图尺寸 < 1000px 时自动上采样 |
| 模糊 | 拉普拉斯告警 | 降低 BLUR_THRESHOLD（手拍通常偏模糊） |

### 文件 4：`monitor/processor.py` — 按页分组批改

OCR + 版面分析后 → 按 `page` 字段分组 → 每页独立 `grade()`：

```python
from collections import defaultdict

student_answers = extract_student_answers(all_ocr_results, ...)

# 按页分组（每页 = 一个学生）
page_answers = defaultdict(list)
for r in student_answers:
    page_answers[r["page"]].append(r)

# 每页独立与答案做 DP 对齐
graded = []
for page_idx in sorted(page_answers.keys()):
    page_graded = grade(page_answers[page_idx], answers)
    graded.extend(page_graded)
```

### 文件 5：`web/app.py` — 每页统计展示

在任务详情页新增"每页统计"卡片区域：

```python
# 按 page 计算每页正确率
page_stats = defaultdict(lambda: {"total": 0, "correct": 0})
for r in results:
    p = r["page"]
    page_stats[p]["total"] += 1
    if r["status"] == "correct":
        page_stats[p]["correct"] += 1

# 展示每页统计表格
stats_df = pd.DataFrame([
    {"页码": f"第 {p+1} 页", "字数": s["total"],
     "正确": s["correct"], "正确率": f"{s['correct']/s['total']:.0%}" if s["total"] else "-"}
    for p, s in sorted(page_stats.items())
])
st.subheader("📊 每页统计")
st.dataframe(stats_df, use_container_width=True, hide_index=True)
```

---

## 四、更新文档

### 4.1 `架构设计.md`

- 批改引擎描述：`PP-OCRv6_Medium`
- 数据流图增加"按页分组"步骤
- 文件名约定增加多学生合订说明
- 性能预估更新：单页 OCR 1.4s（CPU）

### 4.2 `README.md`

- 依赖：paddleocr >= 3.6.0
- 批改流水线步骤 7：按页分组批改
- 性能指标更新

---

## 五、预期效果

| 指标 | 当前 (v5_server) | 升级后 (v6_medium) |
|------|-----------------|-------------------|
| 单页 OCR 耗时 | ~70s | **~1.4s**（×50 加速） |
| 11 页总耗时 | ~800s | **~30s** |
| 识别精度 | baseline | **+5.1%** |
| 模型大小 | ~130MB | **34.5MB** |
| 多学生合订 | 全部合并对齐 ❌ | 逐页独立对齐 ✅ |

---

## 六、验证

```bash
# 1. 旧试卷（1 页）→ 正确率不变
python run_test.py --pdf 301_2025-03-20_001.pdf

# 2. 新试卷（11 页）→ 每页独立对齐，正确率大幅提升
python run_test.py --pdf D:\MyCode\correction\pictures\301_2026-06-18_001.pdf
```

---