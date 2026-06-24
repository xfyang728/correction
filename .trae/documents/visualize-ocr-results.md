# 可视化识别结果：题目区域框选 + 题号标注

## 目标
对 `301_2026-06-18_003.pdf` 生成一张可视化图片，类似 `qwen.png`：
- 渲染 PDF 页面作为底图
- 每道题用淡灰色半透明矩形框标出
- 左上角标注题号（1、2、3...）

## 当前状态分析

### 已有数据
- **pipeline_service.py**: `_run_ocr_pipeline()` 返回 `(all_results, question_regions)`
- **question_regions** 结构（layout_analyzer）:
  ```python
  [{"q_idx": 0, "y_start": 100, "y_end": 300,
    "marker_bbox": (x0,y0,x2,y2), "marker_text": "(1)", "column": 0}, ...]
  ```
- **graded_results** 结构:
  ```python
  {"page": 0, "bbox_pixel": (x0,y0,x2,y2), "char": "春",
   "status": "correct", "question_idx": 0, "img_pixel_w": 1654, "img_pixel_h": 2339}
  ```
- **渲染**: fitz `get_pixmap(matrix=fitz.Matrix(200/72, 200/72))` → 像素坐标与 OCR 一致

### qwen.png 参考
- 淡灰色半透明矩形框覆盖每道题的内容区域
- 左上角有白色/深色圆角标签显示题号（1、2、3）
- 框的 x 范围基本覆盖整页内容宽度

## 实现方案

### 创建脚本 `auto_marker/visualize_questions.py`

**步骤 1: 运行流水线获取数据**
```python
from services.pipeline_service import PipelineService
svc = PipelineService()
all_results, question_regions = svc._run_ocr_pipeline(Path("301_2026-06-18_003.pdf"))
```

**步骤 2: 渲染 PDF 页面**
```python
import fitz
doc = fitz.open("301_2026-06-18_003.pdf")
page = doc[0]
zoom = 200 / 72
pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), colorspace=fitz.csRGB)
img_array = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)
```

**步骤 3: 确定每道题的矩形区域**
- 从 `question_regions[0]` 获取所有题的 y 区间
- 按 `(column, y_start)` 排序得到阅读顺序
- x 范围：使用页面宽度（`pix.width`）减去左右边距（如 20px）
- 为每个区域分配显示题号 1, 2, 3...（按阅读顺序）

**步骤 4: 用 OpenCV 绘制**
- 复制底图 `img_array.copy()`
- 对每个题区域：
  - 用 `cv2.rectangle()` 画半透明灰色填充（alpha=0.25, color=(200,200,200)）
  - 在左上角画圆角矩形标签（深色背景），写上题号文字
- 用 `cv2.putText()` 或 PIL 绘制题号文字

**步骤 5: 保存输出**
- 保存到 `data/output/301_2026-06-18_003_visualized.png`

### 关键细节

**题号排序逻辑**:
- `q_idx` 有偏移量：`(1)`→0-99, `1.`→100-199, `①`→200-209
- 按 `(column, y_start)` 排序后，依次分配显示题号 1, 2, 3...

**x 范围确定**:
- 简单方案：使用全页宽度减边距 `x0=20, x2=pix.width-20`
- 可选优化：从 OCR 记录中找该 y 区间内最左/最右的 rec_bbox 来确定实际内容宽度

**题号标签样式**（参考 qwen.png）:
- 深色圆角矩形背景（如深灰色/黑色，alpha=0.7）
- 白色数字文字
- 位置：矩形框左上角内侧

## 涉及文件

| 文件 | 操作 | 说明 |
|------|------|------|
| `auto_marker/visualize_questions.py` | 新建 | 可视化脚本 |
| `auto_marker/services/pipeline_service.py` | 只读 | 获取 question_regions |
| `auto_marker/core/layout_analyzer.py` | 只读 | 理解 question_regions 结构 |

## 验证
1. 运行脚本，检查输出图片中矩形框是否覆盖每道题
2. 题号顺序是否正确（按阅读顺序 1, 2, 3...）
3. 半透明效果是否清晰可见底图文字
