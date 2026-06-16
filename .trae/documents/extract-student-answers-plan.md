# 方案：四步工程化提取学生手写答案

## 一、问题分析

### 现状

当前流水线将 EasyOCR 的**全部检测结果**直接送入比对器，混合印刷体（拼音）和手写体（答案），导致全错。

### 根因

扫描件 `301_2025-03-20_001.pdf` 是"看拼音写词语"练习题，页面结构：

```
Row 1 (y≈422-500):  [拼音提示 ×4] 上半 → [手写答案 ×4] 下半
Row 2 (y≈665-760):  [拼音提示 ×4] 上半 → [手写答案 ×4] 下半
Row 3 (y≈914-984):  [拼音提示 ×4] 上半 → [手写答案 ×4] 下半
Row 4 (y≈1153-1248): [拼音提示 ×4] 上半 → [手写答案 ×4] 下半
```

缺少：图像预处理、版面分析、手写/印刷体分离三个环节。

***

## 二、架构变更

```
原流水线：
  PDF → OCR(全页) → 比对 → 批注

新流水线：
  PDF → 图像预处理 → 版面分析(分行/分元素) → OCR(逐区域) → 答案提取 → 比对 → 批注
```

***

## 三、具体变更（4 个文件）

### 变更 1：新建 `core/image_processor.py` — 图像预处理

对 PDF 渲染后的页面图片进行处理，提升 OCR 质量：

```python
def preprocess_image(img: PIL.Image) -> PIL.Image:
    """
    输入：PIL Image (RGB)
    输出：预处理后的 PIL Image (RGB)
    
    处理流程：
    1. 灰度化 + 去噪 → cv2.fastNlMeansDenoising / medianBlur
    2. 二值化 → cv2.adaptiveThreshold / OTSU
    3. 对比度增强 → cv2.equalizeHist / CLAHE
    4. 倾斜校正 → 检测文本轮廓最小外接矩形，计算旋转角度并校正
    5. 质量筛查 → 计算拉普拉斯方差（模糊检测），低于阈值则跳过
    """
```

**具体实现：**

| 步骤    | 方法                                                            | OpenCV 函数                                |
| ----- | ------------------------------------------------------------- | ---------------------------------------- |
| 去噪    | 中值滤波（核 3×3）                                                   | `cv2.medianBlur()`                       |
| 二值化   | OTSU 自适应阈值                                                    | `cv2.threshold(..., cv2.THRESH_OTSU)`    |
| 对比度增强 | CLAHE（限制对比度自适应直方图均衡）                                          | `cv2.createCLAHE()`                      |
| 倾斜校正  | 检测所有文本轮廓，用 `minAreaRect` 计算整体倾斜角，`cv2.getRotationMatrix2D` 校正 | `cv2.minAreaRect()` → `cv2.warpAffine()` |
| 质量筛查  | 拉普拉斯方差 < 阈值则标注为"模糊页"                                          | `cv2.Laplacian().var()`                  |

***

### 变更 2：新建 `core/layout_analyzer.py` — 版面分析

对预处理后的图像进行版面结构分析：

```python
def analyze_layout(img: np.ndarray, ocr_results: list[dict]) -> dict:
    """
    输入：预处理后的图像 (numpy array)，OCR 原始结果
    输出：版面结构信息
    
    分析内容：
    1. 行检测（Row Detection）
       - 对 OCR 结果按 y 坐标聚类（gap < 页高/20）
       - 返回每行的 y 范围、排列序号
       
    2. 元素分类（Element Classification）
       - 根据 bbox 高度 + 置信度组合判断每个元素类型：
         - "printed_prompt"：拼音提示（h < 70px，位于行上半）
         - "handwritten_answer"：手写答案（h >= 75px，位于行下半）
       
    3. 答案区域定位（Answer Zone Localization）
       - 对每行计算拼音区与答案区的分界线（y 中位数）
       - 返回每个答案区域的 bbox 坐标
    """
```

**核心逻辑：**

```
对每行 OCR 项按 y 排序：
  上半部分（y < 行中位数）→ 拼音提示 → 标记为 "printed"
  下半部分（y >= 行中位数）→ 手写答案 → 标记为 "handwritten"

过滤规则：
  - 丢弃 "printed" 项
  - 保留 "handwritten" 项
  - 过滤掉非汉字残留（纯字母、单个符号）
```

***

### 变更 3：`core/ocr_engine.py` — 替换为 PaddleOCR

**EasyOCR（当前）→ PaddleOCR（架构要求）：**

| 对比项 | EasyOCR                              | PaddleOCR                                   |
| --- | ------------------------------------ | ------------------------------------------- |
| 模型  | 通用 OCR                               | ch\_PP-OCRv4\_mobile（轻量中文手写）                |
| 初始化 | `easyocr.Reader(["ch_sim","en"])`    | `PaddleOCR(use_angle_cls=False, lang='ch')` |
| 推理  | `reader.readtext(img_path)`          | `ocr.ocr(img_path, cls=False)`              |
| 返回值 | `[([[x0,y0],...], text, conf), ...]` | `[ [ [[x0,y0],...], (text, conf) ], ... ]`  |

变更点：

* singlton 从 `easyocr.Reader` 改为 `PaddleOCR`

* 返回值解析适配 PaddleOCR 层级（外层多一层 list）

* 模型名从 `config.ini` 读取（`[ocr] model`）

* 新增 `ocr_image(img: np.ndarray)` 函数，支持直接对 numpy array 推理（避免临时文件读写）

**PaddleOCR 返回值结构：**

```python
# PaddleOCR 返回:
result = ocr.ocr(img_path, cls=False)
# result = [ [ [[x0,y0],[x1,y1],[x2,y2],[x3,y3]], (text, confidence) ], ... ]
# 注意：最外层还有一个 list 包装
```

***

### 变更 4：`monitor/processor.py` — 整合四步流水线

```python
def process_pdf(pdf_path: str) -> None:
    # ... 解析文件名、加载答案（不变） ...

    # Step 1: 渲染 PDF 为图片
    doc = fitz.open(pdf_path)
    for page_idx in range(len(doc)):
        pix = doc[page_idx].get_pixmap(matrix=fitz.Matrix(dpi/72, dpi/72))
        img = Image.open(io.BytesIO(pix.tobytes("png")))

        # Step 2: 图像预处理（新增）
        from core.image_processor import preprocess_image
        processed_img = preprocess_image(img)

        # Step 3: OCR（用 PaddleOCR）
        from core.ocr_engine import ocr_image  # 新增：直接对 numpy array OCR
        ocr_results = ocr_image(np.array(processed_img), page_idx)

        # Step 4: 版面分析 + 答案提取（新增）
        from core.layout_analyzer import extract_student_answers
        student_answers = extract_student_answers(ocr_results, img.width, img.height)

    # Step 5: 比对 + 批注 + 打印（不变）
    graded = grade(student_answers, answers)
    # ...
```

***

## 四、依赖变更

`requirements.txt`：

```
- easyocr>=1.7       # 删除
+ paddleocr>=2.8     # 新增（含 paddlepaddle）
+ opencv-python>=4.8 # 新增（图像预处理）
```

> 注：`opencv-python-headless` 也可能已随 easyocr 安装，切换后确保保留。

***

## 五、验证步骤

### 5.1 安装依赖

```bash
pip install paddleocr opencv-python
```

### 5.2 PaddleOCR 模型下载

```bash
# 首次初始化自动下载 ch_PP-OCRv4_mobile 模型
python -c "from paddleocr import PaddleOCR; PaddleOCR(use_angle_cls=False, lang='ch', use_gpu=False)"
```

### 5.3 端到端测试

```bash
cd d:\MyCode\correction\auto_marker
C:\Users\yang\AppData\Local\Programs\Python\Python312\python.exe run_test.py --pdf 301_2025-03-20_001.pdf
```

### 5.4 验证指标

* 预处理后图像对比度提升、倾斜校正

* `extract_student_answers` 输出数量 ≈ 32（标准答案字数）

* 批改结果应有部分正确/存疑（非全错）

* Web 界面能看到准确比对数据

***

## 六、边界情况

| 场景      | 处理                 |
| ------- | ------------------ |
| 图片过暗/过亮 | CLAHE 自适应增强        |
| 扫描倾斜    | 自动旋转校正             |
| 图片模糊    | 拉普拉斯方差检测，标注"模糊"    |
| 学生漏写答案  | 该行无 handwrite 项，跳过 |
| 多页 PDF  | 逐页预处理 + 分析         |
| 非标准题型   | 基于空间聚类的分行逻辑自动适配    |

***

## 七、不包含范围

* ❌ 不使用 YOLO 等目标检测模型做区域检测（场景简单，规则足够）

* ❌ 不使用 LLM 做语义理解（字词级别比对用规则即可）

* ❌ 不修改数据库模型 / CRUD / Web 界面

