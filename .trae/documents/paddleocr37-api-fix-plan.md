# PP-OCRv6 三阶段流水线 — PaddleOCR 3.7 API 适配与验证计划

## 1. 摘要

修复 `handwriting_recognizer.py` 以适配 PaddleOCR 3.7 API（`ocr()` → `predict()`），确保三阶段流水线（检测→分离→提取）端到端可运行。`text_detector.py` 和 `layout_analyzer.py` 已在之前会话中完成适配。

---

## 2. 当前状态分析

### 已完成的适配
| 文件 | 状态 | 说明 |
|------|------|------|
| `core/text_detector.py` | ✅ 完成 | 使用 `PaddleOCR(lang='ch')` + `predict()` + `dt_polys` |
| `core/layout_analyzer.py` | ✅ 完成 | 使用 `PPStructureV3` + `predict()` + `layout_det_res.boxes` |
| `monitor/processor.py` | ✅ 无需修改 | 已导入三阶段模块，流水线顺序正确 |

### 待修复的问题
`core/handwriting_recognizer.py` 仍使用 PaddleOCR **旧版 API**，有两个不兼容点：

1. **构造参数不兼容**（第 27-33 行）：
   ```python
   # 旧版 API（PaddleOCR < 3.7）
   _rec_instance = PaddleOCR(det=False, rec=True, use_angle_cls=False, use_gpu=False, lang='ch')
   ```
   PaddleOCR 3.7 构造器不再支持 `det`/`rec`/`use_gpu` 参数，应改为 `PaddleOCR(lang='ch')`。

2. **推理 API 不兼容**（第 116 行）：
   ```python
   # 旧版 API
   rec_result = recognizer.ocr(cropped, det=False, rec=True)
   ```
   PaddleOCR 3.7 废弃了 `ocr()` 方法，统一使用 `predict()`。

3. **结果解析不兼容**（第 124-129 行）：
   ```python
   # 旧版返回格式: [[('春眠不觉晓', 0.95)]]
   rec_data = rec_result[0]
   recognized_text = rec_data[0][0]
   line_confidence = float(rec_data[0][1])
   ```
   `predict()` 返回 `OCRResult` 对象，需通过 `rec_texts` 和 `rec_scores` 属性访问。

---

## 3. 修改计划

### 3.1 修复 `core/handwriting_recognizer.py`

**变更 1 — 初始化方式**（第 27-33 行）：
```
旧: PaddleOCR(det=False, rec=True, use_angle_cls=False, use_gpu=False, lang='ch')
新: PaddleOCR(lang='ch')
```
- 与 `text_detector.py` 保持一致，统一初始化方式
- `predict()` 会自动返回检测和识别结果，我们只使用识别部分

**变更 2 — 推理调用**（第 116 行）：
```
旧: recognizer.ocr(cropped, det=False, rec=True)
新: recognizer.predict(cropped)
```

**变更 3 — 结果解析**（第 118-133 行）：
```
旧:
    rec_result = recognizer.ocr(cropped, det=False, rec=True)
    if not rec_result or not rec_result[0]:
        continue
    rec_data = rec_result[0]
    recognized_text = rec_data[0][0]
    line_confidence = float(rec_data[0][1])

新:
    raw_results = recognizer.predict(cropped)
    recognized_text = None
    line_confidence = 0.0
    for page_result in raw_results:
        rec_texts = getattr(page_result, 'rec_texts', None)
        rec_scores = getattr(page_result, 'rec_scores', None)
        if rec_texts and len(rec_texts) > 0:
            recognized_text = rec_texts[0]
            line_confidence = float(rec_scores[0]) if rec_scores else 0.0
            break
    if not recognized_text:
        continue
```

### 3.2 端到端测试验证

1. **环境准备**：确认依赖已安装（`paddlepaddle>=3.3.0`, `paddleocr>=3.7.0`, `paddlex[ocr]`）
2. **运行测试**：
   ```bash
   cd auto_marker
   python run_test.py --seed-answers "春眠不觉晓处处闻啼鸟"
   ```
3. **验证内容**：
   - 流水线是否完整执行（无 API 错误）
   - 手写识别结果是否正确提取
   - 比对结果是否正确
   - 批注 PDF 是否生成
4. **性能基线**（记录首次运行数据）：
   - 处理时间
   - 检测到的文本区域数
   - 分离出的手写区域数
   - 识别字符数
   - 平均置信度

---

## 4. 不修改的部分

| 文件 | 原因 |
|------|------|
| `core/text_detector.py` | 已适配 PaddleOCR 3.7 |
| `core/layout_analyzer.py` | 已适配 PPStructureV3 |
| `core/grader.py` | 接口不依赖 PaddleOCR，无需修改 |
| `core/pdf_annotator.py` | 接口不依赖 PaddleOCR，无需修改 |
| `core/image_processor.py` | 独立模块，不依赖 PaddleOCR |
| `monitor/processor.py` | 已使用正确导入 |
| `db/*` / `web/*` | 不涉及本次变更 |

---

## 5. 假设与风险

- **假设**：PaddleOCR 3.7 的 `predict()` 返回的 `OCRResult` 对象包含 `rec_texts` 和 `rec_scores` 属性，且排序与检测框对应
- **风险**：如果 `predict()` 返回的识别结果包含非中文字符（空格、标点），等宽切分逻辑可能偏移。当前 `_is_chinese_char()` 过滤已处理此情况
- **风险**：`predict()` 同时跑检测和识别，对纯手写区域可能检测到额外的小框。当前代码只取第一个识别结果，应能正常工作

---

## 6. 验证步骤

1. 修改 `handwriting_recognizer.py`
2. 执行 `python run_test.py --seed-answers "春眠不觉晓处处闻啼鸟"`
3. 检查日志输出，确认无错误
4. 检查 `data/output/` 目录是否有批注 PDF 生成
5. 运行 `python run_test.py --show` 查看结果汇总