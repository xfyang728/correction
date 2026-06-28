# 默写批改台 — 项目规则

## 硬约束（AI 生成代码时必须遵守）

1. **OCR 缓存**：只缓存 `(det_boxes, ocr_records)`，不缓存版面分析和手写识别结果
2. **Session 配置**：`sessionmaker` 必须设 `expire_on_commit=False`，确保 Task 对象在 session 关闭后可访问
3. **per_char 标记**：逐字题型必须逐字标记，即使全对也不用单个大勾
4. **Qwen3-VL 输入**：必须使用原始 PDF 图像，不做预处理（deskew 可选但禁用 CLAHE）
5. **答案解析**：必须顺序处理结果列表，不用字典（防止答案覆盖）
6. **标记坐标**：勾用 bbox 右下角，圈/三角用中心坐标

## 代码规范

- 服务层类（PipelineService, AnswerService, TaskService）处理核心业务逻辑；processor.py 是薄包装
- 题型在 layout_analyzer.py 确定，经 grader.py 传播到 pdf_annotator.py
- 文字渲染颜色：正确=绿，错误=红，存疑=橙
- 文字透明度 0.7，字号自适应 bbox 高度（8-24pt）
- 中文字体自动注册（微软雅黑/黑体/宋体）
- 函数添加 docstring，说明参数和返回值

## 坐标映射规则

- 模型归一化坐标 [0-1] → 像素坐标：`x_pixel = x_norm * img_w`，`y_pixel = y_norm * img_h`
- 像素坐标 → PDF 页面坐标：`x_page = x_pixel / img_w * page_width`，`y_page = page_height - y_pixel / img_h * page_height`
- **Y 坐标必须 clamp**：Qwen3-VL 的 y 坐标系统性偏移，必须用 `_clamp_chars_y_to_region()` 替换为手写框 y 范围
- **X 坐标修正策略**：文字渲染模式下，当模型 x 跨度偏大（>1.3×目标范围）或整体偏移（>80px）时，用 `_rescale_chars_x_to_region()` 线性映射到 marker_text+answer_text 推导的目标范围；偏差小时保留模型 x（保留逐字相对间距）
- 圈号 q_idx（200+）不存在时，通过标记文本中的圈号字符回退匹配
- 无题号格式的 q_marker 通过标记文本内容回退匹配

## 测试要求

- 修改 `core/qwen_vl_recognizer.py` 坐标映射逻辑后，必须运行 `tests/test_qwen_vl_recognizer.py`
- 修改 `core/grader.py` 后，必须运行 `tests/test_grader.py`
- 修改 `core/layout_analyzer.py` 后，必须运行 `tests/test_layout_analyzer.py`
- 代码变更影响坐标定位时，必须清除 `data/cache/` 下的 OCR 缓存

## 已知陷阱

- Qwen3-VL y 坐标：多题共享同一 y 值（如题1/5/7 都是 y=0.45-0.52），必须 y-clamp 修正
- Qwen3-VL 圈号误读：①② 可能被读为 (1)(2)，导致 q_idx 不匹配
- Qwen3-VL 答案合并：拆分题（如 (7)(8)）的答案可能被合并，需后处理拆分
- Qwen3-VL 过度识别：手写区域含印刷体时会过度识别，需精确裁剪
- 旧缓存导致偏移：代码变更影响定位后，旧 OCR 缓存会导致持续坐标偏移
