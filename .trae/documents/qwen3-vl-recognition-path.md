# 重构识别：添加 Qwen3-VL 双路并行识别路径

## 目标
在现有 PaddleOCR 识别路径基础上，添加 Qwen3-VL 作为第二路识别引擎，双路并行运行，合并结果提高准确率。

## 用户环境
- **模型**: Qwen3VL-4B-Instruct-Q4_K_M.gguf
- **服务**: llama.cpp server (`llama-server.exe`)
- **地址**: `http://localhost:8080`
- **API**: OpenAI 兼容 (`/v1/chat/completions`)
- **GPU**: RTX 3060 Laptop (6GB VRAM)

## 当前架构分析

### 现有流水线 (`_run_ocr_pipeline`)
```
PDF → 渲染(200DPI) → 预处理 → PaddleOCR检测+识别 → 版面分析 → 手写识别 → 逐字切分 → grader
```

### 关键接口
- `detect_text(img, page_idx)` → `(det_boxes, ocr_records)` — PaddleOCR 检测+识别
- `recognize_handwriting(img, hw_boxes, ocr_records, page_idx)` → `[{char, bbox_pixel, confidence, question_idx, ...}]`
- grader 期望的格式：每个字符一个 dict，含 `bbox_pixel`, `char`, `confidence`, `question_idx`

### Qwen3-VL 的定位
- **不替代** PaddleOCR 的文本检测和版面分析（这些 PaddleOCR 做得很好）
- **替代/补充** `recognize_handwriting` — 对每个手写区域用 VL 模型重新识别
- 双路并行：PaddleOCR 和 Qwen3-VL 同时识别，结果合并

## 实施方案

### 1. 新建 `core/qwen_vl_recognizer.py`

Qwen3-VL 识别器，通过 llama.cpp OpenAI 兼容 API 调用。

**核心函数**:
```python
def recognize_with_qwen_vl(
    img: np.ndarray,
    handwriting_boxes: list[dict],
    page_idx: int = 0,
) -> list[dict]:
    """用 Qwen3-VL 识别手写区域，返回与 PaddleOCR 兼容的逐字结果。"""
```

**实现细节**:
- 对每个 `handwriting_box`，裁剪对应区域
- 转为 base64，通过 `http://localhost:8080/v1/chat/completions` 发送
- Prompt: `"请识别图片中的手写中文字符。只返回识别出的文字内容，不要添加任何解释、标点或其他字符。"`
- 从响应中提取中文字符（过滤非中文字符）
- 用等宽切分生成逐字 bbox（与现有 `handwriting_recognizer` 的 Step 2 逻辑一致）
- 返回格式与 `recognize_handwriting` 完全兼容

**API 调用细节**:
```python
import base64, io, requests
from PIL import Image

def _call_qwen_vl(cropped_img: np.ndarray) -> str:
    """调用 llama.cpp server 的 Qwen3-VL 识别手写文字。"""
    # 转 base64
    pil_img = Image.fromarray(cropped_img)
    buffer = io.BytesIO()
    pil_img.save(buffer, format="PNG")
    b64 = base64.b64encode(buffer.getvalue()).decode()

    resp = requests.post(
        "http://localhost:8080/v1/chat/completions",
        json={
            "model": "Qwen3VL-4B-Instruct",
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
                    {"type": "text", "text": "请识别图片中的手写中文字符。只返回识别出的文字内容，不要添加任何解释、标点或其他字符。"}
                ]
            }],
            "max_tokens": 128,
            "temperature": 0.1,
        },
        timeout=30,
    )
    return resp.json()["choices"][0]["message"]["content"]
```

**并发优化**: 使用 `concurrent.futures.ThreadPoolExecutor` 并发处理多个手写区域（llama.cpp server 支持 `n_parallel=4`）。

### 2. 新建 `core/recognition_config.py`

识别引擎配置：
```python
# 识别引擎选择: "paddle" | "qwen_vl" | "dual"
RECOGNITION_ENGINE = "dual"

# Qwen3-VL 配置
QWEN_VL_API_URL = "http://localhost:8080/v1/chat/completions"
QWEN_VL_MODEL = "Qwen3VL-4B-Instruct"
QWEN_VL_TIMEOUT = 30  # 秒
QWEN_VL_MAX_WORKERS = 4  # 并发数（匹配 llama.cpp n_parallel）

# 双路合并策略
# "paddle_priority": PaddleOCR 为主，Qwen3-VL 仅在低置信度时覆盖
# "qwen_priority": Qwen3-VL 为主，PaddleOCR 提供 bbox
# "vote": 两路结果一致时提高置信度，不一致时取高置信度方
DUAL_MERGE_STRATEGY = "vote"
```

### 3. 新建 `core/recognition_merger.py`

双路结果合并模块：

```python
def merge_recognition_results(
    paddle_results: list[dict],
    qwen_results: list[dict],
    strategy: str = "vote",
) -> list[dict]:
    """合并 PaddleOCR 和 Qwen3-VL 的识别结果。"""
```

**合并逻辑**（按 handwriting_box 分组后逐组合并）:

1. **按 `question_idx` + `bbox_pixel`  proximity 配对**两路的识别结果
2. **vote 策略**:
   - 两路识别出相同字符 → 置信度取 `max(paddle_conf, qwen_conf) * 1.1`（上限 1.0），标记为高可信
   - 字符不同 → 取置信度高的一方；若都低于阈值，标记为 uncertain
   - 某路缺失 → 用另一路结果，置信度不打折
3. **paddle_priority 策略**: PaddleOCR 结果为主，Qwen3-VL 仅在 PaddleOCR 置信度 < 0.6 时覆盖
4. **qwen_priority 策略**: Qwen3-VL 文本为主，用 PaddleOCR 的 bbox 做等宽切分

### 4. 修改 `services/pipeline_service.py`

在 `_run_ocr_pipeline` 的 Step D 处改为双路识别：

```python
# Step D: 双路手写识别
from core.recognition_config import RECOGNITION_ENGINE

if RECOGNITION_ENGINE == "paddle":
    # 原有逻辑
    page_results = recognize_handwriting(processed_np, hw_boxes, ocr_records, page_idx)
elif RECOGNITION_ENGINE == "qwen_vl":
    from core.qwen_vl_recognizer import recognize_with_qwen_vl
    page_results = recognize_with_qwen_vl(processed_np, hw_boxes, page_idx)
elif RECOGNITION_ENGINE == "dual":
    from concurrent.futures import ThreadPoolExecutor
    from core.qwen_vl_recognizer import recognize_with_qwen_vl
    from core.recognition_merger import merge_recognition_results

    with ThreadPoolExecutor(max_workers=2) as pool:
        paddle_future = pool.submit(
            recognize_handwriting, processed_np, hw_boxes, ocr_records, page_idx
        )
        qwen_future = pool.submit(
            recognize_with_qwen_vl, processed_np, hw_boxes, page_idx
        )
        paddle_results = paddle_future.result()
        qwen_results = qwen_future.result()

    page_results = merge_recognition_results(paddle_results, qwen_results)
```

### 5. 修改 `requirements.txt`

添加 `requests` 依赖（如果尚未包含）。

## 涉及文件

| 文件 | 操作 | 说明 |
|------|------|------|
| `core/qwen_vl_recognizer.py` | **新建** | Qwen3-VL 识别器，调用 llama.cpp API |
| `core/recognition_config.py` | **新建** | 识别引擎配置 |
| `core/recognition_merger.py` | **新建** | 双路结果合并 |
| `services/pipeline_service.py` | **修改** | Step D 改为双路识别 |
| `requirements.txt` | **修改** | 添加 requests |

## 假设与决策

1. **Qwen3-VL 不提供字符级 bbox** → 使用等宽切分（与现有 handwriting_recognizer 一致）
2. **llama.cpp server 已运行在 8080 端口** → 配置中硬编码默认地址，可通过配置文件修改
3. **并发数 = 4** → 匹配 llama.cpp 的 `n_parallel=4` 设置
4. **合并策略默认 "vote"** → 两路一致时提高置信度，不一致时取高置信度方
5. **Qwen3-VL 失败时 fallback 到 PaddleOCR** → 不影响原有流程的稳定性

## 验证步骤

1. 确保 llama.cpp server 运行在 `http://localhost:8080`
2. 设置 `RECOGNITION_ENGINE = "qwen_vl"` 单独测试 Qwen3-VL 路径
3. 对 `301_2026-06-18_003.pdf` 运行，对比两路识别结果
4. 设置 `RECOGNITION_ENGINE = "dual"` 测试合并效果
5. 检查合并后的置信度分布和 grading 准确率
