"""识别引擎配置 — 控制使用 PaddleOCR / Qwen3-VL / 双路并行。"""

# 识别引擎选择: "paddle" | "qwen_vl" | "dual" | "page_level"
RECOGNITION_ENGINE = "page_level"

# ---- Qwen3-VL 配置 ----
QWEN_VL_API_URL = "http://localhost:8080/v1/chat/completions"
QWEN_VL_MODEL = "Qwen3VL-4B-Instruct"
QWEN_VL_TIMEOUT = 30  # 秒
QWEN_VL_MAX_WORKERS = 4  # 并发数（匹配 llama.cpp n_parallel）

# 识别 prompt — 强调只识别手写字符，保守输出
QWEN_VL_PROMPT = (
    "图片中是手写的中文字符。请只识别手写部分，忽略任何印刷体文字。"
    "只返回识别出的汉字，不要标点、数字、拼音或解释。"
    "如果不确定某个字，宁可跳过也不要猜。"
)

# 整页识别 prompt — 只识别，不批改
QWEN_VL_PAGE_LEVEL_PROMPT = (
    "识别每道题的手写答案，按题号输出。\n"
    "格式：(题号) 答案\n"
    "只写汉字，不要标点、解释、判断。\n"
    "没写就写（未作答）"
)

# ---- 双路合并策略 ----
# "paddle_priority": PaddleOCR 为主，Qwen3-VL 仅在低置信度时覆盖
# "qwen_priority": Qwen3-VL 文本为主，PaddleOCR 提供 bbox
# "vote": 两路结果一致时提高置信度，不一致时取高置信度方
DUAL_MERGE_STRATEGY = "paddle_priority"

# PaddleOCR 低置信度阈值（paddle_priority 策略下，低于此值才用 Qwen3-VL 覆盖）
PADDLE_LOW_CONF_THRESHOLD = 0.60
