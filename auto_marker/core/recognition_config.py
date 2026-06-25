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

# 整页识别 prompt 模板 — 只识别 + 输出逐字归一化 bbox（批改由 grader 完成）
# {answers_text} 由 recognizer 动态填充为原始多行标准答案
QWEN_VL_PAGE_LEVEL_PROMPT_TEMPLATE = (
    "你是手写文字识别助手。图片是一张学生作业扫描页。请完成：\n"
    "1. 找到每道题的学生手写答案（忽略印刷题目、拼音提示、涂涂画画）。\n"
    "2. 逐个识别学生\"实际书写\"的汉字——学生可能写错，必须如实转录，不得纠正或照抄标准答案。\n"
    "3. 对每个手写汉字，给出它在整页图像中的归一化包围盒 [x0,y0,x1,y1]，"
    "坐标范围 0~1，左上角(0,0)右下角(1,1)。\n\n"
    "只输出 JSON 数组，不要任何解释、评分或多余文字。格式：\n"
    "[\n"
    "  {{\"q\":\"（1）\",\"chars\":[{{\"c\":\"随\",\"bbox\":[0.12,0.45,0.16,0.52]}},{{\"c\":\"君\",\"bbox\":[0.16,0.45,0.20,0.52]}}]}},\n"
    "  {{\"q\":\"①\",\"chars\":[{{\"c\":\"质\",\"bbox\":[0.31,0.60,0.34,0.67]}}]}}\n"
    "]\n"
    "其中 \"q\" 是题号原文，\"c\" 是识别汉字，\"bbox\" 是归一化坐标。未作答则 \"chars\" 为空数组。坐标必须是 0~1 的小数。\n\n"
    "参考标准答案（仅供识别辅助，务必转录学生实际书写）：\n"
    "{answers_text}"
)

# 标准答案为空时的回退默认值
QWEN_VL_PAGE_LEVEL_DEFAULT_ANSWERS = (
    "（1）随君直到夜郎西\n"
    "（2）海内存知己\n"
    "（3）百般红紫斗芳菲\n"
    "（4）水中藻荇交横\n"
    "（5）人生自古谁无死\n"
    "（6）半竿斜日旧关城\n"
    "（7）采菊东篱下\n"
    "（8）悠然见南山\n"
    "① 质朴；② 绚丽"
)

# 整页识别的 token 上限（逐字坐标输出较长）
QWEN_VL_PAGE_LEVEL_MAX_TOKENS = 4096
# 整页识别超时（4B 模型生成 2000+ token 需要更久，单独控制不影响单框识别的 30s）
QWEN_VL_PAGE_LEVEL_TIMEOUT = 120

# ---- 双路合并策略 ----
# "paddle_priority": PaddleOCR 为主，Qwen3-VL 仅在低置信度时覆盖
# "qwen_priority": Qwen3-VL 文本为主，PaddleOCR 提供 bbox
# "vote": 两路结果一致时提高置信度，不一致时取高置信度方
DUAL_MERGE_STRATEGY = "paddle_priority"

# PaddleOCR 低置信度阈值（paddle_priority 策略下，低于此值才用 Qwen3-VL 覆盖）
PADDLE_LOW_CONF_THRESHOLD = 0.60
