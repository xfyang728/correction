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

# 整页识别 prompt — 识别 + 批改
QWEN_VL_PAGE_LEVEL_PROMPT = (
    "请根据以下学生作答内容，对照标准答案逐题进行批改，并输出：\n\n"
    "✅ 每题得分（满分分值）\n"
    "❗ 扣分原因说明（准确指出错误点）\n"
    '📌 建议评分（如"全对""错1字扣1分""语义通顺但结构不工整"等）\n'
    "📊 总分（满分 × 题数）\n"
    '💡 附带教学建议（如"加强字词辨析""对联结构需对仗工整"等）\n\n'
    "特别说明：\n"
    "- 填涂答题卡部分（如选择题涂黑处）视为标准答案，无需判断。\n"
    "- 手写答案部分（如默写、填空、作文）以内容为准，错误需指出。\n"
    "- 诗句默写、标点、对联等需结合语境与规范判断。\n\n"
    "📄 试卷内容（请按题号顺序提供）：\n"
    "（请从图像中识别题目结构）\n\n"
    "⭐ 标准答案：\n"
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

# ---- 双路合并策略 ----
# "paddle_priority": PaddleOCR 为主，Qwen3-VL 仅在低置信度时覆盖
# "qwen_priority": Qwen3-VL 文本为主，PaddleOCR 提供 bbox
# "vote": 两路结果一致时提高置信度，不一致时取高置信度方
DUAL_MERGE_STRATEGY = "paddle_priority"

# PaddleOCR 低置信度阈值（paddle_priority 策略下，低于此值才用 Qwen3-VL 覆盖）
PADDLE_LOW_CONF_THRESHOLD = 0.60
