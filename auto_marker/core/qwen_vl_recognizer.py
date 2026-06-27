"""
Qwen3-VL 手写识别模块 — 通过 llama.cpp server 调用 Qwen3-VL 识别手写文字。

作为 PaddleOCR 的并行/替代识别路径，对每个手写区域裁剪后发送 VL 模型识别，
返回与 handwriting_recognizer 兼容的逐字结果格式。

依赖：llama.cpp server 运行在 http://localhost:8080（OpenAI 兼容 API）。
"""

import base64
import io
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
from PIL import Image

from core.recognition_config import (
    QWEN_VL_API_URL,
    QWEN_VL_MAX_WORKERS,
    QWEN_VL_MODEL,
    QWEN_VL_PROMPT,
    QWEN_VL_TIMEOUT,
)
from core.utils import is_chinese_char

logger = logging.getLogger("qwen_vl_recognizer")


def _img_to_base64(img_array: np.ndarray) -> str:
    """将 numpy 图像数组转为 base64 PNG 字符串。"""
    pil_img = Image.fromarray(img_array)
    buffer = io.BytesIO()
    pil_img.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode()


def _call_qwen_vl(cropped_img: np.ndarray) -> str:
    """调用 llama.cpp server 的 Qwen3-VL 识别手写文字。

    返回识别出的纯文本（可能包含非中文字符，需后续过滤）。
    调用失败返回空字符串。
    """
    import requests

    b64 = _img_to_base64(cropped_img)

    try:
        resp = requests.post(
            QWEN_VL_API_URL,
            json={
                "model": QWEN_VL_MODEL,
                "messages": [{
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{b64}"},
                        },
                        {"type": "text", "text": QWEN_VL_PROMPT},
                    ],
                }],
                "max_tokens": 128,
                "temperature": 0.1,
            },
            timeout=QWEN_VL_TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"]
    except Exception as e:
        logger.warning("Qwen3-VL API 调用失败: %s", e)
        return ""


def _recognize_single_box(
    img_array: np.ndarray,
    hw_box: dict,
    page_idx: int,
) -> list[dict]:
    """对单个手写区域用 Qwen3-VL 识别，返回逐字结果。"""
    bbox = hw_box["bbox_pixel"]
    question_idx = hw_box.get("question_idx")

    x0, y0, x2, y2 = bbox
    h, w = img_array.shape[:2]
    x0 = max(0, int(x0))
    y0 = max(0, int(y0))
    x2 = min(w, int(x2))
    y2 = min(h, int(y2))
    if x2 <= x0 or y2 <= y0:
        return []

    cropped = img_array[y0:y2, x0:x2]
    text = _call_qwen_vl(cropped)

    # 提取中文字符
    chars = [ch for ch in text if is_chinese_char(ch)]
    if not chars:
        logger.debug("Qwen3-VL 识别无中文字符: '%s' (bbox=%s)", text, bbox)
        return []

    # 等宽切分（与 handwriting_recognizer 的 Step 2 逻辑一致）
    img_h, img_w = img_array.shape[:2]
    bx0, by0, bx2, by2 = bbox
    region_w = bx2 - bx0
    char_w = region_w / len(chars)
    margin = char_w * 0.05

    results = []
    for i, ch in enumerate(chars):
        cx0 = int(bx0 + i * char_w + margin)
        cx2 = int(bx0 + (i + 1) * char_w - margin)
        if cx2 <= cx0:
            cx0 = int(bx0 + i * char_w)
            cx2 = int(bx0 + (i + 1) * char_w)

        results.append({
            "page": page_idx,
            "bbox_pixel": (cx0, by0, cx2, by2),
            "char": ch,
            "confidence": 0.85,  # VL 模型无逐字置信度，给默认值
            "question_idx": question_idx,
            "img_pixel_w": img_w,
            "img_pixel_h": img_h,
            "engine": "qwen_vl",
        })

    return results


def recognize_with_qwen_vl(
    img,
    handwriting_boxes: list[dict],
    page_idx: int = 0,
) -> list[dict]:
    """用 Qwen3-VL 识别所有手写区域，返回与 PaddleOCR 兼容的逐字结果。

    参数:
        img: RGB numpy array (H, W, 3)
        handwriting_boxes: layout_analyzer 输出的手写区域列表
        page_idx: 页码

    返回:
        与 recognize_handwriting() 相同格式的逐字结果列表，
        每项额外含 "engine": "qwen_vl" 标记来源。
    """
    if not isinstance(img, np.ndarray):
        img = np.array(Image.open(img) if isinstance(img, str) else img)

    if not handwriting_boxes:
        return []

    t_start = time.time()
    all_results: list[dict] = []

    # 并发调用 Qwen3-VL（llama.cpp n_parallel=4）
    with ThreadPoolExecutor(max_workers=QWEN_VL_MAX_WORKERS) as pool:
        futures = {
            pool.submit(_recognize_single_box, img, box, page_idx): box
            for box in handwriting_boxes
        }
        for future in as_completed(futures):
            try:
                results = future.result()
                all_results.extend(results)
            except Exception as e:
                box = futures[future]
                logger.warning("Qwen3-VL 识别失败 (bbox=%s): %s", box.get("bbox_pixel"), e)

    elapsed = time.time() - t_start
    logger.info(
        "第 %d 页: Qwen3-VL 识别 %d 个手写区域 → %d 个字符 (%.1fs)",
        page_idx + 1, len(handwriting_boxes), len(all_results), elapsed,
    )
    return all_results


def _build_page_level_prompt(answers_text: str) -> str:
    """用模板 + answers_text 拼装整页识别 prompt。"""
    from core.recognition_config import (
        QWEN_VL_PAGE_LEVEL_DEFAULT_ANSWERS,
        QWEN_VL_PAGE_LEVEL_PROMPT_TEMPLATE,
    )

    if not answers_text or not answers_text.strip():
        answers_text = QWEN_VL_PAGE_LEVEL_DEFAULT_ANSWERS
    return QWEN_VL_PAGE_LEVEL_PROMPT_TEMPLATE.format(answers_text=answers_text)


def _call_qwen_vl_page_level(img_array: np.ndarray, answers_text: str = "") -> str:
    """调用 Qwen3-VL 整页识别，返回模型原始响应文本。"""
    import requests

    from core.recognition_config import (
        QWEN_VL_API_URL,
        QWEN_VL_MODEL,
        QWEN_VL_PAGE_LEVEL_MAX_TOKENS,
        QWEN_VL_PAGE_LEVEL_TIMEOUT,
    )

    b64 = _img_to_base64(img_array)
    prompt = _build_page_level_prompt(answers_text)

    try:
        resp = requests.post(
            QWEN_VL_API_URL,
            json={
                "model": QWEN_VL_MODEL,
                "messages": [{
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{b64}"},
                        },
                        {"type": "text", "text": prompt},
                    ],
                }],
                "max_tokens": QWEN_VL_PAGE_LEVEL_MAX_TOKENS,
                "temperature": 0.1,
                "repeat_penalty": 1.1,
                "repeat_last_n": 64,
                "top_k": 40,
                "top_p": 0.95,
                "min_p": 0.05,
                "frequency_penalty": 0.0,
                "presence_penalty": 0.0,
            },
            timeout=QWEN_VL_PAGE_LEVEL_TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"]
    except Exception as e:
        logger.warning("Qwen3-VL 整页识别 API 调用失败: %s", e)
        return ""


def _parse_page_level_json(response: str) -> list[dict] | None:
    """解析 JSON 格式的整页响应，返回按题分组的逐字坐标列表。

    返回: [{"q_marker": str, "chars": [{"char": str, "bbox_norm": (x0,y0,x1,y1)}]}, ...]
          解析失败返回 None（触发旧文本解析器回退）。
    """
    import json

    text = response.strip()

    # 去除 markdown 围栏
    if text.startswith("```"):
        lines = text.split("\n")
        # 去首行围栏和可能的末行围栏
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()

    # 截取首个 [ 到末个 ] 的子串
    start = text.find("[")
    end = text.rfind("]")
    if start == -1 or end == -1 or end <= start:
        return None

    json_str = text[start:end + 1]

    try:
        data = json.loads(json_str)
    except json.JSONDecodeError:
        return None

    if not isinstance(data, list):
        return None

    results: list[dict] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        q_marker = item.get("q", "")
        if not isinstance(q_marker, str):
            q_marker = str(q_marker) if q_marker is not None else ""

        chars_raw = item.get("chars", [])
        if not isinstance(chars_raw, list):
            continue

        parsed_chars: list[dict] = []
        all_valid = True
        for ch_item in chars_raw:
            if not isinstance(ch_item, dict):
                all_valid = False
                break
            ch = ch_item.get("c", "")
            if not isinstance(ch, str) or not ch:
                all_valid = False
                break
            # 只保留中文字符
            cn_chars = [c for c in ch if is_chinese_char(c)]
            if not cn_chars:
                continue
            bbox = ch_item.get("bbox")
            if not isinstance(bbox, list) or len(bbox) != 4:
                all_valid = False
                break
            try:
                coords = [float(v) for v in bbox]
            except (TypeError, ValueError):
                all_valid = False
                break
            # 值域检测：若任一值 >1.5，判定模型用了 0-1000 量纲，整体除以 1000
            if any(abs(v) > 1.5 for v in coords):
                coords = [v / 1000.0 for v in coords]
            # clamp 到 [0,1]
            coords = [max(0.0, min(1.0, v)) for v in coords]
            x0, y0, x1, y1 = coords
            # 保证 x0<x1, y0<y1
            if x0 > x1:
                x0, x1 = x1, x0
            if y0 > y1:
                y0, y1 = y1, y0
            if x1 <= x0 or y1 <= y0:
                all_valid = False
                break
            # 一个 c 可能含多字（模型误拼），逐字添加同一 bbox
            for c in cn_chars:
                parsed_chars.append({"char": c, "bbox_norm": (x0, y0, x1, y1)})

        if not all_valid:
            # 标记该题无效，但保留 q_marker 以便上层决定回退
            results.append({"q_marker": q_marker, "chars": [], "invalid": True})
        else:
            results.append({"q_marker": q_marker, "chars": parsed_chars, "invalid": False})

    if not results:
        return None
    return results


def _norm_to_pixel_bbox(bbox_norm: tuple, img_w: int, img_h: int) -> tuple:
    """归一化坐标 [0-1] 转像素坐标，并保证 x0<x1, y0<y1。"""
    x0, y0, x1, y1 = bbox_norm
    px0 = int(round(x0 * img_w))
    py0 = int(round(y0 * img_h))
    px1 = int(round(x1 * img_w))
    py1 = int(round(y1 * img_h))
    if px1 <= px0:
        px1 = px0 + 1
    if py1 <= py0:
        py1 = py0 + 1
    return (px0, py0, px1, py1)


def _question_marker_to_q_idx(marker: str) -> int | None:
    """题号原文转 q_idx，复用 layout_analyzer 的 _question_match。"""
    from core.layout_analyzer import _question_match

    return _question_match(marker)


def _parse_answers_text(answers_text: str) -> dict[int, list[str]]:
    """解析原始多行标准答案文本，返回 {q_idx: [char, ...]}。

    支持格式:
        （1）随君直到夜郎西
        （2）海内存知己
        ...
        ① 质朴；② 绚丽

    用于: P0-1 invalid 题回退取答案文本、P0-2 合并题按标准答案字数拆分。
    """
    import re

    result: dict[int, list[str]] = {}
    if not answers_text:
        return result

    _CIRCLED = {'①':1,'②':2,'③':3,'④':4,'⑤':5,
                '⑥':6,'⑦':7,'⑧':8,'⑨':9,'⑩':10}
    _QIDX_OFFSET_PAREN = 0
    _QIDX_OFFSET_DOT = 100
    _QIDX_OFFSET_CIRCLE = 200

    for line in answers_text.split("\n"):
        line = line.strip()
        if not line:
            continue

        # 括号题号: （1）答案
        for m in re.finditer(r"[（(](\d+)[)）]\s*([^（(①②③④⑤⑥⑦⑧⑨⑩]*)", line):
            q_num = int(m.group(1))
            chars = [ch for ch in m.group(2) if is_chinese_char(ch)]
            if chars:
                result[q_num - 1 + _QIDX_OFFSET_PAREN] = chars

        # 带圈数字: ①答案；②答案  （一行可含多个）
        parts = re.split(r"[；;]", line)
        for part in parts:
            m = re.match(r"\s*([①②③④⑤⑥⑦⑧⑨⑩])\s*(.+)", part)
            if m:
                q_num = _CIRCLED.get(m.group(1), 0)
                chars = [ch for ch in m.group(2) if is_chinese_char(ch)]
                if chars and q_num:
                    result[q_num - 1 + _QIDX_OFFSET_CIRCLE] = chars

        # 点号题号: 1. 答案
        m = re.match(r"(\d+)\s*[.、]\s*(.+)", line)
        if m:
            q_num = int(m.group(1))
            chars = [ch for ch in m.group(2) if is_chinese_char(ch)]
            if chars:
                result[q_num - 1 + _QIDX_OFFSET_DOT] = chars

    return result


def _q_idx_format_range(q_idx: int) -> str:
    """根据 q_idx 判断题号格式: 'paren'(0-99), 'dot'(100-199), 'circle'(200-209)。"""
    if 200 <= q_idx < 210:
        return "circle"
    if 100 <= q_idx < 200:
        return "dot"
    return "paren"


def _match_q_idx_by_bbox(
    char_bbox_pixel: tuple,
    region_by_qidx: dict[int, dict],
    img_w: int,
    expected_format: str | None = None,
) -> int | None:
    """用首字像素 bbox 中心 y 匹配 region 的 [y_start,y_end]，x 中心与 marker 列同侧过滤。

    expected_format: 若提供 ('paren'/'dot'/'circle')，只匹配同格式范围的 region，
                     防止 ① 的坐标误匹配到 (3) 等跨格式 region（P0-3）。
    """
    cx = (char_bbox_pixel[0] + char_bbox_pixel[2]) / 2
    cy = (char_bbox_pixel[1] + char_bbox_pixel[3]) / 2
    page_mid = img_w / 2
    for q_idx, region in region_by_qidx.items():
        # P0-3: 格式过滤，防止带圈数字误匹配括号题号 region
        if expected_format and _q_idx_format_range(q_idx) != expected_format:
            continue
        if not (region["y_start"] <= cy < region["y_end"]):
            continue
        marker_bbox = region.get("marker_bbox")
        if marker_bbox:
            marker_cx = (marker_bbox[0] + marker_bbox[2]) / 2
            if (marker_cx < page_mid) != (cx < page_mid):
                continue
        return q_idx
    return None


def _clamp_chars_to_column(
    char_items: list[dict],
    region: dict,
    img_w: int,
    handwriting_boxes: list[dict] | None = None,
) -> list[dict]:
    """P2-6: 当模型 x 坐标落在错误列时，用等宽 x 切分替换，保留模型 y 坐标。

    模型可能把双栏当单栏输出坐标（如左栏答案 x=0.64 落在右栏区域）。
    检测：若某 char 的 x 中心与 marker 不同侧，说明模型坐标列归属错误。
    修复：对该题所有 char，用 _estimate_question_x_range 获取列 x 范围，
    按字数等宽切分 x（保留模型 y 坐标，因 y 通常更准）。
    """
    marker_bbox = region.get("marker_bbox")
    if not marker_bbox or not char_items:
        return char_items

    marker_cx = (marker_bbox[0] + marker_bbox[2]) / 2
    page_mid = img_w / 2
    is_left = marker_cx < page_mid

    # 检测是否有 char 的 x 中心落在错误列
    has_wrong_column = False
    for ci in char_items:
        x0, y0, x1, y1 = ci["bbox_norm"]
        char_cx_pixel = ((x0 + x1) / 2) * img_w
        if (char_cx_pixel < page_mid) != is_left:
            has_wrong_column = True
            break

    if not has_wrong_column:
        return char_items

    # 列归属错误 → 用等宽 x 切分替换，保留模型 y
    x_start, x_end = _estimate_question_x_range(region, handwriting_boxes, img_w)
    if x_start >= x_end:
        return char_items

    n = len(char_items)
    col_w = x_end - x_start
    char_w = col_w / n
    margin = char_w * 0.05

    clamped = []
    for i, ci in enumerate(char_items):
        _, y0, _, y1 = ci["bbox_norm"]
        new_x0 = (x_start + i * char_w + margin) / img_w
        new_x1 = (x_start + (i + 1) * char_w - margin) / img_w
        clamped.append({**ci, "bbox_norm": (new_x0, y0, new_x1, y1)})
    logger.debug("列归属错误，已用等宽 x 切分替换 %d 字", n)
    return clamped


def _estimate_question_x_range(
    region: dict,
    handwriting_boxes: list[dict] | None,
    img_w: int,
) -> tuple:
    """从 handwriting_boxes 中找出同列同 y 区间的手写框，返回 (x_start, x_end)。"""
    marker_bbox = region.get("marker_bbox")
    y_start = region["y_start"]
    y_end = region["y_end"]
    x_start, x_end = None, None
    if handwriting_boxes:
        marker_cx = (marker_bbox[0] + marker_bbox[2]) / 2 if marker_bbox else img_w / 2
        for hw_box in handwriting_boxes:
            hw_bbox = hw_box.get("bbox_pixel")
            if not hw_bbox:
                continue
            hw_y_center = (hw_bbox[1] + hw_bbox[3]) / 2
            if y_start <= hw_y_center < y_end:
                hw_cx = (hw_bbox[0] + hw_bbox[2]) / 2
                page_mid = img_w / 2
                if (marker_cx < page_mid) != (hw_cx < page_mid):
                    continue
                if x_start is None or hw_bbox[0] < x_start:
                    x_start = hw_bbox[0]
                if x_end is None or hw_bbox[2] > x_end:
                    x_end = hw_bbox[2]

    if x_start is None or x_end is None:
        if marker_bbox:
            x_start = marker_bbox[2]
            x_end = img_w
        else:
            x_start = x_start or 0
            x_end = x_end or img_w
    return x_start, x_end


def _estimate_question_y_range(
    region: dict,
    handwriting_boxes: list[dict] | None,
    img_h: int,
) -> tuple:
    """从 handwriting_boxes 中找出同列同 y 区间的手写框，返回实际 (y_start, y_end)。

    当 region 过大（如整个下半页）时，用 handwriting_boxes 的实际书写 y 范围替代，
    防止字符 bbox 被拉到 region 底部（P2-6 下联问题）。
    """
    y_start = region["y_start"]
    y_end = region["y_end"]
    region_h = y_end - y_start

    # region 过大（>400px）且 handwriting_boxes 可用时，用实际书写 y 范围
    if region_h > 400 and handwriting_boxes:
        marker_bbox = region.get("marker_bbox")
        marker_cx = (marker_bbox[0] + marker_bbox[2]) / 2 if marker_bbox else img_h / 2
        hw_ys = []
        for hw_box in handwriting_boxes:
            hw_bbox = hw_box.get("bbox_pixel")
            if not hw_bbox:
                continue
            hw_y_center = (hw_bbox[1] + hw_bbox[3]) / 2
            hw_cx = (hw_bbox[0] + hw_bbox[2]) / 2
            page_mid = img_h  # 用 img_h 避免与 x 列判断冲突
            if (marker_cx < page_mid) == (hw_cx < page_mid):
                hw_ys.append((hw_bbox[1], hw_bbox[3]))
        if hw_ys:
            actual_y_start = min(y for y, _ in hw_ys)
            actual_y_end = max(y for _, y in hw_ys)
            # 只在实际范围比 region 范围更紧凑时使用
            if actual_y_end - actual_y_start < region_h * 0.7:
                logger.debug("region 过大 (%dpx)，用 handwriting_boxes y 范围替代 (%dpx)",
                             region_h, actual_y_end - actual_y_start)
                return actual_y_start, actual_y_end

    return y_start, y_end


def _clamp_chars_y_to_region(
    char_items: list[dict],
    region: dict,
    img_h: int,
    handwriting_boxes: list[dict] | None = None,
) -> list[dict]:
    """将 chars 的 y 坐标约束到 region 的 y 范围内。

    模型 bbox 的 y 可能系统性偏高/偏低，用 region y_start/y_end 约束，
    同时保留模型的 x 坐标（x 通常更准）。
    """
    y_start, y_end = _estimate_question_y_range(region, handwriting_boxes, img_h)
    if y_start >= y_end:
        return char_items

    clamped = []
    for ci in char_items:
        x0, y0, x1, y1 = ci["bbox_norm"]
        # 将 y 约束到 region 范围内
        new_y0 = max(y_start / img_h, min(y0, y_end / img_h))
        new_y1 = max(y_start / img_h, min(y1, y_end / img_h))
        if new_y1 <= new_y0:
            new_y1 = new_y0 + 0.01
        clamped.append({**ci, "bbox_norm": (x0, new_y0, x1, new_y1)})
    return clamped


def _equal_width_split(
    chars: list[str],
    x_start: float,
    x_end: float,
    y_start: float,
    y_end: float,
    page_idx: int,
    q_idx: int,
    img_w: int,
    img_h: int,
) -> list[dict]:
    """等宽切分字符生成 per-char dict（回退路径使用，零行为变化）。"""
    region_w = x_end - x_start
    if region_w <= 0:
        return []
    char_w = region_w / len(chars)
    margin = char_w * 0.05

    results = []
    for j, ch in enumerate(chars):
        cx0 = int(x_start + j * char_w + margin)
        cx2 = int(x_start + (j + 1) * char_w - margin)
        if cx2 <= cx0:
            cx0 = int(x_start + j * char_w)
            cx2 = int(x_start + (j + 1) * char_w)
        results.append({
            "page": page_idx,
            "bbox_pixel": (cx0, y_start, cx2, y_end),
            "char": ch,
            "confidence": 0.85,
            "question_idx": q_idx,
            "img_pixel_w": img_w,
            "img_pixel_h": img_h,
            "engine": "qwen_vl_page_level",
        })
    return results


def _split_embedded_questions(answer: str, start_q: int) -> list[tuple[int, str]]:
    """检测答案文本中是否内嵌了额外题号（如 "（2）海内存知己"），拆分为多题。

    返回: [(q_num, answer_text), ...]
    """
    import re
    # 匹配内嵌题号: （2）xxx 或 (2) xxx
    pattern = re.compile(r"[（(](\d+)[)）]\s*(.+?)(?=[（(]\d+[)）]|$)")
    matches = pattern.findall(answer)
    if not matches:
        return [(start_q, answer)]
    result = []
    for q_str, text in matches:
        q = int(q_str)
        text = text.strip()
        if text:
            result.append((q, text))
    return result if result else [(start_q, answer)]


def _clean_answer_text(text: str) -> str:
    """清理答案文本，去除批改结论、解释说明等非答案内容。

    例如："随君直到夜郎西 —— ✅ 正确" → "随君直到夜郎西"
          "（1）随君直到夜郎西" → "随君直到夜郎西"
    """
    import re

    # 去掉以 —— 或 | 开头的批改结论部分
    text = re.split(r"\s*(?:——|\|)\s*", text, maxsplit=1)[0]
    # 去掉常见的解释前缀
    text = re.sub(r"^(?:学生(?:填|答|写)|答案|正确答案|标准答案|应填|可填)[:：是]\s*", "", text)
    # 去掉引号包裹
    text = text.strip('""')
    text = text.strip("''")
    return text.strip()


def _is_explain_line(line: str) -> bool:
    """判断一行是否为解释/说明/理由等非答案文本。"""
    explain_prefixes = [
        "题目", "理由", "原因", "解析", "分析", "说明", "注意", "结论",
        "总结", "建议", "教学建议", "扣分", "错误点", "错误类型",
        "实际应为", "例如", "正确做法", "学生写", "学生答", "学生填",
        "本题", "本小题", "本题答案", "→", "➡️", "⚠️", "📌", "❗", "✅", "❌",
    ]
    line_lower = line.strip()
    for prefix in explain_prefixes:
        if line_lower.startswith(prefix):
            return True
    # 包含明显解释性关键词且较长
    explain_keywords = ["理由", "扣分原因", "教学建议", "建议评分", "错误点", "错误类型"]
    if any(kw in line for kw in explain_keywords):
        return True
    return False


def _parse_page_level_response(response: str) -> list[dict]:
    """解析整页识别响应，返回按顺序排列的答案列表。

    返回: [{"q_num": int|None, "q_format": str, "answer": str,
            "correct": bool|None, "error": str}, ...]
    """
    import re

    results: list[dict] = []
    last_q_num = 0  # 记录上一个题号，用于判断是否重复

    for line in response.strip().split("\n"):
        line = line.strip()
        if not line:
            continue

        # 过滤列表标记行
        if line in ("-", "—", "*"):
            continue

        # 过滤解释性/说明性行
        if _is_explain_line(line):
            continue

        # 匹配: - （1）答案 —— ✅ 正确  或  （1）答案 | 正确 | 说明
        _PAREN_RE = re.compile(
            r"(?:-\s*)?[（(](\d+)[)）]\s*(.+?)(?:\s*(?:——|\|)\s*(✅\s*正确|❌\s*错误|正确|错误)(?:\s*[（(].*?[)）])?)?(?:\s*\|\s*(.*))?$")
        m = _PAREN_RE.match(line)
        if m:
            q_num = int(m.group(1))
            answer = _clean_answer_text(m.group(2).strip())
            correct_str = m.group(3) or ""
            is_correct = "正确" in correct_str if correct_str else None
            error = m.group(4).strip() if m.group(4) else ""
            if "未作答" not in answer and answer:
                if q_num <= last_q_num:
                    q_num = last_q_num + 1
                # 检测答案文本中是否内嵌了额外题号（如 "（2）海内存知己"），拆分
                sub_items = _split_embedded_questions(answer, q_num)
                if len(sub_items) > 1:
                    for sq, sa in sub_items:
                        if sq <= last_q_num:
                            sq = last_q_num + 1
                        results.append({
                            "q_num": sq, "q_format": "paren",
                            "answer": sa, "correct": None, "error": "",
                        })
                        last_q_num = sq
                else:
                    results.append({
                        "q_num": q_num,
                        "q_format": "paren",
                        "answer": answer,
                        "correct": is_correct,
                        "error": error,
                    })
                    last_q_num = q_num
            continue

        # 带圈数字: ① 答案
        _CIRCLED = {'①':1,'②':2,'③':3,'④':4,'⑤':5,'⑥':6,'⑦':7,'⑧':8,'⑨':9,'⑩':10}
        m = re.match(r"(?:-\s*)?([①②③④⑤⑥⑦⑧⑨⑩])\s*[.、]?\s*(.+)", line)
        if m:
            circled = m.group(1)
            answer = _clean_answer_text(m.group(2).strip())
            if answer:
                q_num = _CIRCLED.get(circled, last_q_num + 1)
                results.append({
                    "q_num": q_num, "q_format": "circle",
                    "answer": answer, "correct": None, "error": "",
                })
                last_q_num = q_num
            continue

        # 数字+点: 1. 答案
        m = re.match(r"(?:-\s*)?(\d+)\s*[.、]\s*(.+)", line)
        if m:
            answer = _clean_answer_text(m.group(2).strip())
            if answer and not _is_explain_line(answer):
                q_num = int(m.group(1))
                results.append({
                    "q_num": q_num, "q_format": "dot",
                    "answer": answer, "correct": None, "error": "",
                })
                last_q_num = q_num
            continue

        # 无题号行不再追加到答案，避免解释性文字混入

    return results


def _estimate_confidence(
    char_items: list[dict],
    std_chars: list[str],
    is_invalid: bool,
) -> float:
    """P1-4: 估算伪置信度，让 needs_review 可触发。

    - invalid（坐标无效）→ 0.50
    - 字数匹配标准答案 → 0.90
    - 字数不匹配 → 0.70
    - 无标准答案参照 → 0.85
    """
    if is_invalid:
        return 0.50
    if not std_chars:
        return 0.85
    if len(char_items) == len(std_chars):
        return 0.90
    return 0.70


def _split_merged_json_chars(
    char_items: list[dict],
    q_idx: int,
    std_answers_by_qidx: dict[int, list[str]],
    region_by_qidx: dict[int, dict],
    page_idx: int,
    img_w: int,
    img_h: int,
    used_regions: set[int],
) -> list[dict]:
    """P0-2: 将合并的 JSON chars 按标准答案字数拆分为多题。

    模型可能把 (7)(8) 合并输出为一题。按标准答案字数切分：
      (7) 标准答案 5 字 → char_items[:5] → q_idx
      (8) 标准答案 5 字 → char_items[5:10] → q_idx+1

    找不到下一题标准答案时，剩余全部归当前题。
    """
    results: list[dict] = []
    remaining = list(char_items)
    current_q_idx = q_idx
    fmt = _q_idx_format_range(q_idx)

    while remaining:
        std_chars = std_answers_by_qidx.get(current_q_idx, [])
        if std_chars and len(remaining) > len(std_chars) + 2:
            # 取标准答案字数个字归当前题
            take = len(std_chars)
        else:
            # 剩余全部归当前题
            take = len(remaining)

        chunk = remaining[:take]
        remaining = remaining[take:]

        # 查找当前题的 region（可能不存在）
        region = region_by_qidx.get(current_q_idx)
        if region:
            used_regions.add(current_q_idx)
            # P2-6: clamp 到列范围
            chunk = _clamp_chars_to_column(chunk, region, img_w, handwriting_boxes=None)

        conf = _estimate_confidence(chunk, std_chars, is_invalid=False)
        for ci in chunk:
            pixel_bbox = _norm_to_pixel_bbox(ci["bbox_norm"], img_w, img_h)
            results.append({
                "page": page_idx,
                "bbox_pixel": pixel_bbox,
                "char": ci["char"],
                "confidence": conf,
                "question_idx": current_q_idx,
                "img_pixel_w": img_w,
                "img_pixel_h": img_h,
                "engine": "qwen_vl_page_level",
            })

        # 找下一题 q_idx（同格式范围内递增）
        next_q_idx = None
        for candidate in sorted(std_answers_by_qidx.keys()):
            if candidate > current_q_idx and _q_idx_format_range(candidate) == fmt:
                next_q_idx = candidate
                break

        if next_q_idx is None or not remaining:
            break
        current_q_idx = next_q_idx

    return results


def recognize_page_level(
    img,
    question_regions: list[dict],
    page_idx: int = 0,
    handwriting_boxes: list[dict] | None = None,
    answers_text: str = "",
) -> list[dict]:
    """用 Qwen3-VL 识别整页，按题目输出结果，转换为逐字格式。

    优先走 JSON 逐字坐标路径（模型直接输出每个字的归一化 bbox）；
    JSON 解析失败或某题坐标无效时回退到旧文本解析 + 等宽切分。

    参数:
        img: RGB numpy array (H, W, 3)
        question_regions: layout_analyzer 输出的题目区域列表
        page_idx: 页码
        handwriting_boxes: layout_analyzer 输出的手写区域列表（回退路径用）
        answers_text: 原始多行标准答案文本，注入 prompt

    返回:
        与 recognize_handwriting() 相同格式的逐字结果列表。
    """
    if not isinstance(img, np.ndarray):
        img = np.array(Image.open(img) if isinstance(img, str) else img)

    img_h, img_w = img.shape[:2]
    t_start = time.time()

    # 调用 Qwen3-VL 整页识别
    response = _call_qwen_vl_page_level(img, answers_text)
    if not response:
        logger.warning("第 %d 页: Qwen3-VL 整页识别无结果", page_idx + 1)
        return []

    logger.info("第 %d 页: Qwen3-VL 整页识别原始响应:\n%s", page_idx + 1, response)

    # 构建 q_idx → region 查找表
    region_by_qidx: dict[int, dict] = {}
    for region in question_regions:
        region_by_qidx[region["q_idx"]] = region

    all_results: list[dict] = []
    used_regions: set[int] = set()

    # P0-1/P0-2: 解析标准答案，用于 invalid 回退和合并题拆分
    std_answers_by_qidx = _parse_answers_text(answers_text)

    # ---- 优先尝试 JSON 逐字坐标路径 ----
    parsed_json = _parse_page_level_json(response)

    if parsed_json is not None:
        import re as _re
        json_success_count = 0
        fallback_count = 0
        logger.info("第 %d 页: JSON 解析成功，%d 道题", page_idx + 1, len(parsed_json))

        for q_item in parsed_json:
            q_marker = q_item.get("q_marker", "")
            char_items = q_item.get("chars", [])
            is_invalid = q_item.get("invalid", False)

            # 题号 → q_idx（复用 layout_analyzer 逻辑）
            target_q_idx = _question_marker_to_q_idx(q_marker) if q_marker else None

            # P0-3: q_idx 匹配失败时，用首字像素坐标兜底定位（格式感知）
            # 注意：bbox 兜底只对有题号格式但未匹配的题生效；
            # 无题号格式的 q_marker（如"下联："）若 region 已被占用则跳过，避免覆盖
            if (target_q_idx is None or target_q_idx not in region_by_qidx) and char_items:
                first_bbox_norm = char_items[0].get("bbox_norm")
                if first_bbox_norm:
                    first_pixel = _norm_to_pixel_bbox(first_bbox_norm, img_w, img_h)
                    # 推断期望格式，防止 ① 误匹配到 (3) 等跨格式 region
                    expected_fmt = None
                    has_q_format = False
                    if any(c in q_marker for c in "①②③④⑤⑥⑦⑧⑨⑩"):
                        expected_fmt = "circle"
                        has_q_format = True
                    elif _re.match(r"\d+\s*[.、]", q_marker):
                        expected_fmt = "dot"
                        has_q_format = True
                    elif _re.match(r"[（(]\d+[)）]", q_marker):
                        expected_fmt = "paren"
                        has_q_format = True

                    if has_q_format:
                        # 先尝试格式精确匹配
                        target_q_idx = _match_q_idx_by_bbox(
                            first_pixel, region_by_qidx, img_w,
                            expected_format=expected_fmt)
                        # 格式匹配失败 → 退化为无格式匹配（跳过已占用 region）
                        if target_q_idx is None:
                            for candidate_qi, candidate_region in region_by_qidx.items():
                                if candidate_qi in used_regions:
                                    continue
                                cy = (first_pixel[1] + first_pixel[3]) / 2
                                if not (candidate_region["y_start"] <= cy < candidate_region["y_end"]):
                                    continue
                                mb = candidate_region.get("marker_bbox")
                                if mb:
                                    marker_cx = (mb[0] + mb[2]) / 2
                                    char_cx = (first_pixel[0] + first_pixel[2]) / 2
                                    if (marker_cx < img_w / 2) != (char_cx < img_w / 2):
                                        continue
                                target_q_idx = candidate_qi
                                break
                    else:
                        # 无题号格式（如"下联："）→ 尝试匹配未占用的 region
                        for candidate_qi, candidate_region in region_by_qidx.items():
                            if candidate_qi in used_regions:
                                continue
                            cy = (first_pixel[1] + first_pixel[3]) / 2
                            if not (candidate_region["y_start"] <= cy < candidate_region["y_end"]):
                                continue
                            marker_bbox = candidate_region.get("marker_bbox")
                            if marker_bbox:
                                marker_cx = (marker_bbox[0] + marker_bbox[2]) / 2
                                char_cx = (first_pixel[0] + first_pixel[2]) / 2
                                if (marker_cx < img_w / 2) != (char_cx < img_w / 2):
                                    continue
                            target_q_idx = candidate_qi
                            break

            if target_q_idx is None or target_q_idx not in region_by_qidx:
                logger.debug("题号 '%s' (q_idx=%s) 无匹配区域，跳过", q_marker, target_q_idx)
                continue

            region = region_by_qidx[target_q_idx]
            used_regions.add(target_q_idx)
            q_idx = region["q_idx"]

            # 该题所有字都有合法 bbox → 用模型坐标
            if char_items and not is_invalid:
                std_chars = std_answers_by_qidx.get(q_idx, [])

                # P0-2: 检测合并题（chars 数 > 标准答案字数 + 2）
                if std_chars and len(char_items) > len(std_chars) + 2:
                    split_results = _split_merged_json_chars(
                        char_items, q_idx, std_answers_by_qidx,
                        region_by_qidx, page_idx, img_w, img_h, used_regions,
                    )
                    all_results.extend(split_results)
                    json_success_count += 1
                    logger.info("题号 '%s' q_idx=%d: 检测到合并题，已拆分 (%d 字 → %d 字)",
                                q_marker, q_idx, len(char_items), len(split_results))
                else:
                    # P3: 文字渲染模式 — 跳过坐标 clamp，直接使用模型原始坐标
                    from core.recognition_config import RENDER_TEXT_MODE
                    if RENDER_TEXT_MODE:
                        clamped_items = char_items
                    else:
                        clamped_items = _clamp_chars_to_column(char_items, region, img_w, handwriting_boxes)
                        clamped_items = _clamp_chars_y_to_region(clamped_items, region, img_h, handwriting_boxes)
                    # P1-4: 估算置信度
                    conf = _estimate_confidence(char_items, std_chars, is_invalid=False)
                    for ci in clamped_items:
                        pixel_bbox = _norm_to_pixel_bbox(ci["bbox_norm"], img_w, img_h)
                        all_results.append({
                            "page": page_idx,
                            "bbox_pixel": pixel_bbox,
                            "char": ci["char"],
                            "confidence": conf,
                            "question_idx": q_idx,
                            "img_pixel_w": img_w,
                            "img_pixel_h": img_h,
                            "engine": "qwen_vl_page_level",
                        })
                    json_success_count += 1
                    logger.debug("题号 '%s' q_idx=%d: 用模型坐标 (%d 字, conf=%.2f)",
                                 q_marker, q_idx, len(char_items), conf)
            else:
                # P0-1: 坐标无效 → 从标准答案取文本，走等宽切分
                fallback_count += 1
                std_chars = std_answers_by_qidx.get(q_idx, [])
                if std_chars:
                    x_start, x_end = _estimate_question_x_range(
                        region, handwriting_boxes, img_w)
                    actual_y_start, actual_y_end = _estimate_question_y_range(
                        region, handwriting_boxes, img_h)
                    split = _equal_width_split(
                        std_chars, x_start, x_end,
                        actual_y_start, actual_y_end,
                        page_idx, q_idx, img_w, img_h,
                    )
                    # P1-4: invalid 题置信度 0.50
                    for r in split:
                        r["confidence"] = 0.50
                    all_results.extend(split)
                    logger.info("题号 '%s' q_idx=%d: 坐标无效，用标准答案等宽切分 (%d 字)",
                                q_marker, q_idx, len(split))
                else:
                    logger.warning("题号 '%s' q_idx=%d: 坐标无效且无标准答案，跳过",
                                   q_marker, q_idx)

        logger.info(
            "第 %d 页: JSON 路径 — 模型坐标 %d 题, 回退 %d 题",
            page_idx + 1, json_success_count, fallback_count,
        )

        # 若 JSON 路径一个字都没产出，回退到旧文本解析
        if not all_results and fallback_count > 0:
            logger.info("第 %d 页: JSON 路径无产出，回退旧文本解析", page_idx + 1)
            all_results = _fallback_text_path(
                response, region_by_qidx, question_regions,
                handwriting_boxes, page_idx, img_w, img_h, used_regions,
            )
    else:
        # ---- JSON 解析失败 → 旧文本解析路径 ----
        logger.info("第 %d 页: JSON 解析失败，回退旧文本解析", page_idx + 1)
        all_results = _fallback_text_path(
            response, region_by_qidx, question_regions,
            handwriting_boxes, page_idx, img_w, img_h, used_regions,
        )

    # 统计未匹配的区域
    unmatched = [r for r in question_regions if r["q_idx"] not in used_regions]
    if unmatched:
        logger.info(
            "第 %d 页: %d 个题目区域未匹配到答案: %s",
            page_idx + 1, len(unmatched),
            [r.get("marker_text", f"q_idx={r['q_idx']}") for r in unmatched],
        )

    elapsed = time.time() - t_start
    logger.info(
        "第 %d 页: Qwen3-VL 整页识别 → %d 个字符 (%.1fs)",
        page_idx + 1, len(all_results), elapsed,
    )
    return all_results


def _fallback_text_path(
    response: str,
    region_by_qidx: dict[int, dict],
    question_regions: list[dict],
    handwriting_boxes: list[dict] | None,
    page_idx: int,
    img_w: int,
    img_h: int,
    used_regions: set[int],
) -> list[dict]:
    """旧文本解析 + 等宽切分回退路径（零行为变化）。"""
    _QIDX_OFFSET_PAREN = 0
    _QIDX_OFFSET_DOT = 100
    _QIDX_OFFSET_CIRCLE = 200

    def _answer_to_q_idx(answer_item: dict) -> int | None:
        q_num = answer_item.get("q_num")
        if q_num is None:
            return None
        q_format = answer_item.get("q_format", "paren")
        if q_format == "circle":
            return q_num - 1 + _QIDX_OFFSET_CIRCLE
        elif q_format == "dot":
            return q_num - 1 + _QIDX_OFFSET_DOT
        else:
            return q_num - 1 + _QIDX_OFFSET_PAREN

    question_answers = _parse_page_level_response(response)
    if not question_answers:
        logger.warning("第 %d 页: 旧文本解析也无结果", page_idx + 1)
        return []

    logger.info("第 %d 页: 旧文本解析到 %d 道题", page_idx + 1, len(question_answers))

    all_results: list[dict] = []
    for answer_item in question_answers:
        answer_text = answer_item["answer"]
        chars = [ch for ch in answer_text if is_chinese_char(ch)]
        if not chars:
            continue

        # 过滤理由文本：长度 >30 字的答案不应参与比对
        if len(chars) > 30:
            logger.debug("跳过理由文本（%d 字）: '%s'", len(chars), answer_text[:30])
            continue

        target_q_idx = _answer_to_q_idx(answer_item)
        if target_q_idx is None or target_q_idx not in region_by_qidx:
            logger.debug("答案 '%s' (q_idx=%s) 无匹配区域，跳过", answer_text[:20], target_q_idx)
            continue

        region = region_by_qidx[target_q_idx]
        used_regions.add(target_q_idx)
        q_idx = region["q_idx"]
        actual_y_start, actual_y_end = _estimate_question_y_range(
            region, handwriting_boxes, img_h)

        x_start, x_end = _estimate_question_x_range(region, handwriting_boxes, img_w)
        split_results = _equal_width_split(
            chars, x_start, x_end, actual_y_start, actual_y_end,
            page_idx, q_idx, img_w, img_h,
        )
        all_results.extend(split_results)

    return all_results
