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


def _call_qwen_vl_page_level(img_array: np.ndarray, answers: list[str] | None = None) -> str:
    """调用 Qwen3-VL 识别整页+批改，返回结构化文本。"""
    import requests

    from core.recognition_config import (
        QWEN_VL_API_URL,
        QWEN_VL_MODEL,
        QWEN_VL_PAGE_LEVEL_PROMPT,
        QWEN_VL_TIMEOUT,
    )

    b64 = _img_to_base64(img_array)

    prompt = QWEN_VL_PAGE_LEVEL_PROMPT

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
                "max_tokens": 1024,
                "temperature": 0.1,
                "repeat_penalty": 1.1,
                "repeat_last_n": 64,
                "top_k": 40,
                "top_p": 0.95,
                "min_p": 0.05,
                "frequency_penalty": 0.0,
                "presence_penalty": 0.0,
            },
            timeout=QWEN_VL_TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"]
    except Exception as e:
        logger.warning("Qwen3-VL 整页识别 API 调用失败: %s", e)
        return ""


def _parse_page_level_response(response: str) -> list[dict]:
    """解析整页识别响应，返回按顺序排列的答案列表。

    处理新格式：(1) 学生答案 | 正确/错误 | 错误说明
    返回: [{"answer": "答案", "correct": True/False, "error": "错误说明"}, ...]
    """
    import re

    results: list[dict] = []
    last_q_num = 0  # 记录上一个题号，用于判断是否重复

    for line in response.strip().split("\n"):
        line = line.strip()
        if not line:
            continue

        # 匹配新格式: (1) 学生答案 | 正确/错误 | 错误说明
        m = re.match(r"[（(](\d+)[)）]\s*(.+?)\s*\|\s*(正确|错误)\s*\|\s*(.*)", line)
        if m:
            q_num = int(m.group(1))
            answer = m.group(2).strip()
            is_correct = m.group(3) == "正确"
            error = m.group(4).strip() if m.group(4) else ""
            # 过滤"未作答"
            if "未作答" not in answer:
                # 检测题号是否重复（如拼音填空的①②被识别成(1)(2)）
                if q_num <= last_q_num:
                    # 题号重复，说明是新的题型，重置题号
                    q_num = last_q_num + 1
                results.append({
                    "answer": answer,
                    "correct": is_correct,
                    "error": error,
                })
                last_q_num = q_num
            continue

        # 兼容旧格式：只提取答案文本
        # 去除题号前缀
        m = re.match(r"[（(](\d+)[)）]\s*(.+)", line)
        if m:
            q_num = int(m.group(1))
            answer = m.group(2).strip()
            # 检测题号是否重复
            if q_num <= last_q_num:
                q_num = last_q_num + 1
            results.append({"answer": answer, "correct": None, "error": ""})
            last_q_num = q_num
            continue

        m = re.match(r"[①②③④⑤⑥⑦⑧⑨⑩]\s*[.、]?\s*(.+)", line)
        if m:
            results.append({"answer": m.group(1).strip(), "correct": None, "error": ""})
            last_q_num += 1
            continue

        m = re.match(r"\d+\s*[.、]\s*(.+)", line)
        if m:
            results.append({"answer": m.group(1).strip(), "correct": None, "error": ""})
            last_q_num += 1
            continue

        # 无题号行：如果是汉字内容且长度合理，追加到上一个答案
        chinese_chars = [ch for ch in line if is_chinese_char(ch)]
        if len(chinese_chars) >= 2 and len(line) < 30:
            if results:
                # 追加到上一个答案
                results[-1]["answer"] += line
            else:
                results.append({"answer": line, "correct": None, "error": ""})
                last_q_num += 1

    # 拆分包含空格的答案（如 "采菊东篱下 悠然见南山" → 两个答案）
    split_result = []
    for item in results:
        parts = item["answer"].split()
        for part in parts:
            # 只保留包含至少2个中文字符的答案
            chinese_chars = [ch for ch in part if is_chinese_char(ch)]
            if len(chinese_chars) >= 2:
                split_result.append({
                    "answer": part,
                    "correct": item["correct"],
                    "error": item["error"],
                })

    return split_result


def recognize_page_level(
    img,
    question_regions: list[dict],
    page_idx: int = 0,
    handwriting_boxes: list[dict] | None = None,
) -> list[dict]:
    """用 Qwen3-VL 识别整页，按题目输出结果，转换为逐字格式。

    参数:
        img: RGB numpy array (H, W, 3)
        question_regions: layout_analyzer 输出的题目区域列表，每项含：
            - q_idx: 题目索引
            - y_start, y_end: y 区间
            - marker_bbox: 题号位置
        page_idx: 页码
        handwriting_boxes: layout_analyzer 输出的手写区域列表，用于确定实际书写 x 范围

    返回:
        与 recognize_handwriting() 相同格式的逐字结果列表。
    """
    if not isinstance(img, np.ndarray):
        img = np.array(Image.open(img) if isinstance(img, str) else img)

    img_h, img_w = img.shape[:2]

    t_start = time.time()

    # 调用 Qwen3-VL 整页识别
    response = _call_qwen_vl_page_level(img)
    if not response:
        logger.warning("第 %d 页: Qwen3-VL 整页识别无结果", page_idx + 1)
        return []

    logger.info("第 %d 页: Qwen3-VL 整页识别原始响应:\n%s", page_idx + 1, response)

    # 解析响应（返回 list[str]，按顺序对应 question_regions）
    question_answers = _parse_page_level_response(response)
    if not question_answers:
        logger.warning("第 %d 页: 解析整页响应无结果", page_idx + 1)
        return []

    logger.info(
        "第 %d 页: 解析到 %d 道题的答案",
        page_idx + 1, len(question_answers),
    )

    # 将答案按顺序与 question_regions 对应
    # 如果答案数量与题目数量不匹配，取较小值
    num_questions = min(len(question_answers), len(question_regions))
    if len(question_answers) != len(question_regions):
        logger.warning(
            "第 %d 页: 答案数量 (%d) 与题目数量 (%d) 不匹配，取前 %d 个",
            page_idx + 1, len(question_answers), len(question_regions), num_questions
        )

    # 为每道题生成逐字结果
    all_results: list[dict] = []
    for i in range(num_questions):
        region = question_regions[i]
        answer_item = question_answers[i]
        answer_text = answer_item["answer"] if isinstance(answer_item, dict) else answer_item
        q_idx = region["q_idx"]

        # 提取中文字符
        chars = [ch for ch in answer_text if is_chinese_char(ch)]
        if not chars:
            continue

        # 使用该题的 y 区间
        y_start = region["y_start"]
        y_end = region["y_end"]
        
        # 从 handwriting_boxes 中找出落在该 y 区间内的手写框，确定实际书写 x 范围
        x_start, x_end = None, None
        if handwriting_boxes:
            for hw_box in handwriting_boxes:
                hw_bbox = hw_box.get("bbox_pixel")
                if not hw_bbox:
                    continue
                hw_y_center = (hw_bbox[1] + hw_bbox[3]) / 2
                # 检查手写框是否在该题的 y 区间内
                if y_start <= hw_y_center < y_end:
                    if x_start is None or hw_bbox[0] < x_start:
                        x_start = hw_bbox[0]
                    if x_end is None or hw_bbox[2] > x_end:
                        x_end = hw_bbox[2]
        
        # 如果没有找到匹配的手写框，使用默认范围（题号右侧到页面右边缘）
        if x_start is None or x_end is None:
            marker_bbox = region.get("marker_bbox")
            if marker_bbox:
                x_start = marker_bbox[2]  # 从题号右侧开始
                x_end = img_w
            else:
                x_start = 0
                x_end = img_w

        # 等宽切分字符
        region_w = x_end - x_start
        char_w = region_w / len(chars)
        margin = char_w * 0.05

        for j, ch in enumerate(chars):
            cx0 = int(x_start + j * char_w + margin)
            cx2 = int(x_start + (j + 1) * char_w - margin)
            if cx2 <= cx0:
                cx0 = int(x_start + j * char_w)
                cx2 = int(x_start + (j + 1) * char_w)

            all_results.append({
                "page": page_idx,
                "bbox_pixel": (cx0, y_start, cx2, y_end),
                "char": ch,
                "confidence": 0.85,  # VL 模型无逐字置信度，给默认值
                "question_idx": q_idx,
                "img_pixel_w": img_w,
                "img_pixel_h": img_h,
                "engine": "qwen_vl_page_level",
            })

    elapsed = time.time() - t_start
    logger.info(
        "第 %d 页: Qwen3-VL 整页识别 → %d 个字符 (%.1fs)",
        page_idx + 1, len(all_results), elapsed,
    )
    return all_results
