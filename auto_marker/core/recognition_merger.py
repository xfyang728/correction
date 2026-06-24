"""
双路识别结果合并 — 将 PaddleOCR 和 Qwen3-VL 的识别结果合并，提高准确率。

合并策略（通过 recognition_config.DUAL_MERGE_STRATEGY 配置）：
  - "vote": 两路一致时提高置信度，不一致时取高置信度方
  - "paddle_priority": PaddleOCR 为主，Qwen3-VL 仅在低置信度时覆盖
  - "qwen_priority": Qwen3-VL 文本为主，PaddleOCR 提供 bbox
"""

import logging
from collections import defaultdict

from core.recognition_config import (
    DUAL_MERGE_STRATEGY,
    PADDLE_LOW_CONF_THRESHOLD,
)

logger = logging.getLogger("recognition_merger")


def _group_by_question(results: list[dict]) -> dict:
    """按 question_idx 分组，每组内按 bbox 左边缘排序。

    返回: {question_idx: [result, ...]}
    question_idx 为 None 的归入 key=None。
    """
    groups: dict = defaultdict(list)
    for r in results:
        q_idx = r.get("question_idx")
        groups[q_idx].append(r)
    # 每组内按 bbox 左边缘 (x0) 排序，保证阅读顺序
    for q_idx in groups:
        groups[q_idx].sort(key=lambda r: r["bbox_pixel"][0])
    return dict(groups)


def _merge_vote(paddle_groups: dict, qwen_groups: dict) -> list[dict]:
    """vote 策略：两路结果逐组比较合并。"""
    all_q_indices = set(paddle_groups.keys()) | set(qwen_groups.keys())
    merged = []

    for q_idx in sorted(all_q_indices, key=lambda x: (x is None, x or 0)):
        paddle_chars = paddle_groups.get(q_idx, [])
        qwen_chars = qwen_groups.get(q_idx, [])

        if not paddle_chars:
            # 只有 Qwen3-VL 结果
            merged.extend(qwen_chars)
            continue
        if not qwen_chars:
            # 只有 PaddleOCR 结果
            merged.extend(paddle_chars)
            continue

        # 两路都有结果 — 逐字符比较
        paddle_text = "".join(r["char"] for r in paddle_chars)
        qwen_text = "".join(r["char"] for r in qwen_chars)

        # 长度差异过大时，以 PaddleOCR 长度为基准截断 Qwen3-VL（防止过识别）
        _MAX_LEN_RATIO = 1.5
        if len(qwen_chars) > len(paddle_chars) * _MAX_LEN_RATIO:
            qwen_chars = qwen_chars[:int(len(paddle_chars) * _MAX_LEN_RATIO)]
            qwen_text = "".join(r["char"] for r in qwen_chars)
            logger.debug(
                "题 %s: Qwen3-VL 过长 (%d vs %d) → 截断到 %d",
                q_idx, len(qwen_chars) + int(len(paddle_chars) * _MAX_LEN_RATIO),
                len(paddle_chars), int(len(paddle_chars) * _MAX_LEN_RATIO),
            )

        if paddle_text == qwen_text:
            # 完全一致 → 提高置信度
            for r in paddle_chars:
                r["confidence"] = min(r["confidence"] * 1.1, 1.0)
                r["merged"] = "agree"
            merged.extend(paddle_chars)
            logger.debug("题 %s: 两路一致 '%s' → 置信度提升", q_idx, paddle_text)
        else:
            # 不一致 → 逐字符取高置信度方
            max_len = max(len(paddle_chars), len(qwen_chars))
            for i in range(max_len):
                p = paddle_chars[i] if i < len(paddle_chars) else None
                q = qwen_chars[i] if i < len(qwen_chars) else None

                if p and q:
                    if p["char"] == q["char"]:
                        # 该字符一致，取高置信度
                        winner = p if p["confidence"] >= q["confidence"] else q
                        winner["confidence"] = min(winner["confidence"] * 1.05, 1.0)
                        winner["merged"] = "char_agree"
                        merged.append(winner)
                    else:
                        # 该字符不一致，取高置信度方
                        if p["confidence"] >= q["confidence"]:
                            p["merged"] = "paddle_win"
                            merged.append(p)
                        else:
                            # Qwen3-VL 赢了，但需要保留 PaddleOCR 的 bbox
                            q_copy = dict(q)
                            q_copy["bbox_pixel"] = p["bbox_pixel"]
                            q_copy["img_pixel_w"] = p["img_pixel_w"]
                            q_copy["img_pixel_h"] = p["img_pixel_h"]
                            q_copy["merged"] = "qwen_win"
                            merged.append(q_copy)
                elif p:
                    merged.append(p)
                else:
                    merged.append(q)

            logger.debug(
                "题 %s: 两路不一致 P='%s' Q='%s' → 逐字符取优",
                q_idx, paddle_text, qwen_text,
            )

    return merged


def _merge_paddle_priority(paddle_groups: dict, qwen_groups: dict) -> list[dict]:
    """paddle_priority 策略：PaddleOCR 为主，Qwen3-VL 仅在低置信度时覆盖。"""
    all_q_indices = set(paddle_groups.keys()) | set(qwen_groups.keys())
    merged = []

    for q_idx in sorted(all_q_indices, key=lambda x: (x is None, x or 0)):
        paddle_chars = paddle_groups.get(q_idx, [])
        qwen_chars = qwen_groups.get(q_idx, [])

        if not paddle_chars:
            merged.extend(qwen_chars)
            continue
        if not qwen_chars:
            merged.extend(paddle_chars)
            continue

        # PaddleOCR 平均置信度
        paddle_avg_conf = sum(r["confidence"] for r in paddle_chars) / len(paddle_chars)

        if paddle_avg_conf >= PADDLE_LOW_CONF_THRESHOLD:
            # PaddleOCR 置信度足够，直接用
            merged.extend(paddle_chars)
        else:
            # PaddleOCR 置信度低，用 Qwen3-VL 文本 + PaddleOCR bbox
            qwen_text = "".join(r["char"] for r in qwen_chars)
            for i, p in enumerate(paddle_chars):
                if i < len(qwen_chars):
                    p["char"] = qwen_chars[i]["char"]
                    p["confidence"] = max(p["confidence"], qwen_chars[i]["confidence"])
                    p["merged"] = "qwen_override"
                merged.append(p)
            logger.debug(
                "题 %s: PaddleOCR 低置信度 (%.2f) → Qwen3-VL 覆盖 '%s'",
                q_idx, paddle_avg_conf, qwen_text,
            )

    return merged


def _merge_qwen_priority(paddle_groups: dict, qwen_groups: dict) -> list[dict]:
    """qwen_priority 策略：Qwen3-VL 文本为主，PaddleOCR 提供 bbox。"""
    all_q_indices = set(paddle_groups.keys()) | set(qwen_groups.keys())
    merged = []

    for q_idx in sorted(all_q_indices, key=lambda x: (x is None, x or 0)):
        paddle_chars = paddle_groups.get(q_idx, [])
        qwen_chars = qwen_groups.get(q_idx, [])

        if not qwen_chars:
            merged.extend(paddle_chars)
            continue
        if not paddle_chars:
            merged.extend(qwen_chars)
            continue

        # 用 Qwen3-VL 的字符 + PaddleOCR 的 bbox 等宽切分
        qwen_text = "".join(r["char"] for r in qwen_chars)
        # 取 PaddleOCR 的 bbox 范围
        p_x0 = min(r["bbox_pixel"][0] for r in paddle_chars)
        p_x2 = max(r["bbox_pixel"][2] for r in paddle_chars)
        p_y0 = paddle_chars[0]["bbox_pixel"][1]
        p_y2 = paddle_chars[0]["bbox_pixel"][3]
        img_w = paddle_chars[0]["img_pixel_w"]
        img_h = paddle_chars[0]["img_pixel_h"]

        region_w = p_x2 - p_x0
        char_w = region_w / len(qwen_text) if qwen_text else region_w
        margin = char_w * 0.05

        for i, ch in enumerate(qwen_text):
            cx0 = int(p_x0 + i * char_w + margin)
            cx2 = int(p_x0 + (i + 1) * char_w - margin)
            if cx2 <= cx0:
                cx0 = int(p_x0 + i * char_w)
                cx2 = int(p_x0 + (i + 1) * char_w)

            # Qwen3-VL 置信度取 PaddleOCR 对应位置的值（如有）
            conf = qwen_chars[i]["confidence"] if i < len(qwen_chars) else 0.85
            if i < len(paddle_chars):
                conf = max(conf, paddle_chars[i]["confidence"])

            merged.append({
                "page": paddle_chars[0]["page"],
                "bbox_pixel": (cx0, p_y0, cx2, p_y2),
                "char": ch,
                "confidence": conf,
                "question_idx": q_idx,
                "img_pixel_w": img_w,
                "img_pixel_h": img_h,
                "engine": "qwen_vl",
                "merged": "qwen_text_paddle_bbox",
            })

        logger.debug("题 %s: Qwen3-VL 文本 + PaddleOCR bbox → '%s'", q_idx, qwen_text)

    return merged


def merge_recognition_results(
    paddle_results: list[dict],
    qwen_results: list[dict],
    strategy: str | None = None,
) -> list[dict]:
    """合并 PaddleOCR 和 Qwen3-VL 的识别结果。

    参数:
        paddle_results: PaddleOCR 路径的逐字结果
        qwen_results: Qwen3-VL 路径的逐字结果
        strategy: 合并策略，默认使用配置文件中的 DUAL_MERGE_STRATEGY

    返回:
        合并后的逐字结果列表（格式与 grader 兼容）
    """
    if strategy is None:
        strategy = DUAL_MERGE_STRATEGY

    if not paddle_results:
        logger.info("PaddleOCR 无结果，直接使用 Qwen3-VL 结果 (%d 字符)", len(qwen_results))
        return qwen_results
    if not qwen_results:
        logger.info("Qwen3-VL 无结果，直接使用 PaddleOCR 结果 (%d 字符)", len(paddle_results))
        return paddle_results

    paddle_groups = _group_by_question(paddle_results)
    qwen_groups = _group_by_question(qwen_results)

    logger.info(
        "双路合并 (strategy=%s): PaddleOCR %d 字符 / %d 组, Qwen3-VL %d 字符 / %d 组",
        strategy,
        len(paddle_results), len(paddle_groups),
        len(qwen_results), len(qwen_groups),
    )

    if strategy == "vote":
        merged = _merge_vote(paddle_groups, qwen_groups)
    elif strategy == "paddle_priority":
        merged = _merge_paddle_priority(paddle_groups, qwen_groups)
    elif strategy == "qwen_priority":
        merged = _merge_qwen_priority(paddle_groups, qwen_groups)
    else:
        logger.warning("未知合并策略 '%s'，回退到 vote", strategy)
        merged = _merge_vote(paddle_groups, qwen_groups)

    # 统计合并结果
    merge_stats = defaultdict(int)
    for r in merged:
        merge_stats[r.get("merged", "unmerged")] += 1
    logger.info("合并结果统计: %s", dict(merge_stats))
    logger.info("合并后总字符数: %d", len(merged))

    return merged
