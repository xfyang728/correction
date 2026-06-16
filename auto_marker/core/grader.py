"""
规则比对 — 将 OCR 识别结果与标准答案比对，生成状态。

使用 Needleman-Wunsch 动态规划对齐算法，容忍插入/删除/替换错位。
结合置信度门控区分 correct / uncertain / wrong。
"""

import logging

logger = logging.getLogger("grader")


def grade(ocr_results: list[dict], answers: list[str],
          green_threshold: float = 0.85,
          orange_threshold: float = 0.60) -> list[dict]:
    """
    使用动态规划对齐 OCR 结果与答案，容忍错位。

    ocr_results: OCR 引擎的输出，已按阅读顺序排序。
    answers: 标准答案列表，如 ["春", "眠", "不", "觉", "晓"]

    返回:
        [
            { "page": 0, "bbox_pixel": (x0, y0, x2, y2),
              "char": "春", "expected": "春", "confidence": 0.92,
              "status": "correct",
              "img_pixel_w": 1654, "img_pixel_h": 2339 },
            ...
        ]
    """
    if not ocr_results:
        logger.warning("OCR 结果为空，无法比对")
        return []

    ocr_chars = [r["char"] for r in ocr_results]

    # 动态规划对齐
    aligned = _align(ocr_chars, answers)

    # 生成比对结果
    graded = []
    for ocr_idx, ans_idx in aligned:
        if ocr_idx is not None and ans_idx is not None:
            # 匹配对
            r = ocr_results[ocr_idx]
            char = r["char"]
            expected = answers[ans_idx]
            conf = r["confidence"]

            if char == expected:
                if conf >= green_threshold:
                    status = "correct"
                elif conf >= orange_threshold:
                    status = "uncertain"
                else:
                    status = "wrong"
            else:
                # 字不匹配但位置对齐 — 置信度高则 wrong，低则 uncertain
                if conf < orange_threshold:
                    status = "uncertain"
                else:
                    status = "wrong"

            graded.append({
                "page": r["page"],
                "bbox_pixel": r["bbox_pixel"],
                "char": char,
                "expected": expected,
                "confidence": conf,
                "status": status,
                "img_pixel_w": r["img_pixel_w"],
                "img_pixel_h": r["img_pixel_h"],
            })
        elif ocr_idx is not None:
            # OCR 多识别的字（答案中没有对应位置）
            r = ocr_results[ocr_idx]
            graded.append({
                "page": r["page"],
                "bbox_pixel": r["bbox_pixel"],
                "char": r["char"],
                "expected": "",
                "confidence": r["confidence"],
                "status": "wrong",
                "img_pixel_w": r["img_pixel_w"],
                "img_pixel_h": r["img_pixel_h"],
            })
        # ans_idx is not None, ocr_idx is None 的情况（漏写）不生成条目

    # 统计
    stats = {s: sum(1 for g in graded if g["status"] == s)
             for s in ("correct", "uncertain", "wrong")}
    logger.info("比对结果: 正确 %(correct)d, 存疑 %(uncertain)d, 错误 %(wrong)d, 共 %(total)d 字",
                {**stats, "total": len(graded)})

    return graded


def _align(ocr_chars: list[str], answers: list[str]) -> list[tuple]:
    """Needleman-Wunsch 全局对齐算法。

    返回对齐路径列表，每个元素为 (ocr_idx, ans_idx)，
    其中 None 表示插入/删除。
    """
    m, n = len(ocr_chars), len(answers)

    # 评分矩阵
    MATCH = 2
    MISMATCH = -1
    GAP = -2

    # DP 矩阵
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        dp[i][0] = i * GAP
    for j in range(1, n + 1):
        dp[0][j] = j * GAP

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if ocr_chars[i - 1] == answers[j - 1]:
                score = MATCH
            else:
                score = MISMATCH
            dp[i][j] = max(
                dp[i - 1][j - 1] + score,  # 匹配/替换
                dp[i - 1][j] + GAP,          # OCR 多余字（删除）
                dp[i][j - 1] + GAP,          # 答案多余字（插入/漏写）
            )

    # 回溯
    aligned = []
    i, j = m, n
    while i > 0 or j > 0:
        if i > 0 and j > 0:
            if ocr_chars[i - 1] == answers[j - 1]:
                score = MATCH
            else:
                score = MISMATCH
            if dp[i][j] == dp[i - 1][j - 1] + score:
                aligned.append((i - 1, j - 1))
                i -= 1
                j -= 1
                continue
        if i > 0 and dp[i][j] == dp[i - 1][j] + GAP:
            aligned.append((i - 1, None))
            i -= 1
        else:
            aligned.append((None, j - 1))
            j -= 1

    aligned.reverse()
    return aligned