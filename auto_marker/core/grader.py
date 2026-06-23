"""
规则比对 — 将 OCR 识别结果与标准答案比对，生成状态。

使用 Needleman-Wunsch 动态规划对齐算法，容忍插入/删除/替换错位。
结合置信度门控区分 correct / uncertain / wrong。
"""

import logging

logger = logging.getLogger("grader")


def grade(ocr_results: list[dict], answers: list[str],
          green_threshold: float = 0.85,
          orange_threshold: float = 0.60,
          skip_threshold: float = 0.30) -> list[dict]:
    """
    使用动态规划对齐 OCR 结果与答案，容忍错位。

    ocr_results: OCR 引擎的输出，已按阅读顺序排序。
    answers: 标准答案列表，如 ["春", "眠", "不", "觉", "晓"]
    green_threshold: 正确且置信度 >= 此值 → correct
    orange_threshold: 正确且置信度 >= 此值 → uncertain；不匹配且置信度 < 此值 → uncertain
    skip_threshold: 置信度 < 此值的字不参与比对

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

    # 置信度门控：极低置信度的字跳过，不参与比对
    filtered_ocr = [r for r in ocr_results if r["confidence"] >= skip_threshold]
    skipped = len(ocr_results) - len(filtered_ocr)
    if skipped:
        logger.info("跳过 %d 个低置信度字（< %.2f）", skipped, skip_threshold)

    ocr_chars = [r["char"] for r in filtered_ocr]
    ocr_confs = [r["confidence"] for r in filtered_ocr]

    # 动态规划对齐（带置信度加权）
    aligned = _align(ocr_chars, answers, ocr_confs)

    # 生成比对结果
    graded = []
    for ocr_idx, ans_idx in aligned:
        if ocr_idx is not None and ans_idx is not None:
            # 匹配对
            r = filtered_ocr[ocr_idx]
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
                "question_idx": r.get("question_idx"),
                "img_pixel_w": r["img_pixel_w"],
                "img_pixel_h": r["img_pixel_h"],
            })
        elif ocr_idx is not None:
            # OCR 多识别的字（答案中没有对应位置）
            r = filtered_ocr[ocr_idx]
            graded.append({
                "page": r["page"],
                "bbox_pixel": r["bbox_pixel"],
                "char": r["char"],
                "expected": "",
                "confidence": r["confidence"],
                "status": "wrong",
                "question_idx": r.get("question_idx"),
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


def _align(ocr_chars: list[str], answers: list[str],
           ocr_confs: list[float] = None) -> list[tuple]:
    """Needleman-Wunsch 全局对齐算法（带置信度加权）。

    高置信度的匹配获得更高分，低置信度的字更容易被当作"多余字"跳过。

    返回对齐路径列表，每个元素为 (ocr_idx, ans_idx)，
    其中 None 表示插入/删除。
    """
    m, n = len(ocr_chars), len(answers)

    if ocr_confs is None:
        ocr_confs = [0.9] * m  # 默认高置信度

    # 评分参数
    GAP = -2

    # DP 矩阵
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        dp[i][0] = i * GAP
    for j in range(1, n + 1):
        dp[0][j] = j * GAP

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            # 匹配/替换得分：置信度越高，匹配奖励越大，不匹配惩罚越小
            conf = ocr_confs[i - 1]
            if ocr_chars[i - 1] == answers[j - 1]:
                # 匹配：基础分 2 + 置信度奖励（0~1）
                score = 2 + conf
            else:
                # 不匹配：置信度越高惩罚越大（更确定地错了），
                # 置信度低惩罚小（可能是误识，倾向跳过）
                score = -1 - conf
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
            conf = ocr_confs[i - 1]
            if ocr_chars[i - 1] == answers[j - 1]:
                score = 2 + conf
            else:
                score = -1 - conf
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


def summarize_by_question(graded: list[dict],
                          question_regions: dict[int, list[dict]]
                          ) -> dict[int, dict]:
    """对批改结果按题目分组统计。

    参数:
        graded: grade() 的输出，每项需含 "question_idx" 字段
        question_regions: layout_analyzer.detect_question_regions() 的输出

    返回:
        {page_idx: {
            q_idx: {
                "total": 2,
                "correct": 2,
                "uncertain": 0,
                "wrong": 0,
                "all_correct": True,
                "marker_bbox": (x0,y0,x2,y2),  # 题号位置
            }
        }}

        如果某页无题号信息，则该页不在返回结果中。
    """
    from collections import defaultdict

    # 构建题号位置查找表: {page_idx: {q_idx: {marker_bbox, question_type}}}
    marker_map: dict[int, dict] = {}
    for page_idx, regions in question_regions.items():
        marker_map[page_idx] = {
            r["q_idx"]: {
                "marker_bbox": r["marker_bbox"],
                "question_type": r.get("question_type", "default"),
            }
            for r in regions
        }

    # 按 (page, question_idx) 分组
    groups: dict[tuple[int, int], list[dict]] = defaultdict(list)
    for g in graded:
        q_idx = g.get("question_idx")
        if q_idx is None:
            continue  # 未分配到题号的字不参与按题统计
        groups[(g["page"], q_idx)].append(g)

    result: dict[int, dict] = {}
    for (page_idx, q_idx), items in groups.items():
        total = len(items)
        correct = sum(1 for r in items if r["status"] == "correct")
        uncertain = sum(1 for r in items if r["status"] == "uncertain")
        wrong = sum(1 for r in items if r["status"] == "wrong")

        if page_idx not in result:
            result[page_idx] = {}

        marker_info = marker_map.get(page_idx, {}).get(q_idx, {})
        marker_bbox = marker_info.get("marker_bbox")
        question_type = marker_info.get("question_type", "default")
        result[page_idx][q_idx] = {
            "total": total,
            "correct": correct,
            "uncertain": uncertain,
            "wrong": wrong,
            "all_correct": correct == total and total > 0,
            "marker_bbox": marker_bbox,
            "question_type": question_type,
        }

    # 统计日志
    total_questions = sum(len(qs) for qs in result.values())
    correct_questions = sum(
        1 for qs in result.values() for q in qs.values() if q["all_correct"]
    )
    logger.info("按题统计: %d 题, 其中 %d 题全部正确", total_questions, correct_questions)
    return result
