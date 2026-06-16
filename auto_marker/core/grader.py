"""
规则比对 — 将 OCR 识别结果与标准答案比对，生成状态。

状态:
    - correct (绿): 字符匹配且置信度 >= green_threshold
    - uncertain (橙): 字符匹配但置信度介于 orange ~ green 之间
    - wrong (红): 字符不匹配或置信度 < orange_threshold
"""

import logging

logger = logging.getLogger("grader")


def grade(ocr_results: list[dict], answers: list[str],
          green_threshold: float = 0.85,
          orange_threshold: float = 0.60) -> list[dict]:
    """
    逐字比对 OCR 结果与答案。

    ocr_results: OCR 引擎的输出，按 (page, y坐标) 排序后处理。
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

    # extract_student_answers 已按阅读顺序排序，直接使用
    sorted_results = ocr_results

    graded = []
    for idx, r in enumerate(sorted_results):
        expected = answers[idx] if idx < len(answers) else ""
        char = r["char"]
        conf = r["confidence"]

        if char == expected:
            if conf >= green_threshold:
                status = "correct"
            elif conf >= orange_threshold:
                status = "uncertain"
            else:
                status = "wrong"
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

    # 统计
    stats = {s: sum(1 for g in graded if g["status"] == s)
             for s in ("correct", "uncertain", "wrong")}
    logger.info("比对结果: 正确 %(correct)d, 存疑 %(uncertain)d, 错误 %(wrong)d, 共 %(total)d 字",
                {**stats, "total": len(graded)})

    return graded
