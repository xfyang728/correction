"""
版面分析 — 基于文本检测框的布局元素分类。

阶段二：对文本检测器输出的检测框进行版面分析与分离。
  - 使用文本检测框的 bbox 高度进行手写/印刷分类
  - 替代方案：基于 PPStructureV3，但改为直接使用检测框更轻量、更快
  - 分类规则（200 DPI）：
    - handwriting（手写答案，h ≥ 70px）
    - printed_question（印刷体题目）
    - pinyin_hint（拼音提示，h < 70px）
  - 检测题目序号并对手写答案按题分组
"""

import logging
import re

logger = logging.getLogger("layout_analyzer")

_QUESTION_PATTERN = re.compile(r'[（(](\d+)[)）]|^(\d+)[.、]')

# 手写/印刷判别阈值（200 DPI 下）
HANDWRITTEN_HEIGHT_MIN = 70   # 手写字高度 ≥ 70px
PINYIN_HEIGHT_MAX = 50        # 拼音提示高度 ≤ 50px


def _bbox_h(bbox: tuple) -> int:
    """返回 bbox 高度。"""
    return bbox[3] - bbox[1]


def _bbox_center_y(bbox: tuple) -> float:
    """返回 bbox 中心 y。"""
    return (bbox[1] + bbox[3]) / 2.0


def analyze_layout(det_boxes: list[dict], img_h: int, page_idx: int = 0) -> dict:
    """基于文本检测框进行版面分析，返回分类后的区域信息。

    参数:
        det_boxes: text_detector.detect_text() 的输出，每项含：
            - bbox_pixel: (x0,y0,x2,y2) 外接矩形
            - confidence: 检测置信度
            - page: 页码
        img_h: 原图高度（用于题号区间推断）
        page_idx: 页码

    返回:
        {
            "handwriting_boxes": [
                {"page": 0, "bbox_pixel": (x0,y0,x2,y2),
                 "confidence": 0.88, "type": "handwriting"},
                ...
            ],
            "printed_boxes": [
                {"page": 0, "bbox_pixel": (x0,y0,x2,y2),
                 "confidence": 0.95, "type": "printed_question"},
                ...
            ],
            "all_regions": [
                {"page": 0, "bbox_pixel": (x0,y0,x2,y2),
                 "type": "handwriting", ...},
                ...
            ],
            "question_regions": {
                page_idx: [
                    {"q_idx": 0, "y_start": ..., "y_end": ...,
                     "marker_bbox": (x0,y0,x2,y2), "marker_text": "(1)"},
                    ...
                ]
            }
        }
    """
    handwriting_boxes: list[dict] = []
    printed_boxes: list[dict] = []
    all_regions: list[dict] = []

    for box in det_boxes:
        bbox = box["bbox_pixel"]
        conf = box.get("confidence", 0.9)
        h = _bbox_h(bbox)

        # ---- 分类逻辑（基于 bbox 高度） ----
        if h >= HANDWRITTEN_HEIGHT_MIN:
            region_type = "handwriting"
            handwriting_boxes.append({
                "page": page_idx,
                "bbox_pixel": bbox,
                "confidence": conf,
                "type": "handwriting",
            })
        elif h <= PINYIN_HEIGHT_MAX:
            region_type = "pinyin_hint"
            printed_boxes.append({
                "page": page_idx,
                "bbox_pixel": bbox,
                "confidence": conf,
                "type": "pinyin_hint",
            })
        else:
            region_type = "printed_question"
            printed_boxes.append({
                "page": page_idx,
                "bbox_pixel": bbox,
                "confidence": conf,
                "type": "printed_question",
            })

        all_regions.append({
            "page": page_idx,
            "bbox_pixel": bbox,
            "type": region_type,
            "confidence": conf,
        })

    logger.info(
        "第 %d 页: 手写 %d 个, 印刷 %d 个, 拼音 %d 个",
        page_idx + 1,
        len(handwriting_boxes),
        sum(1 for r in printed_boxes if r["type"] == "printed_question"),
        sum(1 for r in printed_boxes if r["type"] == "pinyin_hint"),
    )

    # ---- 题目序号检测（从印刷体区域中识别） ----
    question_regions = _detect_questions_from_boxes(
        printed_boxes, img_h, page_idx
    )

    # ---- 将题号分配到手写答案 ----
    _assign_to_handwriting(handwriting_boxes, question_regions, page_idx)

    return {
        "handwriting_boxes": handwriting_boxes,
        "printed_boxes": printed_boxes,
        "all_regions": all_regions,
        "question_regions": {page_idx: question_regions} if question_regions else {},
    }


def _detect_questions_from_boxes(
    printed_boxes: list[dict], img_h: int, page_idx: int
) -> list[dict]:
    """从印刷体区域中推断题目序号（基于 y 位置）。

    由于分类阶段尚未做 OCR，无法直接获取文本内容。
    这里采用兼容策略：
      1. 按 y 排序印刷体区域
      2. 使用相邻区域 y 间隔划分题目区间
      3. 后续由 handwriting_recognizer 填充 marker_text

    返回:
        [{"q_idx": 0, "y_start": 100, "y_end": 300,
          "marker_bbox": (x0,y0,x2,y2), "marker_text": ""}, ...]
    """
    if not printed_boxes:
        return []

    # 按 y 排序
    sorted_boxes = sorted(
        printed_boxes, key=lambda r: _bbox_center_y(r["bbox_pixel"])
    )

    # 尝试从匹配题号模式的印刷体区域中识别题目
    markers = []
    for box in sorted_boxes:
        bbox = box["bbox_pixel"]
        cy = _bbox_center_y(bbox)
        text = box.get("text", "")

        m = _QUESTION_PATTERN.search(text)
        if m:
            q_num = int(m.group(1) or m.group(2))
            markers.append({
                "q_idx": q_num - 1,  # 0-based
                "marker_bbox": bbox,
                "marker_text": text,
                "y_center": cy,
            })

    # 如果未从文本中检测到题号，使用位置推断
    if not markers:
        # 按 y 顺序分配题号
        for i, box in enumerate(sorted_boxes):
            bbox = box["bbox_pixel"]
            cy = _bbox_center_y(bbox)
            markers.append({
                "q_idx": i,
                "marker_bbox": bbox,
                "marker_text": f"({i + 1})",
                "y_center": cy,
            })

    # 按 q_idx 去重
    markers_dict = {}
    for m in markers:
        qi = m["q_idx"]
        if qi not in markers_dict or m["y_center"] < markers_dict[qi]["y_center"]:
            markers_dict[qi] = m
    markers = sorted(markers_dict.values(), key=lambda x: x["q_idx"])
    markers.sort(key=lambda x: x["y_center"])

    # 构建 y 区间
    regions = []
    for i, m in enumerate(markers):
        y_start = m["y_center"]
        if i + 1 < len(markers):
            y_end = markers[i + 1]["y_center"]
        else:
            y_end = img_h
        regions.append({
            "q_idx": m["q_idx"],
            "y_start": y_start,
            "y_end": y_end,
            "marker_bbox": m["marker_bbox"],
            "marker_text": m["marker_text"],
        })

    logger.debug(
        "第 %d 页: 推断 %d 道题: %s",
        page_idx + 1,
        len(regions),
        [r["q_idx"] + 1 for r in regions],
    )
    return regions


def _assign_to_handwriting(
    handwriting_boxes: list[dict],
    question_regions: list[dict],
    page_idx: int,
) -> None:
    """为手写区域分配题号（原地修改）。"""
    for box in handwriting_boxes:
        cy = _bbox_center_y(box["bbox_pixel"])
        matched = None
        for r in question_regions:
            if r["y_start"] <= cy < r["y_end"]:
                matched = r["q_idx"]
                break
        box["question_idx"] = matched


# ============================================================
#  向后兼容接口 — 供 processor.py 过渡期使用
# ============================================================

def extract_student_answers(
    ocr_results: list[dict],
    raw_lines: list[dict] | None = None,
    img_w: int = 0,
    img_h: int = 0,
) -> tuple[list[dict], dict[int, list[dict]]]:
    """兼容旧接口 — 返回空列表，提醒使用新流水线。"""
    logger.warning(
        "extract_student_answers() 已被新三阶段流水线取代。"
        "请使用: detect_text() → analyze_layout() → recognize_handwriting()"
    )
    return [], {}