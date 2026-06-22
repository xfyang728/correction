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

_QUESTION_PATTERN = re.compile(
    r'[（(](\d+)[)）]'      # (1) / （1）— 子题号
    r'|[①②③④⑤⑥⑦⑧⑨⑩]'  # 带圈数字
    r'|(?:^|\s)(\d+)\.(?!\d)'  # 1. / 2. — 大题号
    r'|(?:^|\s)(\d+)、'     # 1、/ 2、
)

# 不同题号格式的 q_idx 偏移量，避免冲突
_QIDX_OFFSET_PAREN = 0     # (1) → 0  范围 0~99
_QIDX_OFFSET_DOT = 100     # 1.  → 100 范围 100~199
_QIDX_OFFSET_CIRCLE = 200  # ①   → 200 范围 200~209

# 手写/印刷判别阈值（200 DPI 下）
HANDWRITTEN_HEIGHT_MIN = 90   # 手写字高度 ≥ 90px (~12.5pt)，低于此值视为印刷体
PINYIN_HEIGHT_MAX = 55        # 拼音提示高度 ≤ 55px

# 带圈数字 → 数值映射
_CIRCLED_DIGITS = {
    '①': 1, '②': 2, '③': 3, '④': 4, '⑤': 5,
    '⑥': 6, '⑦': 7, '⑧': 8, '⑨': 9, '⑩': 10,
}


def _question_match(text: str) -> int | None:
    """从文本中提取题号，支持多种格式并分配唯一 q_idx。

    不同格式使用不同偏移量，避免 (1) 与 ① 等冲突：
      - (1)/(2)/... → q_idx = digit - 1 + _QIDX_OFFSET_PAREN (0~99)
      - 1./2./...   → q_idx = digit - 1 + _QIDX_OFFSET_DOT (100~199)
      - ①/②/...    → q_idx = circled_val - 1 + _QIDX_OFFSET_CIRCLE (200~209)

    返回 0-based q_idx，未匹配返回 None。
    """
    m = _QUESTION_PATTERN.search(text)
    if not m:
        return None

    # (1) 或 （1）— group 1
    if m.lastindex and m.group(1):
        return int(m.group(1)) - 1 + _QIDX_OFFSET_PAREN

    # 1. / 1、 — group 2 (dot) 或 group 3 (顿号)
    if m.lastindex and m.group(2):
        return int(m.group(2)) - 1 + _QIDX_OFFSET_DOT
    if m.lastindex and m.group(3):
        return int(m.group(3)) - 1 + _QIDX_OFFSET_DOT

    # 带圈数字 — q_idx = circled_val - 1 + _QIDX_OFFSET_CIRCLE
    matched = m.group(0)
    if matched in _CIRCLED_DIGITS:
        return _CIRCLED_DIGITS[matched] - 1 + _QIDX_OFFSET_CIRCLE
    return None


def _bbox_h(bbox: tuple) -> int:
    """返回 bbox 高度。"""
    return bbox[3] - bbox[1]


def _bbox_center_y(bbox: tuple) -> float:
    """返回 bbox 中心 y。"""
    return (bbox[1] + bbox[3]) / 2.0


def analyze_layout(
    det_boxes: list[dict], img_h: int, page_idx: int = 0,
    ocr_records: list[dict] | None = None,
) -> dict:
    """基于文本检测框进行版面分析，返回分类后的区域信息。

    参数:
        det_boxes: text_detector.detect_text() 的输出，每项含：
            - bbox_pixel: (x0,y0,x2,y2) 外接矩形
            - confidence: 检测置信度
            - page: 页码
        img_h: 原图高度（用于题号区间推断）
        page_idx: 页码
        ocr_records: text_detector 返回的全图识别记录，用于题号检测

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

    # ---- 题目序号检测（从 OCR 记录中匹配文本） ----
    question_regions = _detect_questions_from_boxes(
        printed_boxes, img_h, page_idx, ocr_records=ocr_records,
    )

    # ---- 将题号分配到手写答案 ----
    _assign_to_handwriting(handwriting_boxes, question_regions, page_idx)

    return {
        "handwriting_boxes": handwriting_boxes,
        "printed_boxes": printed_boxes,
        "all_regions": all_regions,
        "question_regions": {page_idx: question_regions} if question_regions else {},
    }


# 列检测阈值（200 DPI 下，两栏间距通常 > 150px）
_COLUMN_GAP_THRESHOLD = 150


def _bbox_center_x(bbox: tuple) -> float:
    """返回 bbox 中心 x。"""
    return (bbox[0] + bbox[2]) / 2.0


def _detect_questions_from_boxes(
    printed_boxes: list[dict], img_h: int, page_idx: int,
    ocr_records: list[dict] | None = None,
) -> list[dict]:
    """从 OCR 识别文本中检测题目序号，构建 y 区间。

    从所有 OCR 记录（不限于印刷体框）中匹配题号模式，
    使用列感知（column-aware）方法构建 y 区间：
    - 先根据 marker 的 x 坐标间隙检测列（双栏/单栏）
    - 每列内 marker 按 y 排序分别构建区间
    - 区间边界稳定，不受跨列干扰

    返回:
        [{"q_idx": 0, "y_start": 100, "y_end": 300,
          "marker_bbox": (x0,y0,x2,y2), "marker_text": "(1)",
          "column": 0}, ...]
    """
    if not ocr_records:
        logger.warning("第 %d 页: 无 OCR 记录，无法检测题号", page_idx + 1)
        return []

    # ---- 从 OCR 记录中提取题号标记 ----
    markers: list[dict] = []
    for rec in ocr_records:
        text = rec.get("rec_text", "")
        q = _question_match(text)
        if q is not None:
            bbox = rec.get("rec_bbox")
            if not bbox:
                continue
            cy = _bbox_center_y(bbox)
            cx = _bbox_center_x(bbox)
            markers.append({
                "q_idx": q,
                "marker_bbox": bbox,
                "marker_text": text,
                "y_center": cy,
                "x_center": cx,
            })

    if not markers:
        logger.warning("第 %d 页: 未从 OCR 文本中检测到题号，回退到位置推断", page_idx + 1)
        # 回退：按印刷体框 y 顺序分配题号
        sorted_boxes = sorted(
            printed_boxes, key=lambda r: _bbox_center_y(r["bbox_pixel"])
        )
        for i, box in enumerate(sorted_boxes):
            bbox = box["bbox_pixel"]
            cy = _bbox_center_y(bbox)
            markers.append({
                "q_idx": i,
                "marker_bbox": bbox,
                "marker_text": f"({i + 1})",
                "y_center": cy,
                "x_center": _bbox_center_x(bbox),
            })
    else:
        logger.info(
            "第 %d 页: OCR 文本中检测到 %d 个题号标记",
            page_idx + 1,
            len(markers),
        )

    # ---- 按 q_idx 去重（同题号取 y 最小的） ----
    markers_dict: dict[int, dict] = {}
    for m in markers:
        qi = m["q_idx"]
        if qi not in markers_dict or m["y_center"] < markers_dict[qi]["y_center"]:
            markers_dict[qi] = m

    unique_markers = list(markers_dict.values())

    # ---- 列检测：按 x_center 间隙分列 ----
    columns = _split_into_columns(unique_markers)

    # ---- 每列内按 y 排序构建区间 ----
    regions = _build_column_intervals(columns, img_h)

    logger.debug(
        "第 %d 页: 推断 %d 道题（%d 列）",
        page_idx + 1,
        len(regions),
        len(columns),
    )
    return regions


def _split_into_columns(markers: list[dict]) -> list[list[dict]]:
    """将 markers 分为最多 2 列（双栏/单栏检测）。

    1. 按 x_center 排序
    2. 找出相邻 marker 间最大间隙
    3. 若最大间隙 > _COLUMN_GAP_THRESHOLD，在此处拆分为左右两列
    4. 最多拆 2 列，避免中间宽度框导致过度拆分

    返回:
        [[col0_markers...], [col1_markers...]]  # 最多 2 组
    """
    if len(markers) <= 1:
        return [markers]

    sorted_by_x = sorted(markers, key=lambda m: m["x_center"])

    # 找出所有间隙
    gaps: list[tuple[float, int]] = []  # (gap_size, index_in_sorted)
    for i in range(len(sorted_by_x) - 1):
        gap = sorted_by_x[i + 1]["x_center"] - sorted_by_x[i]["x_center"]
        gaps.append((gap, i))

    # 找最大间隙
    max_gap = max(gaps, key=lambda g: g[0])

    if max_gap[0] <= _COLUMN_GAP_THRESHOLD:
        return [markers]

    # 仅在最大间隙处拆分为左右两列（最多 2 列）
    split_idx = max_gap[1]
    return [
        sorted_by_x[: split_idx + 1],
        sorted_by_x[split_idx + 1:],
    ]


def _build_column_intervals(
    columns: list[list[dict]], img_h: int,
) -> list[dict]:
    """为每列 markers 构建 y 区间。

    每列内 markers 按 y 排序，相邻 marker 间形成区间。
    列间区间独立，互不干扰。

    返回:
        [{"q_idx": ..., "y_start": ..., "y_end": ...,
          "marker_bbox": ..., "marker_text": ..., "column": ...}, ...]
    """
    regions: list[dict] = []
    for col_idx, col_markers in enumerate(columns):
        sorted_by_y = sorted(col_markers, key=lambda m: m["y_center"])
        for i, m in enumerate(sorted_by_y):
            y_start = m["y_center"]
            y_end = (
                sorted_by_y[i + 1]["y_center"]
                if i + 1 < len(sorted_by_y)
                else img_h
            )
            regions.append({
                "q_idx": m["q_idx"],
                "y_start": y_start,
                "y_end": y_end,
                "marker_bbox": m["marker_bbox"],
                "marker_text": m["marker_text"],
                "column": col_idx,
            })
    return regions


def _assign_to_handwriting(
    handwriting_boxes: list[dict],
    question_regions: list[dict],
    page_idx: int,
) -> None:
    """为手写区域分配题号（原地修改）。

    列感知分配策略（三阶）：
    1. 先根据手写框 x_center 确定所属列，在该列内匹配 y 区间
    2. 若列内匹配成功，检查是否在 y 方向极靠近不同列的题号 marker
       （处理跨列书写，如竖排/双栏布局中学生答案写在另一列的情况）
    3. 若前两步失败，回退到全局 y 区间匹配（按 y_start 降序）
    """
    # ---- 预计算所有 marker 的 y_center ----
    # 用于策略2的跨列 y 距离比较
    marker_y_map: dict[int, float] = {}  # q_idx -> y_center of marker
    for r in question_regions:
        qi = r["q_idx"]
        if qi not in marker_y_map:
            marker_y_map[qi] = _bbox_center_y(r["marker_bbox"])
        else:
            marker_y_map[qi] = min(marker_y_map[qi], _bbox_center_y(r["marker_bbox"]))

    for box in handwriting_boxes:
        bbox = box["bbox_pixel"]
        cx = _bbox_center_x(bbox)
        cy = _bbox_center_y(bbox)
        matched = None

        # ---- 策略1：列感知匹配 ----
        col_candidates: dict[int, list[dict]] = {}
        for r in question_regions:
            col = r.get("column", 0)
            col_candidates.setdefault(col, []).append(r)

        best_col = _find_column_for_box(cx, col_candidates, question_regions)
        matched_col: int | None = None  # 记录列内匹配的题号

        if best_col is not None:
            col_regions = col_candidates[best_col]
            for r in sorted(col_regions, key=lambda x: x["y_start"]):
                if r["y_start"] <= cy < r["y_end"]:
                    matched = r["q_idx"]
                    matched_col = best_col
                    break

        # ---- 策略2：跨列 y 距离极近纠正 ----
        # 当列内匹配成功，但手写框在 y 方向上极靠近不同列的题号 marker 时，
        # 优先使用该 marker 的题号（处理双栏中学生答案跨列书写的情况）
        if matched is not None and len(col_candidates) > 1:
            _Y_PROXIMITY_THRESHOLD = 30  # 200 DPI 下 ~4pt
            for qi, my in marker_y_map.items():
                if qi == matched:
                    continue
                dy = abs(cy - my)
                if dy < _Y_PROXIMITY_THRESHOLD:
                    # 找到 y 方向更近的题号 → 优先使用
                    current_dy = abs(cy - marker_y_map.get(matched, cy))
                    if dy < current_dy:
                        logger.debug(
                            "跨列纠正: cy=%.1f 从 q_idx=%d(dy=%.1f) 改为 q_idx=%d(dy=%.1f)",
                            cy, matched, current_dy, qi, dy,
                        )
                        matched = qi
                        break

        # ---- 策略3：回退到全局 y 区间匹配 ----
        if matched is None:
            for r in sorted(question_regions, key=lambda x: x["y_start"], reverse=True):
                if r["y_start"] <= cy < r["y_end"]:
                    matched = r["q_idx"]
                    break

        box["question_idx"] = matched


def _find_column_for_box(
    cx: float,
    col_candidates: dict[int, list[dict]],
    all_regions: list[dict],
) -> int | None:
    """确定手写框 cx 所属的列。

    策略：
    1. 如果 cx 落在某列 marker x 范围内，优先使用该列
    2. 否则选择最近的列，但距离 > 300px 时返回 None（触发回退）

    返回 column_idx，None 表示无法确定（将触发全局 y 回退）。
    """
    if len(col_candidates) <= 1:
        return 0 if col_candidates else None

    # 计算每列的 x 范围和平均值
    col_x_ranges: dict[int, tuple[float, float]] = {}
    col_x_avgs: dict[int, float] = {}

    for col_idx, regions in col_candidates.items():
        x_centers = [_bbox_center_x(r["marker_bbox"]) for r in regions]
        if not x_centers:
            continue
        col_x_ranges[col_idx] = (min(x_centers), max(x_centers))
        col_x_avgs[col_idx] = sum(x_centers) / len(x_centers)

    # 策略1：cx 在列 x 范围内 → 明确归属
    for col_idx, (x_min, x_max) in col_x_ranges.items():
        if x_min <= cx <= x_max:
            return col_idx

    # 策略2：不在任何列 x 范围内 → 选最近的，但距离太大则返回 None
    best_col = None
    best_dist = float("inf")
    for col_idx, avg_x in col_x_avgs.items():
        dist = abs(cx - avg_x)
        if dist < best_dist:
            best_dist = dist
            best_col = col_idx

    # 距最近列仍 > 300px → 不确定性高，触发 y-only 回退
    if best_dist > 300:
        return None

    return best_col


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