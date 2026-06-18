"""
版面分析 — 对 OCR 结果进行行检测、元素分类、手写答案提取、题目检测与分组。

核心功能：
    1. 将 OCR 检测项按 y 坐标聚类为行
    2. 对每行区分拼音提示（印刷）与手写答案（手写）
    3. 提取学生手写答案，按阅读顺序排序
    4. 检测题目序号（(1)(2)(3)），按题目对手写答案分组
"""

import logging
import re

logger = logging.getLogger("layout_analyzer")

# 题目序号模式： (1) 或 1. 或 1、
_QUESTION_PATTERN = re.compile(r'^\((\d+)\)|^(\d+)[.、]')


def cluster_by_row(ocr_results: list[dict], img_h: int) -> list[list[dict]]:
    """将 OCR 结果按 y 坐标聚类为行。"""
    if not ocr_results:
        return []

    # 按 page 分组
    pages: dict[int, list[dict]] = {}
    for r in ocr_results:
        pages.setdefault(r.get("page", 0), []).append(r)

    rows: list[list[dict]] = []

    for page_idx in sorted(pages.keys()):
        items = sorted(pages[page_idx], key=lambda x: _center_y(x))
        if len(items) < 2:
            rows.append(items)
            continue

        ys = [_center_y(it) for it in items]
        gaps = sorted([ys[i+1] - ys[i] for i in range(len(ys) - 1)])
        median_gap = gaps[len(gaps) // 2]
        row_gap = max(median_gap * 2.0, 30)

        current_row = [items[0]]
        for item in items[1:]:
            prev_y = _center_y(current_row[-1])
            curr_y = _center_y(item)
            if curr_y - prev_y < row_gap:
                current_row.append(item)
            else:
                rows.append(current_row)
                current_row = [item]
        if current_row:
            rows.append(current_row)

    return rows


def _center_y(item: dict) -> float:
    """返回项的中心 y 坐标。"""
    bbox = item.get("bbox_pixel", (0, 0, 0, 0))
    return (bbox[1] + bbox[3]) / 2.0


def _center_x(item: dict) -> float:
    """返回项的中心 x 坐标。"""
    bbox = item.get("bbox_pixel", (0, 0, 0, 0))
    return (bbox[0] + bbox[2]) / 2.0


def _is_chinese_char(char: str) -> bool:
    """判断是否包含中文字符。"""
    if not char:
        return False
    for c in char:
        if '\u4e00' <= c <= '\u9fff' or '\u3000' <= c <= '\u303f':
            return True
    return False


def _is_pure_chinese(text: str) -> bool:
    """判断是否为纯中文文本（只含汉字，不含字母、数字、标点）。"""
    if not text:
        return False
    for c in text:
        if not ('\u4e00' <= c <= '\u9fff'):
            return False
    return True


def _item_height(item: dict) -> float:
    """返回项的 bbox 高度。"""
    bbox = item.get("bbox_pixel", (0, 0, 0, 0))
    return bbox[3] - bbox[1]


def classify_item(item: dict) -> str:
    """根据 bbox 高度和内容判断元素类型。"""
    text = item.get("char", "")
    h = _item_height(item)

    if _is_chinese_char(text):
        if h >= 70:
            return "handwritten"
        else:
            return "pinyin"

    conf = item.get("confidence", 0)
    if conf > 0.85:
        return "pinyin"
    if h >= 60:
        return "handwritten"
    return "pinyin"


# ============================================================
#  题目检测与分组（新增）
# ============================================================

def detect_question_regions(raw_lines: list[dict], img_h: int) -> dict[int, list[dict]]:
    """从行级 OCR 数据中检测题目序号，返回每页的题目区域。

    扫描 raw_lines 中每行的 text，匹配题目序号模式（如 (1)、1.、1、），
    将匹配行标记为题号行，以其 y 位置划分题目区域。

    参数:
        raw_lines: OCR 行级原始数据（含非中文字符）
        img_h: 图片高度（像素），用于确定最后一道题的底部

    返回:
        {page_idx: [
            {"q_idx": 0, "y_start": 100, "y_end": 300,
             "marker_bbox": (x0,y0,x2,y2), "marker_text": "(1)"},
            ...
        ]}
    """
    if not raw_lines:
        return {}

    # 按 page 分组
    page_lines: dict[int, list[dict]] = {}
    for r in raw_lines:
        page_lines.setdefault(r["page"], []).append(r)

    result: dict[int, list[dict]] = {}

    for page_idx in sorted(page_lines.keys()):
        lines = page_lines[page_idx]
        # 按 y 排序
        lines.sort(key=lambda x: _center_y(x))

        # 匹配题号
        markers = []
        for line in lines:
            bbox = line.get("bbox_pixel", (0, 0, 0, 0))
            text = line.get("text", "").strip()
            m = _QUESTION_PATTERN.search(text)
            if m:
                q_num = int(m.group(1) or m.group(2))
                markers.append({
                    "q_idx": q_num - 1,  # 转为 0-based
                    "marker_bbox": bbox,
                    "marker_text": text,
                    "y_center": _center_y(line),
                })

        if not markers:
            # 该页未检测到题号
            continue

        # 按 q_idx 去重（同一题号可能有多个 OCR 项），取 y 最小的
        markers_dict = {}
        for m in markers:
            qi = m["q_idx"]
            if qi not in markers_dict or m["y_center"] < markers_dict[qi]["y_center"]:
                markers_dict[qi] = m
        markers = sorted(markers_dict.values(), key=lambda x: x["q_idx"])

        # 按 y 排序（物理位置顺序）
        markers.sort(key=lambda x: x["y_center"])

        # 构建题目区域
        regions = []
        for i, m in enumerate(markers):
            y_start = m["y_center"]
            if i + 1 < len(markers):
                y_end = markers[i + 1]["y_center"]
            else:
                y_end = img_h  # 最后一道题到页底
            regions.append({
                "q_idx": m["q_idx"],
                "y_start": y_start,
                "y_end": y_end,
                "marker_bbox": m["marker_bbox"],
                "marker_text": m["marker_text"],
            })

        result[page_idx] = regions
        logger.debug("第 %d 页: 检测到 %d 道题: %s",
                     page_idx + 1, len(regions),
                     [r["q_idx"] + 1 for r in regions])

    return result


def assign_to_questions(handwritten_items: list[dict],
                        question_regions: dict[int, list[dict]]) -> None:
    """为每个手写项赋予题目索引（原地修改）。

    根据 handwritten_items 中每项的 bbox 中心 y 坐标，
    找到其所在题目区域，设置 item["question_idx"]。

    参数:
        handwritten_items: 手写答案列表，每项需含 "page" 和 "bbox_pixel"
        question_regions: detect_question_regions() 的输出
    """
    for item in handwritten_items:
        page_idx = item.get("page", 0)
        regions = question_regions.get(page_idx, [])
        if not regions:
            item["question_idx"] = None
            continue

        cy = _center_y(item)
        matched = None
        for r in regions:
            if r["y_start"] <= cy <= r["y_end"]:
                matched = r["q_idx"]
                break

        item["question_idx"] = matched


# ============================================================
#  主入口
# ============================================================

def extract_student_answers(ocr_results: list[dict],
                            raw_lines: list[dict] | None = None,
                            img_w: int = 0, img_h: int = 0
                            ) -> tuple[list[dict], dict[int, list[dict]]]:
    """从 OCR 结果中提取学生手写答案，并按题目分组。

    策略：
        - PaddleOCR 对印刷拼音和学生手写汉字都能识别
        - 过滤后保留纯汉字项，按行内 x 坐标排序
        - 多字项拆分为单字项
        - 通过 raw_lines 检测题目序号，对手写答案进行分组

    参数:
        ocr_results: OCR 逐字结果列表
        raw_lines: OCR 行级原始数据（用于题目检测）
        img_w: 图片宽度（像素）
        img_h: 图片高度（像素）

    返回:
        (sorted_items, question_regions)

        sorted_items: 按 (page, y, x) 排序的手写答案列表，
                      每项增加 "question_idx" 字段（None=未匹配到题号）
        question_regions: detect_question_regions() 的原始输出，
                          用于 annotator 定位题号标记位置
    """
    # 1. 过滤出纯汉字项
    chinese_items: list[dict] = []
    for item in ocr_results:
        text = item.get("char", "").strip()
        if not text:
            continue
        if _is_pure_chinese(text):
            chinese_items.append(item)

    # 2. 拆分多字项
    expanded: list[dict] = []
    for item in chinese_items:
        text = item.get("char", "")
        if len(text) == 1:
            expanded.append(item)
        else:
            bbox = item["bbox_pixel"]
            x0, y0, x2, y2 = bbox
            char_w = (x2 - x0) / len(text)
            for i, ch in enumerate(text):
                expanded.append({
                    "page": item["page"],
                    "bbox_pixel": (int(x0 + i * char_w), y0,
                                   int(x0 + (i + 1) * char_w), y2),
                    "char": ch,
                    "confidence": item["confidence"],
                    "img_pixel_w": item["img_pixel_w"],
                    "img_pixel_h": item["img_pixel_h"],
                })

    # 3. 排序
    expanded.sort(key=lambda r: r.get("bbox_pixel", (0,))[0])
    expanded.sort(key=lambda r: _center_y(r))
    rows = cluster_by_row(expanded, img_h)
    final: list[dict] = []
    for row in rows:
        row.sort(key=lambda r: r.get("bbox_pixel", (0,))[0])
        final.extend(row)

    # 4. 检测题目区域并分配题号
    question_regions = detect_question_regions(raw_lines or [], img_h)
    assign_to_questions(final, question_regions)

    # 统计
    q_assigned = sum(1 for r in final if r.get("question_idx") is not None)
    logger.info("答案提取: %d 字, 其中 %d 字已分配题号 (%d 页有题号检测)",
                len(final), q_assigned, len(question_regions))
    return final, question_regions