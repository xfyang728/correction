"""
版面分析 — 对 OCR 结果进行行检测、元素分类、手写答案提取。

核心功能：
    1. 将 OCR 检测项按 y 坐标聚类为行
    2. 对每行区分拼音提示（印刷）与手写答案（手写）
    3. 提取学生手写答案，按阅读顺序排序
"""

import logging
import re

logger = logging.getLogger("layout_analyzer")


def cluster_by_row(ocr_results: list[dict], img_h: int) -> list[list[dict]]:
    """将 OCR 结果按 y 坐标聚类为行。

    参数:
        ocr_results: OCR 结果列表
        img_h: 图片高度（像素）

    返回:
        行列表，每行是 OCR 项列表
    """
    if not ocr_results:
        return []

    # 按 page 分组
    pages: dict[int, list[dict]] = {}
    for r in ocr_results:
        pages.setdefault(r.get("page", 0), []).append(r)

    rows: list[list[dict]] = []
    # 行间距阈值：取所有相邻项 y 差距的中位数 * 1.5
    # 这样能自适应不同 DPI 和排版

    for page_idx in sorted(pages.keys()):
        items = sorted(pages[page_idx], key=lambda x: _center_y(x))

        # 计算相邻项 y 差距，取中位数作为行间距参考
        ys = [_center_y(it) for it in items]
        if len(ys) < 2:
            rows.append(items)
            continue

        gaps = [ys[i+1] - ys[i] for i in range(len(ys)-1)]
        gaps.sort()
        median_gap = gaps[len(gaps) // 2]

        # 行间距阈值 = 中位数差距 * 2（同一行内项间距小，跨行差距大）
        # 但至少 30px（避免 DPI 太低时失效）
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


def _is_chinese_char(char: str) -> bool:
    """判断是否包含中文字符。"""
    if not char:
        return False
    for c in char:
        if '\u4e00' <= c <= '\u9fff' or '\u3000' <= c <= '\u303f':
            return True
    return False


def _is_pure_chinese(text: str) -> bool:
    """判断是否为纯中文文本（只含汉字，不含字母、数字、标点）。

    用于区分学生手写答案（纯汉字）和拼音提示（纯字母）。
    """
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
    """根据 bbox 高度和内容判断元素类型。

    返回:
        "handwritten" — 手写答案
        "pinyin" — 拼音提示/印刷体
    """
    # 如果包含中文字符，优先按高度分类
    text = item.get("char", "")
    h = _item_height(item)

    if _is_chinese_char(text):
        # 中文字符区域一般 > 75px（在 200 DPI 下）
        if h >= 70:
            return "handwritten"
        else:
            return "pinyin"

    # 非中文字符（拼音字母、数字等）
    # 置信度高（>0.85）的是印刷提示
    conf = item.get("confidence", 0)
    if conf > 0.85:
        return "pinyin"
    # 低置信度的字母可能是手写
    if h >= 60:
        return "handwritten"
    return "pinyin"


def extract_student_answers(ocr_results: list[dict], img_w: int, img_h: int) -> list[dict]:
    """从 OCR 结果中提取学生手写答案。

    策略：
        PaddleOCR 对印刷拼音和学生手写汉字都能识别，且：
        - 拼音提示 = 纯字母（如 "hu", "lián", "xi"）
        - 学生答案 = 纯汉字（如 "西", "湖", "莲"）
        - 题目文字 = 含标点的混合文本（如 "、看拼音，写词语"）

        过滤后保留纯汉字项，再按行内 x 坐标排序保持阅读顺序。
        多字项（如"特别"）拆分为单字项以匹配逐字比对。

    参数:
        ocr_results: OCR 原始结果列表
        img_w: 图片宽度（像素）
        img_h: 图片高度（像素）

    返回:
        学生答案项列表，按 (page, y, x) 排序
    """
    # 1. 过滤出纯汉字项
    chinese_items: list[dict] = []
    for item in ocr_results:
        text = item.get("char", "").strip()
        if not text:
            continue
        if _is_pure_chinese(text):
            chinese_items.append(item)

    # 2. 拆分多字项（如"特别" → "特" + "别"）
    expanded: list[dict] = []
    for item in chinese_items:
        text = item.get("char", "")
        if len(text) == 1:
            expanded.append(item)
        else:
            # 多字项：按字符宽度均分 bbox
            bbox = item["bbox_pixel"]
            x0, y0, x2, y2 = bbox
            char_w = (x2 - x0) / len(text)
            for i, ch in enumerate(text):
                expanded.append({
                    "page": item["page"],
                    "bbox_pixel": (int(x0 + i * char_w), y0, int(x0 + (i + 1) * char_w), y2),
                    "char": ch,
                    "confidence": item["confidence"],
                    "img_pixel_w": item["img_pixel_w"],
                    "img_pixel_h": item["img_pixel_h"],
                })

    # 3. 排序 — 先按 y 聚类分行，行内按 x 排序
    # 直接按 y 排序会导致同行内 y 微小差异引起交错
    expanded.sort(key=lambda r: r.get("bbox_pixel", (0,))[0])  # 先按 x 排
    expanded.sort(key=lambda r: _center_y(r))  # 再按 y 排（Python sort 稳定）

    # 用聚类分行后重新排序
    rows = cluster_by_row(expanded, img_h)
    final: list[dict] = []
    for row in rows:
        row.sort(key=lambda r: r.get("bbox_pixel", (0,))[0])  # 行内按 x 排
        final.extend(row)

    logger.info("答案提取: 从 %d 项 OCR 结果中提取到 %d 个答案字", len(ocr_results), len(final))
    return final