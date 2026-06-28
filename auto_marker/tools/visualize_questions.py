"""
可视化识别结果 — 在 PDF 渲染图上用淡灰色半透明框标出每道题，左上角标注题号。

用法:
    python visualize_questions.py [PDF文件名]

默认处理 301_2026-06-18_003.pdf，输出到 data/output/ 目录。
"""

import logging
import re
import sys
from pathlib import Path

import cv2
import fitz
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# 确保项目根在 sys.path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("visualize")

OCR_DPI = 200

# 半透明矩形颜色 (BGR)
_OVERLAY_COLOR = (200, 200, 200)
_OVERLAY_ALPHA = 0.25

# 题号标签样式
_LABEL_BG_COLOR = (40, 40, 40)  # 深灰色背景
_LABEL_TEXT_COLOR = (255, 255, 255)  # 白色文字
_LABEL_FONT_SIZE = 28
_LABEL_PADDING = 6


def _render_pdf_page(pdf_path: str, page_idx: int = 0) -> tuple[np.ndarray, fitz.Page]:
    """渲染 PDF 页面为 RGB numpy array（200 DPI）。"""
    doc = fitz.open(pdf_path)
    page = doc[page_idx]
    zoom = OCR_DPI / 72
    pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), colorspace=fitz.csRGB)
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)
    doc.close()
    return img, page


def _get_question_regions(pdf_path: str) -> list[dict]:
    """运行流水线获取题目区域信息。"""
    from services.pipeline_service import PipelineService

    svc = PipelineService()
    _, question_regions = svc._run_ocr_pipeline(Path(pdf_path))

    # 取第一页的题目区域
    regions = question_regions.get(0, [])
    if not regions:
        logger.warning("未检测到题目区域")
    else:
        logger.info("检测到 %d 道题目", len(regions))

    return regions


def _is_clean_main_marker(marker_text: str) -> bool:
    """判断 marker_text 是否是干净的大题题号。

    有效的大题题号格式：
      - "1." / "2." / "3." （纯题号）
      - "1.(8分)" / "2.(2分)" （题号+分值，分值后无内容）
      - "1.（8分）" （中文括号）

    排除 OCR 误识别的印刷体文本：
      - "3. (2 分).D" （分值后有额外字符）
      - "16. (2 分)A B" （题号过大或含选项）
      - "2.(2分) ①(zhì p) 质朴" （分值后有题目内容）
    """
    import re
    if not marker_text:
        return False
    text = marker_text.strip()
    # 匹配 "数字." 或 "数字、" 开头
    m = re.match(r'^(\d+)[.、]\s*', text)
    if not m:
        return False
    num = int(m.group(1))
    # 题号应该较小（1-10）
    if num < 1 or num > 10:
        return False
    # 获取题号后的剩余文本
    rest = text[m.end():]
    # 情况1: 纯题号，后面无内容或只有空白
    if not rest or rest.isspace():
        return True
    # 情况2: 题号后紧跟分值 "(X分)" 或 "（X分）"，之后必须无内容
    score_match = re.match(r'^[（(]\s*\d+\s*[分分]\s*[)）]\s*$', rest)
    if score_match:
        return True
    return False


def _get_main_question_regions(regions: list[dict], img_h: int) -> list[dict]:
    """从所有题目区域中提取大题，合并小题到对应大题的 y 区间内。

    q_idx 偏移量规则（layout_analyzer）:
      - (1)/(2)/... → 0~99    （小题）
      - 1./2./...   → 100~199 （大题）
      - ①/②/...    → 200~209 （带圈数字，视为小题）

    返回按 (column, y_start) 排序的大题列表，每项含:
      - q_idx, y_start, y_end, marker_bbox, marker_text, column, display_num
    """
    # 分离大题和小题
    # 大题：q_idx 在 100-109 范围内（题号 1-10）
    # 注意：可能包含 OCR 误识别（如 "3. (2 分).D"），但可视化仍可接受
    main_qs = [
        r for r in regions
        if 100 <= r["q_idx"] <= 109
    ]
    sub_qs = [r for r in regions if r["q_idx"] < 200]  # 包含小题和带圈数字

    if not main_qs:
        # 无大题标记，回退：所有区域按 y 排序，每个作为独立题
        logger.warning("未检测到大题题号（1. 2. 格式），回退到全部区域")
        return sorted(regions, key=lambda r: (r.get("column", 0), r["y_start"]))

    # 按 (column, y_start) 排序大题
    main_qs.sort(key=lambda r: (r.get("column", 0), r["y_start"]))

    # 为每个大题计算 y 区间（延伸到下一个大题的 y_start）
    result = []
    for i, mq in enumerate(main_qs):
        y_start = mq["y_start"]
        if i + 1 < len(main_qs):
            next_mq = main_qs[i + 1]
            # 同列才用下一个大题的 y_start 作为边界
            if next_mq.get("column", 0) == mq.get("column", 0):
                y_end = next_mq["y_start"]
            else:
                y_end = img_h
        else:
            y_end = img_h

        # 收集该大题 y 区间内的小题，合并 marker_bbox 的 x 范围
        sub_in_range = [
            s for s in sub_qs
            if y_start <= s["y_start"] < y_end
            and s.get("column", 0) == mq.get("column", 0)
        ]

        # 合并 x 范围：取大题 marker 和所有小题 marker 的 x 并集
        all_bboxes = [mq["marker_bbox"]] + [s["marker_bbox"] for s in sub_in_range]
        merged_x0 = min(b[0] for b in all_bboxes)
        merged_x2 = max(b[2] for b in all_bboxes)

        result.append({
            "q_idx": mq["q_idx"],
            "y_start": y_start,
            "y_end": y_end,
            "marker_bbox": (merged_x0, mq["marker_bbox"][1], merged_x2, mq["marker_bbox"][3]),
            "marker_text": mq.get("marker_text", ""),
            "column": mq.get("column", 0),
            "sub_count": len(sub_in_range),
        })

    # 分配显示题号（按阅读顺序）
    for i, r in enumerate(result, start=1):
        r["display_num"] = i

    logger.info(
        "大题 %d 道（含小题合并）: %s",
        len(result),
        [f"{r['display_num']}. {r['marker_text']!r} (+{r['sub_count']}小题)" for r in result],
    )
    return result


def _draw_overlay(img: np.ndarray, regions: list[dict], page_width: int) -> np.ndarray:
    """在图片上绘制半透明矩形框和题号标签。"""
    # 复制底图用于绘制
    result = img.copy()

    # 页面边距
    margin_x = 20

    for region in regions:
        y_start = int(region["y_start"])
        y_end = int(region["y_end"])
        # 限制 y_end 不超过图片高度
        y_end = min(y_end, img.shape[0])

        # x 范围：全页宽度减边距
        x_start = margin_x
        x_end = page_width - margin_x

        # 绘制半透明矩形（使用 alpha 混合）
        overlay = result.copy()
        cv2.rectangle(overlay, (x_start, y_start), (x_end, y_end), _OVERLAY_COLOR, -1)
        cv2.addWeighted(overlay, _OVERLAY_ALPHA, result, 1 - _OVERLAY_ALPHA, 0, result)

        # 绘制题号标签（左上角）
        display_num = region.get("display_num", 1)
        label_text = str(display_num)
        # 估算标签尺寸
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = _LABEL_FONT_SIZE / 30
        (text_w, text_h), baseline = cv2.getTextSize(label_text, font, font_scale, 2)
        label_w = text_w + _LABEL_PADDING * 2
        label_h = text_h + _LABEL_PADDING * 2

        # 标签位置：矩形框左上角内侧
        label_x = x_start + 5
        label_y = y_start + 5

        # 绘制标签背景（圆角矩形效果用普通矩形近似）
        cv2.rectangle(result, (label_x, label_y), (label_x + label_w, label_y + label_h), _LABEL_BG_COLOR, -1)

        # 绘制题号文字
        cv2.putText(result, label_text, (label_x + _LABEL_PADDING, label_y + label_h - baseline - 2),
                    font, font_scale, _LABEL_TEXT_COLOR, 2, cv2.LINE_AA)

    return result


def visualize(pdf_path: str, output_dir: str | None = None) -> str:
    """生成可视化图片。

    参数:
        pdf_path: PDF 文件路径
        output_dir: 输出目录，默认 data/output/

    返回:
        输出图片路径
    """
    src = Path(pdf_path)
    out_dir = Path(output_dir) if output_dir else ROOT / "data" / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    output_path = str(out_dir / f"{src.stem}_visualized.png")

    logger.info("渲染 PDF 页面: %s", src.name)
    img, page = _render_pdf_page(pdf_path)
    logger.info("页面尺寸: %dx%d", img.shape[1], img.shape[0])

    logger.info("获取题目区域...")
    regions = _get_question_regions(pdf_path)

    if not regions:
        logger.warning("无题目区域，输出原图")
        cv2.imwrite(output_path, img)
        return output_path

    # 提取大题并合并小题
    logger.info("提取大题区域...")
    main_regions = _get_main_question_regions(regions, img.shape[0])

    if not main_regions:
        logger.warning("无大题区域，输出原图")
        cv2.imwrite(output_path, img)
        return output_path

    # 绘制叠加层
    logger.info("绘制题目区域框...")
    result = _draw_overlay(img, main_regions, img.shape[1])

    # 保存
    cv2.imwrite(output_path, result)
    logger.info("可视化图片已保存: %s", output_path)

    return output_path


def main():
    pdf_name = sys.argv[1] if len(sys.argv) > 1 else "301_2026-06-18_003.pdf"

    # 检查文件是否存在
    pdf_path = ROOT / "data" / "test_samples" / pdf_name
    if not pdf_path.exists():
        # 尝试从 data/incoming/ 查找
        incoming = ROOT / "data" / "incoming" / pdf_name
        if incoming.exists():
            pdf_path = incoming
        else:
            print(f" 文件不存在: {pdf_name}")
            print(f"    searched: {ROOT / 'data' / 'test_samples' / pdf_name}")
            print(f"   searched: {incoming}")
            sys.exit(1)

    print(f"\n{'='*50}")
    print(f"🎨 可视化识别结果: {pdf_path.name}")
    print(f"{'='*50}\n")

    output = visualize(str(pdf_path))

    print(f"\n{'='*50}")
    print(f"✅ 完成: {output}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
