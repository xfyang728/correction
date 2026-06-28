"""对比黄金标准 PDF 与流水线输出 PDF，量化坐标误差。

输入:
  - data/output/301_2026-06-18_003_golden.pdf (人工标注黄金标准)
  - data/output/301_2026-06-18_003_annotated.pdf (流水线输出)
  - data/test_samples/301_2026-06-18_003_ground_truth.json (gt 坐标)
  - 数据库中最新 task 的 result_json (流水线坐标)

输出:
  - 渲染两个 PDF 的 PNG 用于肉眼对比
  - 每个字符的坐标误差表 (golden vs pipeline)
  - 误差统计: 平均/最大/分位数
"""
import json
import sys
from pathlib import Path

import fitz
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

GOLDEN_PDF = ROOT / "data" / "output" / "301_2026-06-18_003_golden.pdf"
PIPELINE_PDF = ROOT / "data" / "output" / "301_2026-06-18_003_annotated.pdf"
GT_JSON = ROOT / "data" / "test_samples" / "301_2026-06-18_003_ground_truth.json"

RENDER_DPI = 150


def render_pdf_to_png(pdf_path: Path, png_path: Path):
    """渲染 PDF 第 1 页为 PNG。"""
    doc = fitz.open(str(pdf_path))
    page = doc.load_page(0)
    zoom = RENDER_DPI / 72
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat, colorspace=fitz.csRGB)
    img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
    img.save(png_path)
    doc.close()
    print(f"  渲染: {png_path.name} ({img.size[0]}x{img.size[1]})")
    return img


def main():
    print("=== 渲染 PDF 为 PNG ===")
    golden_png = GOLDEN_PDF.with_suffix(".png")
    pipeline_png = PIPELINE_PDF.parent / f"{PIPELINE_PDF.stem}_page1.png"
    render_pdf_to_png(GOLDEN_PDF, golden_png)
    render_pdf_to_png(PIPELINE_PDF, pipeline_png)

    print("\n=== 加载 ground truth 与流水线数据 ===")
    with open(GT_JSON, "r", encoding="utf-8") as f:
        gt_data = json.load(f)
    print(f"  Ground truth: {len(gt_data)} 字")

    from db.crud import get_tasks
    task = get_tasks(limit=1)[0]
    pipeline_data = task.result_json
    print(f"  Pipeline: {len(pipeline_data)} 字 (task #{task.id})")

    # 按 (q_idx, char_idx) 或 (q_idx, char) 匹配
    # gt_data 已有 char_idx; pipeline_data 没有 char_idx, 需按 (q_idx, 出现顺序) 推断
    pipeline_by_q = {}
    for r in pipeline_data:
        qi = r.get("question_idx", -1)
        pipeline_by_q.setdefault(qi, []).append(r)

    # 按 char_idx 排序 (取 bbox 中心 x 升序)
    for qi, items in pipeline_by_q.items():
        items.sort(key=lambda r: (r["bbox_pixel"][0] + r["bbox_pixel"][2]) / 2)

    gt_by_q = {}
    for gt in gt_data:
        gt_by_q.setdefault(gt["q_idx"], []).append(gt)

    print(f"\n=== 字符级坐标对比 ===")
    print(f"{'q_idx':>5} {'字':>3} {'gt_x':>6} {'gt_y':>6} {'pl_x':>6} {'pl_y':>6} "
          f"{'Δx':>6} {'Δy':>6} {'gt_w':>5} {'gt_h':>5} {'pl_w':>5} {'pl_h':>5}")
    print("-" * 85)

    errors_x = []
    errors_y = []
    matched = 0
    for qi in sorted(set(gt_by_q.keys()) & set(pipeline_by_q.keys())):
        gt_items = gt_by_q[qi]
        pl_items = pipeline_by_q[qi]
        n = min(len(gt_items), len(pl_items))
        for i in range(n):
            gt = gt_items[i]
            pl = pl_items[i]
            gt_cx, gt_cy = gt["center_pixel"]
            pl_bbox = pl["bbox_pixel"]
            pl_cx = (pl_bbox[0] + pl_bbox[2]) / 2
            pl_cy = (pl_bbox[1] + pl_bbox[3]) / 2
            dx = pl_cx - gt_cx
            dy = pl_cy - gt_cy
            errors_x.append(abs(dx))
            errors_y.append(abs(dy))
            matched += 1
            gt_w = gt["rect_pixel"][2] - gt["rect_pixel"][0]
            gt_h = gt["rect_pixel"][3] - gt["rect_pixel"][1]
            pl_w = pl_bbox[2] - pl_bbox[0]
            pl_h = pl_bbox[3] - pl_bbox[1]
            char = gt["char"]
            print(f"{qi:>5} {char:>3} {gt_cx:>6.0f} {gt_cy:>6.0f} "
                  f"{pl_cx:>6.0f} {pl_cy:>6.0f} {dx:>+6.0f} {dy:>+6.0f} "
                  f"{gt_w:>5.0f} {gt_h:>5.0f} {pl_w:>5.0f} {pl_h:>5.0f}")

    print(f"\n=== 误差统计 (n={matched}) ===")
    if errors_x:
        avg_x = sum(errors_x) / len(errors_x)
        avg_y = sum(errors_y) / len(errors_y)
        max_x = max(errors_x)
        max_y = max(errors_y)
        sorted_x = sorted(errors_x)
        sorted_y = sorted(errors_y)
        p50_x = sorted_x[len(sorted_x) // 2]
        p50_y = sorted_y[len(sorted_y) // 2]
        p90_x = sorted_x[int(len(sorted_x) * 0.9)]
        p90_y = sorted_y[int(len(sorted_y) * 0.9)]
        print(f"  X 误差: avg={avg_x:.1f}px, p50={p50_x:.1f}px, p90={p90_x:.1f}px, max={max_x:.1f}px")
        print(f"  Y 误差: avg={avg_y:.1f}px, p50={p50_y:.1f}px, p90={p90_y:.1f}px, max={max_y:.1f}px")
        print(f"  允许容差参考: img={1653}x{2312}, 字符高≈{sum(e['h_pt'] for e in gt_data) / len(gt_data) * 2.778:.0f}px")
        within_20px = sum(1 for e in errors_x + errors_y if e < 20)
        print(f"  误差 < 20px 的占比: {within_20px / (2 * len(errors_x)):.1%}")

    print(f"\n=== 输出文件 ===")
    print(f"  黄金标准 PNG: {golden_png}")
    print(f"  流水线 PNG:   {pipeline_png}")
    print(f"  黄金标准 PDF: {GOLDEN_PDF}")
    print(f"  流水线 PDF:   {PIPELINE_PDF}")
    print(f"  Ground Truth: {GT_JSON}")


if __name__ == "__main__":
    main()
