"""基于人工标注 box 生成黄金标准批注 PDF。

输入:
  - data/test_samples/301_2026-06-18_003-人工标注box.pdf (53 个 Square 注释)
  - 标准答案 (硬编码，与数据库一致)

处理流程:
  1. 提取 53 个 Square 注释的 rect
  2. 按阅读顺序排序 (y 优先, x 次之), 按 y 行分组
  3. 按题目字数分布 (7+5+7+6+7+7+5+5+2+2=53) 将标准答案字符填入 box
  4. 在原 PDF 上用绿色文字渲染每个字符到 box 中心 (颜色编码: 全对=绿)
  5. 同时保存 ground_truth.json (含 pt 和 pixel 坐标, 供对比测试用)

输出:
  - data/output/301_2026-06-18_003_golden.pdf
  - data/test_samples/301_2026-06-18_003_ground_truth.json
"""
import json
import sys
from pathlib import Path

import fitz

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

SRC_PDF = ROOT / "data" / "test_samples" / "301_2026-06-18_003.pdf"
ANNOT_PDF = ROOT / "data" / "test_samples" / "301_2026-06-18_003-人工标注box.pdf"
OUT_PDF = ROOT / "data" / "output" / "301_2026-06-18_003_golden.pdf"
OUT_JSON = ROOT / "data" / "test_samples" / "301_2026-06-18_003_ground_truth.json"

# 标准答案 (按题号顺序, 与人工标注 box 的阅读顺序一致)
# (q_idx, 题号显示, [字符列表])
# 注: q_idx 与 layout_analyzer 实际输出对齐:
#   - 题8 在 pipeline 中是 q_idx=201 (不是 7)
#   - ① 题在 pipeline 中是 q_idx=101
#   - ② 题 (q_idx=102) 在 pipeline 中未识别, gt 保留以备后续验证
STANDARD_ANSWERS = [
    (0, "（1）", list("随君直到夜郎西")),       # 7字 - 行1左
    (1, "（2）", list("海内存知己")),           # 5字 - 行1右
    (2, "（3）", list("百般红紫斗芳菲")),       # 7字 - 行2左
    (3, "（4）", list("水中藻荇交横")),         # 6字 - 行2右
    (4, "（5）", list("人生自古谁无死")),       # 7字 - 行3左
    (5, "（6）", list("半竿斜日旧关城")),       # 7字 - 行3右
    (6, "（7）", list("采菊东篱下")),           # 5字 - 行4左
    (201, "（8）", list("悠然见南山")),         # 5字 - 行4右 (pipeline q_idx=201)
    (101, "①", list("质朴")),                  # 2字 - 行5左
    (102, "②", list("绚丽")),                  # 2字 - 行5右 (pipeline 未识别)
]

# 流水线 img 尺寸 (用于 pt → pixel 转换)
IMG_W = 1653
IMG_H = 2312


def main():
    if not ANNOT_PDF.exists():
        print(f"人工标注 PDF 不存在: {ANNOT_PDF}")
        return
    if not SRC_PDF.exists():
        print(f"原始 PDF 不存在: {SRC_PDF}")
        return

    # Step 1: 提取所有 Square 注释
    src_doc = fitz.open(str(ANNOT_PDF))
    page = src_doc.load_page(0)
    page_w_pt = page.rect.width
    page_h_pt = page.rect.height
    print(f"页面尺寸: {page_w_pt:.1f} x {page_h_pt:.1f} pt")
    print(f"图像尺寸: {IMG_W} x {IMG_H} px")
    print(f"缩放比: x={IMG_W / page_w_pt:.4f}, y={IMG_H / page_h_pt:.4f}")

    annots = list(page.annots())
    squares = []
    for a in annots:
        if a.type[1] != "Square":
            continue
        r = a.rect
        cx = (r.x0 + r.x1) / 2
        cy = (r.y0 + r.y1) / 2
        squares.append({
            "rect_pt": (r.x0, r.y0, r.x1, r.y1),
            "center_pt": (cx, cy),
            "w_pt": r.x1 - r.x0,
            "h_pt": r.y1 - r.y0,
        })
    print(f"提取 Square 注释: {len(squares)} 个")
    src_doc.close()

    # Step 2: 按 y 行分组 (容差 25pt), 行内按 x 排序
    squares_sorted = sorted(squares, key=lambda s: (s["center_pt"][1], s["center_pt"][0]))
    rows = []
    cur_row = [squares_sorted[0]]
    cur_y = squares_sorted[0]["center_pt"][1]
    for s in squares_sorted[1:]:
        if abs(s["center_pt"][1] - cur_y) < 25:
            cur_row.append(s)
        else:
            rows.append(cur_row)
            cur_row = [s]
            cur_y = s["center_pt"][1]
    rows.append(cur_row)

    print(f"\n按 y 分组: {len(rows)} 行")
    for i, row in enumerate(rows):
        row.sort(key=lambda s: s["center_pt"][0])
        ys = [s["center_pt"][1] for s in row]
        xs = [s["center_pt"][0] for s in row]
        print(f"  行{i + 1}: {len(row)} boxes, x=[{min(xs):.0f},{max(xs):.0f}], y={sum(ys) / len(ys):.0f}")

    # Step 3: 按"每行 = 2 个题目"展开 (左列 + 右列), 然后按标准答案字符填充
    # 但每行的 box 是按 x 全局排序的, 需要按 page mid (page_w_pt/2) 拆分左右
    page_mid_pt = page_w_pt / 2
    print(f"\n页面中线 x = {page_mid_pt:.1f} pt")

    # 展开成 [(q_idx, char, box)] 列表
    ground_truth = []
    q_idx_iter = iter(STANDARD_ANSWERS)
    for row_idx, row in enumerate(rows):
        # 拆分左右列
        left_boxes = [s for s in row if s["center_pt"][0] < page_mid_pt]
        right_boxes = [s for s in row if s["center_pt"][0] >= page_mid_pt]
        left_boxes.sort(key=lambda s: s["center_pt"][0])
        right_boxes.sort(key=lambda s: s["center_pt"][0])
        print(f"\n行{row_idx + 1}: 左列 {len(left_boxes)} box, 右列 {len(right_boxes)} box")

        # 左列对应一题
        if left_boxes:
            q_idx, q_label, chars = next(q_idx_iter)
            print(f"  左列 → q_idx={q_idx} {q_label} chars={''.join(chars)} ({len(chars)}字)")
            if len(left_boxes) != len(chars):
                print(f"  ⚠️ 警告: box 数 {len(left_boxes)} != 字数 {len(chars)}")
            for i, box in enumerate(left_boxes):
                ch = chars[i] if i < len(chars) else "?"
                ground_truth.append({
                    "q_idx": q_idx,
                    "q_label": q_label,
                    "char_idx": i,
                    "char": ch,
                    "rect_pt": box["rect_pt"],
                    "center_pt": box["center_pt"],
                    "w_pt": box["w_pt"],
                    "h_pt": box["h_pt"],
                })

        # 右列对应一题
        if right_boxes:
            q_idx, q_label, chars = next(q_idx_iter)
            print(f"  右列 → q_idx={q_idx} {q_label} chars={''.join(chars)} ({len(chars)}字)")
            if len(right_boxes) != len(chars):
                print(f"  ⚠️ 警告: box 数 {len(right_boxes)} != 字数 {len(chars)}")
            for i, box in enumerate(right_boxes):
                ch = chars[i] if i < len(chars) else "?"
                ground_truth.append({
                    "q_idx": q_idx,
                    "q_label": q_label,
                    "char_idx": i,
                    "char": ch,
                    "rect_pt": box["rect_pt"],
                    "center_pt": box["center_pt"],
                    "w_pt": box["w_pt"],
                    "h_pt": box["h_pt"],
                })

    # Step 4: 计算 pixel 坐标 (PyMuPDF rect 是 top-left 原点, 与 pixel 一致)
    scale_x = IMG_W / page_w_pt
    scale_y = IMG_H / page_h_pt
    for gt in ground_truth:
        x0, y0, x1, y1 = gt["rect_pt"]
        gt["rect_pixel"] = (x0 * scale_x, y0 * scale_y, x1 * scale_x, y1 * scale_y)
        cx, cy = gt["center_pt"]
        gt["center_pixel"] = (cx * scale_x, cy * scale_y)

    print(f"\n=== Ground Truth 统计 ===")
    print(f"总字数: {len(ground_truth)}")
    by_q = {}
    for gt in ground_truth:
        by_q.setdefault(gt["q_idx"], []).append(gt["char"])
    for q_idx in sorted(by_q):
        print(f"  q_idx={q_idx}: {''.join(by_q[q_idx])}")

    # Step 5: 在原 PDF 上渲染标准答案字符 (绿色, 表示全对)
    out_doc = fitz.open(str(SRC_PDF))
    out_page = out_doc.load_page(0)

    # 注册中文字体
    fontname = "china-s"  # PyMuPDF 内置简体中文
    for gt in ground_truth:
        cx, cy = gt["center_pt"]
        x0, y0, x1, y1 = gt["rect_pt"]
        w = x1 - x0
        h = y1 - y0
        # 字号自适应 box 高度
        fontsize = min(h * 0.85, 18)
        # 居中插入文字 (绿色 = 全对)
        # fitz insert_text 用基线坐标, 这里用 box 中心向下偏移 fontsize*0.35
        baseline_x = cx - fontsize * 0.5  # 粗略居中
        baseline_y = cy + fontsize * 0.35
        out_page.insert_text(
            (baseline_x, baseline_y),
            gt["char"],
            fontsize=fontsize,
            fontname=fontname,
            color=(0, 0.7, 0),  # 绿色
        )
        # 同时画一个绿色边框 (可选, 注释掉以保持 PDF 简洁)
        # out_page.draw_rect(fitz.Rect(x0, y0, x1, y1), color=(0, 0.7, 0), width=0.5)

    # 保存黄金标准 PDF
    OUT_PDF.parent.mkdir(parents=True, exist_ok=True)
    out_doc.save(str(OUT_PDF))
    out_doc.close()
    print(f"\n已生成黄金标准 PDF: {OUT_PDF}")

    # 保存 ground truth JSON
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        # 转换 tuple 为 list 以便 JSON 序列化
        serializable = []
        for gt in ground_truth:
            item = {k: (list(v) if isinstance(v, tuple) else v) for k, v in gt.items()}
            serializable.append(item)
        json.dump(serializable, f, ensure_ascii=False, indent=2)
    print(f"已保存 ground truth JSON: {OUT_JSON}")


if __name__ == "__main__":
    main()
