"""深入分析流水线输出 vs 黄金标准的问题分类。

按问题类型分组:
  P1: X 累积偏移 (等宽切分) — 长答案越往后偏差越大
  P2: Y 系统性偏移 (y-clamp 用错手写框)
  P3: q_idx 错配 (题号映射错误，如题8 被分配到行5)
  P4: 字符遗漏 (整题未识别，如 ② 题)
  P5: 字符多识别 (pipeline 多识别字符)
  P6: bbox 尺寸异常 (宽高比与 gt 不符)

同时生成左右拼接对比 PNG (左: 流水线, 右: 黄金标准)
"""
import json
import sys
from pathlib import Path
from collections import defaultdict

import fitz
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

GOLDEN_PDF = ROOT / "data" / "output" / "301_2026-06-18_003_golden.pdf"
PIPELINE_PDF = ROOT / "data" / "output" / "301_2026-06-18_003_annotated.pdf"
GT_JSON = ROOT / "data" / "test_samples" / "301_2026-06-18_003_ground_truth.json"
SIDE_BY_SIDE_PNG = ROOT / "data" / "output" / "301_2026-06-18_003_side_by_side.png"

RENDER_DPI = 150


def render_pdf(pdf_path: Path) -> Image.Image:
    doc = fitz.open(str(pdf_path))
    page = doc.load_page(0)
    zoom = RENDER_DPI / 72
    pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), colorspace=fitz.csRGB)
    img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
    doc.close()
    return img


def main():
    # 加载数据
    with open(GT_JSON, "r", encoding="utf-8") as f:
        gt_data = json.load(f)
    from db.crud import get_tasks
    task = get_tasks(limit=1)[0]
    pl_data = task.result_json

    print(f"=== 数据概览 ===")
    print(f"  Ground Truth: {len(gt_data)} 字 (人工标注 53 box, 含 ② 题未识别)")
    print(f"  Pipeline:     {len(pl_data)} 字 (task #{task.id})")
    print()

    # 按 q_idx 分组
    gt_by_q = defaultdict(list)
    for gt in gt_data:
        gt_by_q[gt["q_idx"]].append(gt)
    pl_by_q = defaultdict(list)
    for r in pl_data:
        pl_by_q[r.get("question_idx", -1)].append(r)
    for items in pl_by_q.values():
        items.sort(key=lambda r: (r["bbox_pixel"][0] + r["bbox_pixel"][2]) / 2)
    for items in gt_by_q.values():
        items.sort(key=lambda r: r["center_pixel"][0])

    # ============ P1: X 累积偏移分析 ============
    print("=" * 75)
    print("P1: X 累积偏移分析 (等宽切分 vs 真实位置)")
    print("=" * 75)
    print(f"{'q_idx':>5} {'字':>3} {'位置':>4} {'gt_x':>6} {'pl_x':>6} {'Δx':>6} "
          f"{'累积Δx':>8} {'间距(gt)':>8} {'间距(pl)':>8}")
    print("-" * 75)

    p1_cases = []
    for qi in sorted(set(gt_by_q.keys()) & set(pl_by_q.keys())):
        gt_items = gt_by_q[qi]
        pl_items = pl_by_q[qi]
        n = min(len(gt_items), len(pl_items))
        if n < 2:
            continue
        prev_gt_x = None
        prev_pl_x = None
        for i in range(n):
            gt = gt_items[i]
            pl = pl_items[i]
            gt_x = gt["center_pixel"][0]
            pl_x = (pl["bbox_pixel"][0] + pl["bbox_pixel"][2]) / 2
            dx = pl_x - gt_x
            gt_gap = gt_x - prev_gt_x if prev_gt_x is not None else 0
            pl_gap = pl_x - prev_pl_x if prev_pl_x is not None else 0
            print(f"{qi:>5} {gt['char']:>3} {i + 1:>2}/{n} "
                  f"{gt_x:>6.0f} {pl_x:>6.0f} {dx:>+6.0f} "
                  f"{dx:>+8.0f} {gt_gap:>8.0f} {pl_gap:>8.0f}")
            if abs(dx) > 100:
                p1_cases.append({"q_idx": qi, "char": gt["char"], "pos": i + 1,
                                 "n": n, "dx": dx})
            prev_gt_x = gt_x
            prev_pl_x = pl_x
        print()

    print(f"P1 总结: 累积偏移 > 100px 的字符 {len(p1_cases)} 个")
    if p1_cases:
        # 按位置统计
        by_pos = defaultdict(int)
        for c in p1_cases:
            by_pos[c["pos"]] += 1
        print(f"  按字符位置分布 (1=首字, n=末字):")
        for pos in sorted(by_pos):
            print(f"    位置 {pos}: {by_pos[pos]} 字")
    print()

    # ============ P2: Y 系统性偏移分析 ============
    print("=" * 75)
    print("P2: Y 系统性偏移 (y-clamp 用错手写框)")
    print("=" * 75)
    print(f"{'q_idx':>5} {'题号':>5} {'n':>3} {'gt_y_avg':>9} {'pl_y_avg':>9} "
          f"{'Δy_avg':>8} {'问题':>20}")
    print("-" * 75)

    p2_cases = []
    for qi in sorted(set(gt_by_q.keys()) & set(pl_by_q.keys())):
        gt_items = gt_by_q[qi]
        pl_items = pl_by_q[qi]
        n = min(len(gt_items), len(pl_items))
        if n == 0:
            continue
        gt_y_avg = sum(g["center_pixel"][1] for g in gt_items[:n]) / n
        pl_y_avg = sum((p["bbox_pixel"][1] + p["bbox_pixel"][3]) / 2
                       for p in pl_items[:n]) / n
        dy = pl_y_avg - gt_y_avg
        issue = ""
        if abs(dy) > 200:
            issue = "❌ 严重偏移 (q_idx 错配?)"
        elif abs(dy) > 100:
            issue = "⚠️ 较大偏移 (y-clamp 用错框)"
        elif abs(dy) > 50:
            issue = "⚠ 中等偏移 (手写框边距)"
        else:
            issue = "✓ 正常"
        label = gt_items[0].get("q_label", "")
        print(f"{qi:>5} {label:>5} {n:>3} {gt_y_avg:>9.0f} {pl_y_avg:>9.0f} "
              f"{dy:>+8.0f} {issue:>20}")
        if abs(dy) > 50:
            p2_cases.append({"q_idx": qi, "label": label, "dy": dy, "issue": issue})

    print(f"\nP2 总结: Y 偏移 > 50px 的题目 {len(p2_cases)} 个")
    print()

    # ============ P3: q_idx 错配分析 ============
    print("=" * 75)
    print("P3: q_idx 错配 (题号映射错误)")
    print("=" * 75)
    # 检查 q_idx=201 (题8) 的情况
    if 201 in pl_by_q and 201 in gt_by_q:
        pl_y_range = (min(p["bbox_pixel"][1] for p in pl_by_q[201]),
                      max(p["bbox_pixel"][3] for p in pl_by_q[201]))
        gt_y_range = (min(g["center_pixel"][1] for g in gt_by_q[201]) - 50,
                      max(g["center_pixel"][1] for g in gt_by_q[201]) + 50)
        print(f"  q_idx=201:")
        print(f"    pipeline: y={pl_y_range[0]:.0f}-{pl_y_range[1]:.0f} "
              f"(行5右区域)")
        print(f"    gt:       y={gt_y_range[0]:.0f}-{gt_y_range[1]:.0f} "
              f"(行4右区域, 题8 悠然见南山)")
        print(f"    问题: layout_analyzer 把题8 检测到了行5, 实际在行4")
        print(f"    影响: pipeline 把 '悠然见南' 渲染到了 ① 题下方, 完全错位")
    print()

    # ============ P4: 字符遗漏分析 ============
    print("=" * 75)
    print("P4: 字符遗漏 (整题或单字未识别)")
    print("=" * 75)
    for qi in sorted(set(gt_by_q.keys()) - set(pl_by_q.keys())):
        gt_items = gt_by_q[qi]
        label = gt_items[0].get("q_label", "")
        chars = "".join(g["char"] for g in gt_items)
        print(f"  q_idx={qi} ({label}): 完全遗漏, 应识别 '{chars}' ({len(gt_items)}字)")

    for qi in sorted(set(gt_by_q.keys()) & set(pl_by_q.keys())):
        gt_n = len(gt_by_q[qi])
        pl_n = len(pl_by_q[qi])
        if pl_n < gt_n:
            label = gt_by_q[qi][0].get("q_label", "")
            gt_chars = "".join(g["char"] for g in gt_by_q[qi])
            pl_chars = "".join(p.get("char", "?") for p in pl_by_q[qi])
            print(f"  q_idx={qi} ({label}): 部分遗漏, gt={gt_n}字 '{gt_chars}', "
                  f"pl={pl_n}字 '{pl_chars}'")
    print()

    # ============ P5: 字符多识别分析 ============
    print("=" * 75)
    print("P5: 字符多识别 (pipeline 多识别)")
    print("=" * 75)
    for qi in sorted(set(pl_by_q.keys()) - set(gt_by_q.keys())):
        pl_items = pl_by_q[qi]
        chars = "".join(p.get("char", "?") for p in pl_items)
        print(f"  q_idx={qi}: 多识别 {len(pl_items)}字 '{chars}' "
              f"(gt 中无此 q_idx, 可能是下联等未人工标注的题)")
    print()

    # ============ P6: bbox 尺寸分析 ============
    print("=" * 75)
    print("P6: bbox 尺寸异常")
    print("=" * 75)
    print(f"{'q_idx':>5} {'字':>3} {'gt_w':>5} {'gt_h':>5} {'pl_w':>5} {'pl_h':>5} "
          f"{'宽比':>6} {'高比':>6} {'问题':>20}")
    print("-" * 75)
    p6_cases = []
    for qi in sorted(set(gt_by_q.keys()) & set(pl_by_q.keys())):
        gt_items = gt_by_q[qi]
        pl_items = pl_by_q[qi]
        n = min(len(gt_items), len(pl_items))
        for i in range(n):
            gt = gt_items[i]
            pl = pl_items[i]
            gt_w = gt["rect_pixel"][2] - gt["rect_pixel"][0]
            gt_h = gt["rect_pixel"][3] - gt["rect_pixel"][1]
            pl_bbox = pl["bbox_pixel"]
            pl_w = pl_bbox[2] - pl_bbox[0]
            pl_h = pl_bbox[3] - pl_bbox[1]
            w_ratio = pl_w / gt_w if gt_w > 0 else 0
            h_ratio = pl_h / gt_h if gt_h > 0 else 0
            issue = ""
            if pl_w < 30:
                issue = "⚠ 过窄 (挤压)"
            elif pl_w > 100:
                issue = "⚠ 过宽 (拉伸)"
            elif abs(h_ratio - 1) > 0.5:
                issue = "⚠ 高度异常"
            if issue:
                print(f"{qi:>5} {gt['char']:>3} {gt_w:>5.0f} {gt_h:>5.0f} "
                      f"{pl_w:>5.0f} {pl_h:>5.0f} {w_ratio:>6.2f} {h_ratio:>6.2f} "
                      f"{issue:>20}")
                p6_cases.append({"q_idx": qi, "char": gt["char"], "issue": issue})
    print(f"\nP6 总结: bbox 尺寸异常 {len(p6_cases)} 个")
    print()

    # ============ 生成左右拼接对比图 ============
    print("=" * 75)
    print("生成左右拼接对比图")
    print("=" * 75)
    pl_img = render_pdf(PIPELINE_PDF)
    gt_img = render_pdf(GOLDEN_PDF)
    # 缩放到相同高度
    h = min(pl_img.size[1], gt_img.size[1])
    pl_img = pl_img.resize((int(pl_img.size[0] * h / pl_img.size[1]), h))
    gt_img = gt_img.resize((int(gt_img.size[0] * h / gt_img.size[1]), h))
    # 拼接 + 分隔线
    gap = 20
    canvas = Image.new("RGB", (pl_img.size[0] + gap + gt_img.size[0], h), (255, 255, 255))
    canvas.paste(pl_img, (0, 0))
    canvas.paste(gt_img, (pl_img.size[0] + gap, 0))
    draw = ImageDraw.Draw(canvas)
    draw.line([(pl_img.size[0] + gap // 2, 0),
               (pl_img.size[0] + gap // 2, h)], fill=(255, 0, 0), width=2)
    # 添加标题
    draw.text((10, 5), "Pipeline (annotated.pdf)", fill=(255, 0, 0))
    draw.text((pl_img.size[0] + gap + 10, 5), "Golden (ground truth)",
              fill=(0, 178, 0))
    canvas.save(SIDE_BY_SIDE_PNG)
    print(f"  已保存: {SIDE_BY_SIDE_PNG}")
    print(f"  尺寸: {canvas.size[0]}x{canvas.size[1]}")

    # ============ 总结 ============
    print()
    print("=" * 75)
    print("问题总结")
    print("=" * 75)
    print(f"P1 X 累积偏移 (>100px):    {len(p1_cases)} 字")
    print(f"P2 Y 系统性偏移 (>50px):   {len(p2_cases)} 题")
    print(f"P3 q_idx 错配:             1 题 (q_idx=201 题8)")
    print(f"P4 字符遗漏:               1 题完全遗漏 (② 题), 1 题部分遗漏 (题8 缺'山')")
    print(f"P5 字符多识别:             1 题 (q_idx=104 下联, gt 未标注, 非问题)")
    print(f"P6 bbox 尺寸异常:          {len(p6_cases)} 字")


if __name__ == "__main__":
    main()
