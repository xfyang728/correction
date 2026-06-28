"""视觉验证辅助脚本 — 统计字符位置 vs 手写框覆盖情况。

不替代肉眼检查 PNG，但提供客观数据辅助判断：
  - 每个字符的 bbox 是否落在对应手写框内
  - 每道题的字符 y 范围是否与手写框 y 范围一致（y-clamp 有效性）
  - 颜色编码统计（正确/错误/存疑）

用法: python tools/_verify_annotation_position.py
"""
import sys
from collections import Counter, defaultdict
from pathlib import Path

import fitz

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
PDF_PATH = ROOT / "data" / "output" / "301_2026-06-18_003_annotated.pdf"


def main():
    from db.crud import get_tasks

    if not PDF_PATH.exists():
        print(f"批注 PDF 不存在: {PDF_PATH}")
        return

    t = get_tasks(limit=1)[0]
    results = t.result_json
    print(f"任务 #{t.id}: {t.filename}")
    print(f"  总字数={t.total_chars} 正确={t.correct_count} "
          f"存疑={t.uncertain_count} 错误={t.wrong_count}")
    print()

    # 按 q_idx 分组
    g = defaultdict(list)
    for r in results:
        g[r.get("question_idx", -1)].append(r)

    print(f"{'q_idx':>6} {'n':>3} {'x范围':>15} {'y范围':>15} "
          f"{'正确':>4} {'错误':>4} {'存疑':>4}  首字")
    print("-" * 75)
    for qi, v in sorted(g.items()):
        x0 = min(x["bbox_pixel"][0] for x in v)
        x1 = max(x["bbox_pixel"][2] for x in v)
        y0 = min(x["bbox_pixel"][1] for x in v)
        y1 = max(x["bbox_pixel"][3] for x in v)
        correct = sum(1 for x in v if x.get("status") == "correct")
        wrong = sum(1 for x in v if x.get("status") == "wrong")
        uncertain = sum(1 for x in v if x.get("status") == "uncertain")
        first_char = v[0].get("char", "")
        print(f"{qi:>6} {len(v):>3} ({x0:>4},{x1:>4}) ({y0:>4},{y1:>4}) "
              f"{correct:>4} {wrong:>4} {uncertain:>4}  {first_char}")

    # 坐标来源分布统计（page_level 模式 4 条路径追踪）
    print("\n坐标来源分布:")
    coord_sources = Counter(r.get("coord_source", "unknown") for r in results)
    total = len(results) if results else 1
    for source, count in sorted(coord_sources.items()):
        print(f"  {source:>25}: {count:>3} 字 ({count/total*100:.1f}%)")

    # 按 q_idx × coord_source 交叉表（识别哪些题走了 fallback）
    print(f"\n{'q_idx':>6} {'coord_source':>25} {'n':>3}")
    print("-" * 40)
    q_source: dict = defaultdict(Counter)
    for r in results:
        qi = r.get("question_idx", -1)
        src = r.get("coord_source", "unknown")
        q_source[qi][src] += 1
    for qi in sorted(q_source):
        for src, cnt in q_source[qi].most_common():
            print(f"{qi:>6} {src:>25} {cnt:>3}")

    # 检查 PDF 中文字注记数量
    print()
    doc = fitz.open(str(PDF_PATH))
    for page_idx in range(len(doc)):
        page = doc.load_page(page_idx)
        text_blocks = page.get_text("dict")["blocks"]
        n_text = sum(1 for b in text_blocks if b.get("type") == 0
                     for l in b.get("lines", []))
        print(f"PDF 第 {page_idx + 1} 页: {n_text} 个文字行")
    doc.close()

    png_path = PDF_PATH.parent / f"{PDF_PATH.stem}_page1.png"
    if png_path.exists():
        print(f"\n视觉验证 PNG: {png_path}")


if __name__ == "__main__":
    main()
