"""提取人工标注 PDF 中每个字的 box。

输入: data/test_samples/301_2026-06-18_003-人工标注box.pdf
输出: 打印每个标注的 [页码, 类型, rect, content]，并保存为 JSON

PDF 注释类型常见:
  - Highlight (高亮)
  - Square (矩形框)
  - Circle (椭圆)
  - Ink (手绘)
  - Text (注释便签)
"""
import json
import sys
from pathlib import Path

import fitz

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

PDF_PATH = ROOT / "data" / "test_samples" / "301_2026-06-18_003-人工标注box.pdf"
OUT_JSON = ROOT / "data" / "test_samples" / "301_2026-06-18_003_manual_boxes.json"


def main():
    if not PDF_PATH.exists():
        print(f"文件不存在: {PDF_PATH}")
        return

    doc = fitz.open(str(PDF_PATH))
    print(f"PDF: {PDF_PATH.name}")
    print(f"页数: {len(doc)}")
    page = doc.load_page(0)
    print(f"页面尺寸: {page.rect.width:.1f} x {page.rect.height:.1f} pt")
    print()

    annots = list(page.annots())
    print(f"第 1 页注释总数: {len(annots)}")
    print()

    # 按类型统计
    by_type = {}
    for a in annots:
        t = a.type[1]  # e.g. "Highlight"
        by_type[t] = by_type.get(t, 0) + 1
    print("按类型统计:")
    for t, n in sorted(by_type.items()):
        print(f"  {t}: {n}")
    print()

    # 输出每个注释
    print(f"{'#':>3} {'类型':>12} {'rect (x0,y0,x1,y1)':>30} {'content':>20}")
    print("-" * 75)
    records = []
    for i, a in enumerate(annots):
        rect = a.rect
        content = a.info.get("content", "") or ""
        rtype = a.type[1]
        print(f"{i + 1:>3} {rtype:>12} ({rect.x0:>6.1f},{rect.y0:>6.1f},{rect.x1:>6.1f},{rect.y1:>6.1f}) {content:>20}")
        records.append({
            "idx": i,
            "type": rtype,
            "rect": [rect.x0, rect.y0, rect.x1, rect.y1],
            "content": content,
        })

    # 保存 JSON
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    print(f"\n已保存: {OUT_JSON}")

    doc.close()


if __name__ == "__main__":
    main()
