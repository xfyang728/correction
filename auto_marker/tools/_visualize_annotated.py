"""可视化批注 PDF 的标记位置"""
import sys
from pathlib import Path
import fitz
from PIL import Image

# 渲染 DPI
RENDER_DPI = 150

def visualize_annotated_pdf(pdf_path: str):
    """渲染批注 PDF 并显示标记位置"""
    path = Path(pdf_path)
    if not path.exists():
        print(f"文件不存在: {pdf_path}")
        return

    # 渲染 PDF
    doc = fitz.open(str(path))
    for page_idx in range(len(doc)):
        page = doc.load_page(page_idx)
        zoom = RENDER_DPI / 72
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat, colorspace=fitz.csRGB)

        img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)

        # 保存渲染图像
        output_path = path.parent / f"{path.stem}_page{page_idx + 1}.png"
        img.save(output_path)
        print(f"已保存: {output_path}")

    doc.close()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        pdf_path = "data/output/301_2026-06-18_003_annotated.pdf"
    else:
        pdf_path = sys.argv[1]

    visualize_annotated_pdf(pdf_path)
