"""
调试 v6 — 探查 json['res'] dict 的键。
"""
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

from pathlib import Path
import fitz
import numpy as np
from PIL import Image

pdf_path = Path(__file__).resolve().parent / "data" / "failed" / "301_2025-03-20_001.pdf"
doc = fitz.open(str(pdf_path))
page = doc.load_page(0)
pix = page.get_pixmap(matrix=fitz.Matrix(200/72, 200/72), colorspace=fitz.csRGB)
img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
img_np = np.array(img)
doc.close()

from paddleocr import PaddleOCR
ocr = PaddleOCR(lang='ch')
raw_pages = ocr.predict(img_np)

for page_result in raw_pages:
    j = getattr(page_result, 'json', {})
    print(f"\n=== json top-level keys: {list(j.keys()) if isinstance(j, dict) else type(j)} ===")
    res = j.get('res', {})
    print(f"res type={type(res).__name__}, keys={list(res.keys()) if isinstance(res, dict) else 'N/A'}")
    
    # Print all keys and their types
    if isinstance(res, dict):
        for k, v in res.items():
            val_type = type(v).__name__
            val_len = f", len={len(v)}" if hasattr(v, '__len__') else ""
            print(f"  '{k}': {val_type}{val_len}")
            # If it's a dict, print first 3 keys
            if isinstance(v, dict) and len(v) > 0:
                first_keys = list(v.keys())[:3]
                print(f"      sub-keys: {first_keys}")
            # If it's a list, print first element type
            if isinstance(v, list) and len(v) > 0:
                print(f"      [0] type={type(v[0]).__name__}")
    
    # Check for dt_polys in overall_ocr_res
    oocr = res.get('overall_ocr_res')
    if oocr is not None:
        print(f"\noverall_ocr_res: type={type(oocr).__name__}")
        if isinstance(oocr, dict):
            print(f"  keys={list(oocr.keys())}")
        elif isinstance(oocr, list):
            print(f"  len={len(oocr)}")
            if len(oocr) > 0:
                print(f"  [0] type={type(oocr[0]).__name__}")
            
# Also view the str representation (truncated)
s = str(page_result)
# Find dt_polys in str
for keyword in ['dt_polys', 'rec_texts', 'rec_polys', 'rec_boxes']:
    idx = s.find(keyword)
    if idx >= 0:
        print(f"\n--- '{keyword}' in str() at index {idx} ---")
        start = max(0, idx-100)
        end = min(len(s), idx+200)
        print(s[start:end])