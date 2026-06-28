"""检查 OCR 缓存内容，找出题8 和 ② 的题号匹配情况。"""
import pickle
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from services.pipeline_service import _file_hash

pdf = ROOT / "data" / "test_samples" / "301_2026-06-18_003.pdf"
fh = _file_hash(pdf)
print(f"PDF hash: {fh}")

cache_dir = ROOT / "data" / "cache"
for f in cache_dir.glob("*.pkl"):
    if fh in f.name:
        print(f"匹配缓存: {f.name}")
        d = pickle.load(open(f, "rb"))
        det_boxes, ocr_records = d
        print(f"det_boxes: {len(det_boxes)}, ocr_records: {len(ocr_records)}")
        from core.layout_analyzer import _question_match
        print("\nOCR records 和题号匹配:")
        for i, r in enumerate(ocr_records):
            text = r["rec_text"]
            bbox = r["rec_bbox"]
            qi = _question_match(text)
            cy = (bbox[1] + bbox[3]) / 2
            cx = (bbox[0] + bbox[2]) / 2
            print(f"  [{i:2d}] qi={qi} cy={cy:.0f} cx={cx:.0f} text={text!r}")
