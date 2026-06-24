"""整页 Qwen3-VL 识别测试 — 运行完整流水线（识别 + 批改 + 批注 PDF）。"""
import shutil
import sys
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s", datefmt="%H:%M:%S")

PDF = "301_2026-06-18_003.pdf"

# 确保 recognition_config 已设为 page_level
from core.recognition_config import RECOGNITION_ENGINE
assert RECOGNITION_ENGINE == "page_level", f"引擎应为 page_level，当前为 {RECOGNITION_ENGINE}"
print(f"✅ 识别引擎: {RECOGNITION_ENGINE}")

# 运行完整流水线
from monitor.processor import process_pdf
root = Path(__file__).resolve().parent
pdf_path = root / PDF
if not pdf_path.exists():
    # 尝试从 backup 恢复
    backup = root / "data" / "backup" / PDF
    if backup.exists():
        shutil.copy2(str(backup), str(pdf_path))
        print(f"📄 已从 backup 恢复: {PDF}")
    else:
        print(f"❌ 文件不存在: {pdf_path}")
        sys.exit(1)

process_pdf(str(pdf_path))

print("\n✅ 处理完成，批注 PDF 在 data/output/")
