"""
测试运行器 — 端到端测试流程。

用法:
  1. 先在 Web 界面 (http://localhost:8501) 的「答案管理」标签页输入标准答案
  2. 然后运行: python run_test.py

或者直接预置答案后运行全程:
  python run_test.py --seed-answers "春眠不觉晓处处闻啼鸟"
"""

import argparse
import logging
import sys
from pathlib import Path

# 确保项目根在 sys.path
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)


def seed_answers(class_name: str, date_str: str, text: str):
    """预置标准答案到数据库（模拟 Web UI 保存操作）。"""
    from db.crud import save_answer
    save_answer(class_name, date_str, text)
    print(f"✅ 答案已保存到数据库: 班级={class_name}, 日期={date_str}")
    print(f"   内容: {text[:50]}{'...' if len(text) > 50 else ''}")


def run_pipeline(pdf_name: str):
    """直接对 incoming 目录中的 PDF 运行流水线。"""
    from monitor.processor import process_pdf

    incoming = ROOT / "data" / "incoming" / pdf_name
    if not incoming.exists():
        print(f"❌ 文件不存在: {incoming}")
        # 尝试从根目录复制
        src = ROOT / pdf_name
        if src.exists():
            import shutil
            shutil.copy2(str(src), str(incoming))
            print(f"📄 已从 {src} 复制到 {incoming}")
        else:
            print(f"❌ 源文件也不存在: {src}")
            return

    print(f"\n{'='*50}")
    print(f"🚀 开始处理: {pdf_name}")
    print(f"{'='*50}\n")

    process_pdf(str(incoming))

    print(f"\n{'='*50}")
    print(f"✅ 处理完成，请查看 data/output/ 目录")
    print(f"{'='*50}")


def show_results():
    """显示处理结果汇总。"""
    from db.crud import get_tasks
    tasks = get_tasks(limit=5)
    if not tasks:
        print("\n📭 数据库暂无任务记录")
        return

    print(f"\n{'='*50}")
    print("📊 最近批改任务")
    print(f"{'='*50}")
    for t in tasks:
        print(f"  #{t.id} | {t.filename}")
        print(f"       班级={t.class_name} 日期={t.date_str} 序号={t.seq}")
        print(f"       总字数={t.total_chars} ✅正确={t.correct_count} ⚠️存疑={t.uncertain_count} ❌错误={t.wrong_count}")

    # 检查输出文件
    output_dir = ROOT / "data" / "output"
    if output_dir.exists():
        pdfs = list(output_dir.glob("*_annotated.pdf"))
        if pdfs:
            print(f"\n📄 批注 PDF 已生成:")
            for p in pdfs:
                size = p.stat().st_size / 1024
                print(f"   {p.name} ({size:.1f} KB)")
        else:
            print(f"\n📄 output/ 目录无批注 PDF")


def main():
    parser = argparse.ArgumentParser(description="自动批改测试运行器")
    parser.add_argument("--seed-answers", "-s", type=str, default=None,
                        help="预置答案文本（如不提供，则从数据库读取）")
    parser.add_argument("--class-name", "-c", type=str, default="301",
                        help="班级 (默认: 301)")
    parser.add_argument("--date", "-d", type=str, default="2025-03-20",
                        help="日期 (默认: 2025-03-20)")
    parser.add_argument("--pdf", "-p", type=str, default="301_2025-03-20_001.pdf",
                        help="PDF 文件名 (默认: 301_2025-03-20_001.pdf)")
    parser.add_argument("--show", action="store_true",
                        help="仅显示结果，不运行流水线")

    args = parser.parse_args()

    if args.show:
        show_results()
        return

    # 预置答案（如果提供了 --seed-answers）
    if args.seed_answers:
        seed_answers(args.class_name, args.date, args.seed_answers)

    # 运行流水线
    run_pipeline(args.pdf)

    # 显示结果
    show_results()

    print(f"\n💡 提示: Web 管理界面已启动 → http://localhost:8501")
    print(f"   批注 PDF 保存在: data/output/")


if __name__ == "__main__":
    main()
