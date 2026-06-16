"""
Streamlit 管理界面 — 查看任务、管理答案、复核存疑项。
"""

import sys
import json
from pathlib import Path

import streamlit as st

# 确保项目根在 sys.path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from db.crud import get_tasks, get_task, save_answer, get_answer

st.set_page_config(page_title="半自动批改台", layout="wide")
st.title("📝 半自动批改台")

tab1, tab2, tab3 = st.tabs(["📋 任务列表", "✏️ 答案管理", "⚙️ 配置"])


# ========== 任务列表 ==========
with tab1:
    st.subheader("最近批改任务")

    tasks = get_tasks(limit=100)
    if not tasks:
        st.info("暂无任务数据")
    else:
        data = []
        for t in tasks:
            data.append({
                "ID": t.id,
                "文件名": t.filename,
                "班级": t.class_name or "-",
                "日期": t.date_str or "-",
                "总字数": t.total_chars,
                "✅ 正确": t.correct_count,
                "⚠️ 存疑": t.uncertain_count,
                "❌ 错误": t.wrong_count,
            })
        st.dataframe(data, use_container_width=True, hide_index=True,
                     column_order=["ID", "文件名", "班级", "日期",
                                   "总字数", "✅ 正确", "⚠️ 存疑", "❌ 错误"])

        st.divider()
        task_id = st.number_input("查看任务详情 (ID)", min_value=1, step=1)
        if task_id:
            task = get_task(task_id)
            if task:
                st.write(f"**文件**: {task.filename}")
                st.write(f"**班级**: {task.class_name} | **日期**: {task.date_str} | **序号**: {task.seq}")
                st.write(f"**正确**: {task.correct_count} | **存疑**: {task.uncertain_count} | **错误**: {task.wrong_count}")

                if task.result_json:
                    results = task.result_json if isinstance(task.result_json, list) else json.loads(task.result_json)
                    # 按页分组
                    pages = sorted(set(r["page"] for r in results))
                    page_sel = st.selectbox("选择页", pages, format_func=lambda p: f"第 {p+1} 页")
                    page_results = [r for r in results if r["page"] == page_sel]

                    st.write(f"第 {page_sel+1} 页，共 {len(page_results)} 个字")
                    table = []
                    for r in page_results:
                        emoji = {"correct": "✅", "uncertain": "⚠️", "wrong": "❌"}.get(r["status"], "❓")
                        table.append({
                            "字符": r["char"],
                            "预期": r["expected"],
                            "状态": f"{emoji} {r['status']}",
                            "置信度": f"{r['confidence']:.0%}",
                        })
                    st.dataframe(table, use_container_width=True, hide_index=True)
            else:
                st.warning(f"未找到 ID={task_id} 的任务")


# ========== 答案管理 ==========
with tab2:
    st.subheader("设置标准答案")

    col1, col2 = st.columns(2)
    with col1:
        cls_name = st.text_input("班级", value="301")
    with col2:
        date_str = st.text_input("日期 (YYYY-MM-DD)", value="")

    current = get_answer(cls_name if cls_name else None, date_str if date_str else None)
    answer_text = st.text_area("答案内容（每行一个字/词，或连续文本）",
                               value=current or "",
                               height=200,
                               placeholder="例如：春眠不觉晓处处闻啼鸟")

    if st.button("保存答案"):
        if answer_text.strip():
            save_answer(cls_name, date_str, answer_text.strip())
            st.success(f"答案已保存（班级={cls_name}, 日期={date_str})")
            st.rerun()
        else:
            st.error("答案不能为空")


# ========== 配置 ==========
with tab3:
    st.subheader("系统配置")

    config_path = ROOT / "config.ini"
    if config_path.exists():
        content = config_path.read_text(encoding="utf-8")
        edited = st.text_area("config.ini", content, height=400, font="monospace")
        if st.button("保存配置"):
            config_path.write_text(edited, encoding="utf-8")
            st.success("配置已保存，重启服务生效")
    else:
        st.error("config.ini 不存在")
