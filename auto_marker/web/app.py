"""
Streamlit 管理界面 — 统计看板、任务管理、答案管理、系统配置。
"""

import sys
import json
import configparser
from pathlib import Path

import pandas as pd
import streamlit as st

# 确保项目根在 sys.path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from db.crud import (
    get_tasks,
    get_task,
    save_answer,
    get_answer,
    get_all_answers,
    delete_answer,
    update_task_result,
    save_correction,
    get_corrections,
)

st.set_page_config(page_title="半自动批改台", layout="wide", page_icon="📝")
st.title("📝 半自动批改台")

# ============================================================
#  辅助函数
# ============================================================


def _fmt_pct(numerator: int, denominator: int) -> str:
    return f"{numerator / denominator:.1%}" if denominator else "-"


def _tasks_to_df(tasks) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "ID": t.id,
                "文件名": t.filename,
                "班级": t.class_name or "-",
                "日期": t.date_str or "-",
                "总字数": t.total_chars,
                "正确": t.correct_count,
                "存疑": t.uncertain_count,
                "错误": t.wrong_count,
                "正确率": t.correct_count / t.total_chars if t.total_chars else None,
                "正确率_显示": _fmt_pct(t.correct_count, t.total_chars),
                "置信度": f"{t.avg_confidence:.0%}" if t.avg_confidence else "-",
                "低置信占比": f"{t.low_conf_ratio:.0%}" if t.low_conf_ratio else "-",
                "耗时": f"{t.process_time:.1f}s" if t.process_time else "-",
                "复核": ("✅已复核" if t.review_done else ("🔍待复核" if t.needs_review else "-")),
            }
            for t in tasks
        ]
    )


def _color_status(val: str) -> str:
    if "correct" in val:
        return "background-color: #d4edda; color: #155724"
    if "uncertain" in val:
        return "background-color: #fff3cd; color: #856404"
    if "wrong" in val:
        return "background-color: #f8d7da; color: #721c24"
    return ""


def _style_results_table(df: pd.DataFrame) -> pd.DataFrame.style:
    """为结果表格添加条件颜色。"""
    return df.style.applymap(_color_status, subset=["状态"])


# ============================================================
#  Tab 1 — 统计看板
# ============================================================
tab1, tab2, tab3, tab4 = st.tabs(
    ["📊 统计看板", "📋 任务列表", "✏️ 答案管理", "⚙️ 系统配置"]
)

with tab1:
    tasks = get_tasks(limit=500)
    if not tasks:
        st.info("暂无任务数据")
    else:
        # ── 指标卡片 ──
        total_tasks = len(tasks)
        total_chars = sum(t.total_chars for t in tasks)
        total_correct = sum(t.correct_count for t in tasks)
        total_uncertain = sum(t.uncertain_count for t in tasks)
        total_wrong = sum(t.wrong_count for t in tasks)
        avg_rate = total_correct / total_chars if total_chars else 0
        needs_review_count = sum(1 for t in tasks if t.needs_review and not t.review_done)

        col_m1, col_m2, col_m3, col_m4, col_m5 = st.columns(5)
        col_m1.metric("📋 总任务数", total_tasks)
        col_m2.metric("📝 总批改字数", f"{total_chars:,}")
        col_m3.metric("✅ 平均正确率", f"{avg_rate:.1%}")
        col_m4.metric("⚠️ 总存疑数", total_uncertain)
        col_m5.metric("🔍 待复核", needs_review_count)

        st.divider()

        # ── 可观测性指标 ──
        confs = [t.avg_confidence for t in tasks if t.avg_confidence is not None]
        ratios = [t.low_conf_ratio for t in tasks if t.low_conf_ratio is not None]
        times = [t.process_time for t in tasks if t.process_time is not None]

        if confs or ratios or times:
            st.subheader("🔍 OCR 可观测性")
            col_o1, col_o2, col_o3 = st.columns(3)
            col_o1.metric("平均 OCR 置信度", f"{sum(confs)/len(confs):.1%}" if confs else "-")
            col_o2.metric("平均低置信度字占比", f"{sum(ratios)/len(ratios):.1%}" if ratios else "-")
            col_o3.metric("平均处理耗时", f"{sum(times)/len(times):.1f}s" if times else "-")

            st.divider()

        # ── 图表 ──
        col_c1, col_c2 = st.columns(2)

        with col_c1:
            st.subheader("批改结果分布")
            dist = pd.DataFrame(
                {
                    "类别": ["正确", "存疑", "错误"],
                    "数量": [total_correct, total_uncertain, total_wrong],
                }
            )
            st.bar_chart(dist.set_index("类别"), y="数量", use_container_width=True)

        with col_c2:
            st.subheader("任务正确率趋势")
            df = _tasks_to_df(tasks)
            trend = df.head(30).copy()
            trend["标签"] = trend.apply(
                lambda r: f"#{r['ID']} {r['班级']}", axis=1
            )
            st.line_chart(
                trend.set_index("标签")["正确率"],
                use_container_width=True,
            )

        st.divider()

        # ── 按日期汇总 ──
        st.subheader("按日期汇总")
        df["日期"] = df["日期"].replace("-", pd.NA)
        daily = (
            df.groupby("日期")
            .agg(任务数=("ID", "count"), 总字数=("总字数", "sum"), 正确=("正确", "sum"))
            .reset_index()
        )
        daily = daily[daily["日期"].notna()].sort_values("日期", ascending=False)
        if not daily.empty:
            st.dataframe(daily, use_container_width=True, hide_index=True)


# ============================================================
#  Tab 2 — 任务列表
# ============================================================
with tab2:
    st.subheader("批改任务")

    # ── 加载全部（用于筛选） ──
    all_tasks = get_tasks(limit=1000)
    if not all_tasks:
        st.info("暂无任务数据")
    else:
        classes = sorted({t.class_name for t in all_tasks if t.class_name})
        dates = sorted({t.date_str for t in all_tasks if t.date_str}, reverse=True)

        # ── 筛选栏 ──
        col_f1, col_f2, col_f3 = st.columns(3)
        with col_f1:
            fc = st.selectbox("班级", ["全部"] + classes, key="filter_class")
        with col_f2:
            fd = st.selectbox("日期", ["全部"] + dates, key="filter_date")
        with col_f3:
            fs = st.text_input("搜索文件名", placeholder="关键词…", key="filter_search")

        filtered = all_tasks
        if fc != "全部":
            filtered = [t for t in filtered if t.class_name == fc]
        if fd != "全部":
            filtered = [t for t in filtered if t.date_str == fd]
        if fs:
            filtered = [t for t in filtered if fs.lower() in t.filename.lower()]

        df = _tasks_to_df(filtered)

        # ── 摘要 ──
        t_all = sum(t.total_chars for t in filtered)
        t_ok = sum(t.correct_count for t in filtered)
        st.caption(
            f"共 {len(filtered)} 个任务　｜　{t_all} 个字　｜　"
            f"总体正确率 {_fmt_pct(t_ok, t_all)}"
        )

        # ── 任务表格 ──
        st.dataframe(
            df.drop(columns=["正确率_显示"]),
            use_container_width=True,
            hide_index=True,
            column_config={
                "正确率": st.column_config.ProgressColumn(
                    "正确率",
                    format=".0%",
                    min_value=0,
                    max_value=1,
                ),
            },
            key="task_list_table",
        )

        # ── 选择查看详情 ──
        st.divider()
        opts = {f"#{t.id} — {t.filename}": t.id for t in filtered}
        selected_label = st.selectbox(
            "选择任务查看详情", ["(请选择)"] + list(opts.keys()), key="task_selector"
        )

        if selected_label != "(请选择)":
            task_id = opts[selected_label]
            task = get_task(task_id)
            if not task:
                st.warning(f"任务 #{task_id} 不存在")
            else:
                # ── 详情头部 ──
                col_h1, col_h2, col_h3, col_h4 = st.columns(4)
                col_h1.write(f"**文件**: {task.filename}")
                col_h2.write(
                    f"**班级**: {task.class_name}　|　**日期**: {task.date_str}"
                )
                col_h3.write(f"**序号**: {task.seq or '-'}")
                col_h4.write(
                    f"✅ {task.correct_count}　⚠️ {task.uncertain_count}　"
                    f"❌ {task.wrong_count}"
                )

                # ── PDF 下载 ──
                if task.annotated_pdf:
                    pdf_path = Path(task.annotated_pdf)
                    if pdf_path.is_file():
                        pdf_bytes = pdf_path.read_bytes()
                        st.download_button(
                            label="📄 下载批注后 PDF",
                            data=pdf_bytes,
                            file_name=pdf_path.name,
                            mime="application/pdf",
                            use_container_width=True,
                        )

                # ── 结果浏览 ──
                if task.result_json:
                    results = (
                        task.result_json
                        if isinstance(task.result_json, list)
                        else json.loads(task.result_json)
                    )

                    st.divider()

                    # ── 可观测性指标展示 ──
                    if task.avg_confidence is not None:
                        col_v1, col_v2, col_v3, col_v4 = st.columns(4)
                        col_v1.metric("平均置信度", f"{task.avg_confidence:.1%}")
                        col_v2.metric("低置信度占比", f"{task.low_conf_ratio:.1%}")
                        col_v3.metric("处理耗时", f"{task.process_time:.1f}s" if task.process_time else "-")
                        if task.needs_review and not task.review_done:
                            col_v4.metric("状态", "🔍 需人工复核")
                        elif task.review_done:
                            col_v4.metric("状态", "✅ 已复核")
                        else:
                            col_v4.metric("状态", "✅ 质量正常")

                    st.divider()

                    col_r1, col_r2 = st.columns([1, 2])
                    with col_r1:
                        status_filter = st.radio(
                            "筛选状态",
                            ["全部", "correct", "uncertain", "wrong"],
                            format_func=lambda x: {
                                "全部": "全部",
                                "correct": "✅ 正确",
                                "uncertain": "⚠️ 存疑",
                                "wrong": "❌ 错误",
                            }[x],
                            horizontal=True,
                            key="status_filter",
                        )

                    pages = sorted({r["page"] for r in results})
                    with col_r2:
                        page_sel = st.selectbox(
                            "选择页面",
                            pages,
                            format_func=lambda p: f"第 {p+1} 页",
                            key="page_selector",
                        )

                    # ── 每页统计 ──
                    st.divider()
                    st.subheader("📊 每页统计")
                    page_stats_data = []
                    for p in pages:
                        p_results = [r for r in results if r["page"] == p]
                        total = len(p_results)
                        correct = sum(1 for r in p_results if r["status"] == "correct")
                        page_stats_data.append({
                            "页码": f"第 {p+1} 页",
                            "字数": total,
                            "正确": correct,
                            "正确率": _fmt_pct(correct, total),
                        })
                    st.dataframe(
                        pd.DataFrame(page_stats_data),
                        use_container_width=True,
                        hide_index=True,
                    )

                    # 过滤
                    page_results = [r for r in results if r["page"] == page_sel]
                    if status_filter != "全部":
                        page_results = [
                            r for r in page_results if r["status"] == status_filter
                        ]

                    if page_results:
                        emoji_map = {
                            "correct": "✅",
                            "uncertain": "⚠️",
                            "wrong": "❌",
                        }

                        # ── 人工复核模式 ──
                        st.divider()
                        review_mode = st.checkbox("开启人工复核模式", key="review_mode")
                        st.caption("勾选后可逐字修改状态，修改后点击「保存复核结果」")

                        if review_mode:
                            st.warning("复核模式已开启。修改下方下拉框后请点击保存按钮。")
                            modified = []
                            for i, r in enumerate(page_results):
                                # 找到在完整 results 中的索引
                                global_idx = results.index(r)
                                col_e1, col_e2, col_e3, col_e4 = st.columns([1, 1, 1, 2])
                                col_e1.write(f"**{r['char']}**")
                                col_e2.write(f"预期: {r['expected']}")
                                col_e3.write(f"置信度: {r['confidence']:.0%}")
                                new_status = col_e4.selectbox(
                                    f"状态 (#{global_idx})",
                                    ["correct", "uncertain", "wrong"],
                                    index=["correct", "uncertain", "wrong"].index(r["status"]),
                                    format_func=lambda x: emoji_map[x] + " " + x,
                                    key=f"review_{global_idx}",
                                )
                                if new_status != r["status"]:
                                    modified.append((global_idx, r, new_status))

                            if modified:
                                st.info(f"检测到 {len(modified)} 处修改")
                                if st.button("💾 保存复核结果", type="primary", use_container_width=True):
                                    # 更新 results
                                    for global_idx, orig_r, new_status in modified:
                                        old_status = orig_r["status"]
                                        results[global_idx]["status"] = new_status
                                        # 保存修正记录
                                        save_correction(
                                            task_id=task.id,
                                            char_index=global_idx,
                                            original_char=orig_r["char"],
                                            original_status=old_status,
                                            corrected_char=orig_r["char"],
                                            corrected_status=new_status,
                                            bbox_pixel=orig_r.get("bbox_pixel"),
                                            confidence=orig_r.get("confidence"),
                                        )
                                    # 更新任务
                                    update_task_result(task.id, results, review_done=True)
                                    st.success(f"已保存 {len(modified)} 处修改，任务 #{task.id} 标记为已复核")
                                    st.rerun()
                            else:
                                st.caption("暂无修改")
                        else:
                            # 普通浏览模式
                            detail_df = pd.DataFrame(
                                [
                                    {
                                        "字符": r["char"],
                                        "预期": r["expected"],
                                        "状态": f"{emoji_map.get(r['status'], '❓')} {r['status']}",
                                        "置信度": f"{r['confidence']:.0%}",
                                    }
                                    for r in page_results
                                ]
                            )
                            styled = _style_results_table(detail_df)
                            st.dataframe(
                                styled,
                                use_container_width=True,
                                hide_index=True,
                            )
                            st.caption(f"当前页匹配 {len(page_results)} 个字")
                    else:
                        st.info("该页面无匹配结果")


# ============================================================
#  Tab 3 — 答案管理
# ============================================================
with tab3:
    st.subheader("设置标准答案")

    col_a1, col_a2 = st.columns(2)
    with col_a1:
        cls_name = st.text_input("班级", value="301", key="ans_class")
    with col_a2:
        date_str = st.text_input("日期 (YYYY-MM-DD)", value="", key="ans_date")

    current = get_answer(cls_name if cls_name else None, date_str if date_str else None)
    answer_text = st.text_area(
        "答案内容（每行一个字/词，或连续文本）",
        value=current or "",
        height=200,
        placeholder="例如：春眠不觉晓处处闻啼鸟",
        key="ans_content",
    )

    if st.button("保存答案", type="primary", use_container_width=True):
        if answer_text.strip():
            save_answer(cls_name, date_str, answer_text.strip())
            st.success(f"答案已保存（班级={cls_name}, 日期={date_str})")
            st.rerun()
        else:
            st.error("答案不能为空")

    # ── 已有答案列表 ──
    st.divider()
    st.subheader("已有答案记录")

    all_answers = get_all_answers()
    if not all_answers:
        st.info("暂无答案记录")
    else:
        ans_df = pd.DataFrame(
            [
                {
                    "ID": a.id,
                    "班级": a.class_name or "-",
                    "日期": a.date_str or "-",
                    "内容预览": (a.content[:80] + "…") if len(a.content) > 80 else a.content,
                    "创建时间": a.created_at.strftime("%Y-%m-%d %H:%M") if a.created_at else "-",
                }
                for a in all_answers
            ]
        )
        st.dataframe(ans_df, use_container_width=True, hide_index=True)

        # 删除
        del_opts = {f"#{a.id} — {a.class_name or '-'} / {a.date_str or '-'}": a.id for a in all_answers}
        del_label = st.selectbox(
            "选择要删除的答案", ["(请选择)"] + list(del_opts.keys()), key="del_answer"
        )
        if del_label != "(请选择)":
            del_id = del_opts[del_label]
            if st.button(f"🗑️ 删除答案 #{del_id}", type="secondary", use_container_width=True):
                if delete_answer(del_id):
                    st.success(f"答案 #{del_id} 已删除")
                    st.rerun()
                else:
                    st.error("删除失败")


# ============================================================
#  Tab 4 — 系统配置（分字段表单）
# ============================================================
with tab4:
    st.subheader("系统配置")
    config_path = ROOT / "config.ini"
    if not config_path.exists():
        st.error("config.ini 不存在")
    else:
        # 读取
        cfg = configparser.ConfigParser()
        cfg.read(config_path, encoding="utf-8")

        edited = {}  # section → {key → value}

        # ── [ftp] ──
        with st.expander("📡 FTP 服务", expanded=False):
            if cfg.has_section("ftp"):
                host = st.text_input("host", value=cfg.get("ftp", "host", fallback="0.0.0.0"), key="cfg_ftp_host")
                port = st.number_input("port", min_value=1, max_value=65535, value=cfg.getint("ftp", "port", fallback=2121), key="cfg_ftp_port")
                ftp_dir = st.text_input("dir", value=cfg.get("ftp", "dir", fallback="./data/incoming"), key="cfg_ftp_dir")
                anon = st.checkbox("允许匿名登录", value=cfg.getboolean("ftp", "anon", fallback=True), key="cfg_ftp_anon")
                edited["ftp"] = {"host": host, "port": str(port), "dir": ftp_dir, "anon": "yes" if anon else "no"}

        # ── [monitor] ──
        with st.expander("👀 监控目录", expanded=False):
            if cfg.has_section("monitor"):
                m_in = st.text_input("incoming_dir", value=cfg.get("monitor", "incoming_dir", fallback="./data/incoming"), key="cfg_mon_in")
                m_work = st.text_input("working_dir", value=cfg.get("monitor", "working_dir", fallback="./data/working"), key="cfg_mon_work")
                m_out = st.text_input("output_dir", value=cfg.get("monitor", "output_dir", fallback="./data/output"), key="cfg_mon_out")
                m_bak = st.text_input("backup_dir", value=cfg.get("monitor", "backup_dir", fallback="./data/backup"), key="cfg_mon_bak")
                m_workers = st.number_input("max_workers", min_value=1, max_value=16, value=cfg.getint("monitor", "max_workers", fallback=4), key="cfg_mon_workers")
                m_timeout = st.number_input("file_stable_timeout (秒)", min_value=1, value=cfg.getint("monitor", "file_stable_timeout", fallback=15), key="cfg_mon_timeout")
                edited["monitor"] = {
                    "incoming_dir": m_in,
                    "working_dir": m_work,
                    "output_dir": m_out,
                    "backup_dir": m_bak,
                    "max_workers": str(m_workers),
                    "file_stable_timeout": str(m_timeout),
                }

        # ── [ocr] ──
        with st.expander("🔍 OCR 引擎", expanded=False):
            if cfg.has_section("ocr"):
                model = st.text_input("model", value=cfg.get("ocr", "model", fallback="ch_PP-OCRv4_mobile"), key="cfg_ocr_model")
                use_gpu = st.checkbox("使用 GPU", value=cfg.getboolean("ocr", "use_gpu", fallback=False), key="cfg_ocr_gpu")
                conf_thresh = st.slider("confidence_threshold", min_value=0.0, max_value=1.0, value=cfg.getfloat("ocr", "confidence_threshold", fallback=0.6), key="cfg_ocr_conf")
                edited["ocr"] = {"model": model, "use_gpu": "yes" if use_gpu else "no", "confidence_threshold": f"{conf_thresh:.2f}"}

        # ── [grader] ──
        with st.expander("📊 评分阈值", expanded=False):
            if cfg.has_section("grader"):
                green = st.slider("green_threshold（绿色正确线）", min_value=0.0, max_value=1.0, value=cfg.getfloat("grader", "green_threshold", fallback=0.85), key="cfg_green")
                orange = st.slider("orange_threshold（橙色存疑线）", min_value=0.0, max_value=1.0, value=cfg.getfloat("grader", "orange_threshold", fallback=0.60), key="cfg_orange")
                edited["grader"] = {"green_threshold": f"{green:.2f}", "orange_threshold": f"{orange:.2f}"}

        # ── [printer] ──
        with st.expander("🖨️ 打印设置", expanded=False):
            if cfg.has_section("printer"):
                p_name = st.text_input("打印机名称（留空=默认）", value=cfg.get("printer", "name", fallback=""), key="cfg_printer_name")
                p_viewer = st.text_input("PDF 查看器路径（留空=默认）", value=cfg.get("printer", "pdf_viewer", fallback=""), key="cfg_printer_viewer")
                edited["printer"] = {"name": p_name, "pdf_viewer": p_viewer}

        # ── [web] ──
        with st.expander("🌐 Web 服务", expanded=False):
            if cfg.has_section("web"):
                web_host = st.text_input("host", value=cfg.get("web", "host", fallback="0.0.0.0"), key="cfg_web_host")
                web_port = st.number_input("port", min_value=1, max_value=65535, value=cfg.getint("web", "port", fallback=8501), key="cfg_web_port")
                edited["web"] = {"host": web_host, "port": str(web_port)}

        # ── [log] ──
        with st.expander("📝 日志", expanded=False):
            if cfg.has_section("log"):
                log_level = st.selectbox("level", ["DEBUG", "INFO", "WARNING", "ERROR"], index=["DEBUG", "INFO", "WARNING", "ERROR"].index(cfg.get("log", "level", fallback="INFO")), key="cfg_log_level")
                log_dir = st.text_input("dir", value=cfg.get("log", "dir", fallback="./data/logs"), key="cfg_log_dir")
                edited["log"] = {"level": log_level, "dir": log_dir}

        # ── 保存 ──
        st.divider()
        if st.button("💾 保存全部配置", type="primary", use_container_width=True):
            for section, kv in edited.items():
                if not cfg.has_section(section):
                    cfg.add_section(section)
                for key, val in kv.items():
                    cfg.set(section, key, val)
            with open(config_path, "w", encoding="utf-8") as f:
                cfg.write(f)
            st.success("配置已保存，重启服务后生效")