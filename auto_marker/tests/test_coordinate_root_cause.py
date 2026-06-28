"""坐标错位根因分析测试 — 隔离模型/流水线/渲染三层。

对比三层数据，定位坐标错位根因:
  Layer 0 (Golden):     golden.json 人工标注坐标（像素）
  Layer 1 (Model Raw):  _parse_page_level_json 输出（归一化 → 像素，未经后处理）
  Layer 2 (Pipeline):   recognize_page_level 输出（像素，经 y-clamp + x-rescale）

对比矩阵:
  - Model Raw  vs Golden   → 模型精度（模型问题？）
  - Pipeline   vs Golden   → 流水线最终精度
  - Pipeline   vs Model Raw→ 流水线后处理效果（流水线问题？）

判定逻辑:
  - 若 Model Raw vs Golden 误差大 → 模型输出的坐标有问题
  - 若 Pipeline vs Model Raw 误差变大 → 流水线后处理有问题
  - 若 Pipeline vs Golden 误差小 → 流水线已修正模型误差
"""
import json
import logging
import sys
from pathlib import Path

import fitz
import numpy as np
import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

logging.basicConfig(level=logging.WARNING, format="%(name)s | %(message)s")

TEST_PDF = ROOT / "data" / "test_samples" / "301_2026-06-18_003.pdf"
GT_JSON = ROOT / "data" / "test_samples" / "301_2026-06-18_003_ground_truth.json"

# q_idx 归一化映射表（gt 和 pl 的 q_idx 定义不同，统一到 pl 定义）
GT_QIDX_MAP = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6,
               201: 7, 101: 200, 102: 201}
PL_QIDX_MAP = {}


def _normalize_qi(qi, mapping):
    return mapping.get(qi, qi)


# ============================================================
# Fixtures（class-scoped，只调用一次 API）
# ============================================================

@pytest.fixture(scope="class")
def img_array():
    """加载测试 PDF 第 1 页为 numpy 数组。"""
    from services.pipeline_service import OCR_DPI

    doc = fitz.open(str(TEST_PDF))
    page = doc[0]
    zoom = OCR_DPI / 72
    pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), colorspace=fitz.csRGB)
    img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
    doc.close()
    return np.array(img)


@pytest.fixture(scope="class")
def layout_data(img_array):
    """运行 OCR + 版面分析（用缓存）。"""
    from services.pipeline_service import OCR_DPI, _file_hash, _load_ocr_cache
    from core.layout_analyzer import analyze_layout

    img_h, img_w = img_array.shape[:2]
    fh = _file_hash(TEST_PDF)
    cached = _load_ocr_cache(fh, 0)
    assert cached is not None, "OCR 缓存未命中，请先运行一次完整流水线"
    det_boxes, ocr_records = cached
    layout = analyze_layout(det_boxes, img_h, 0, ocr_records=ocr_records)
    return {
        "hw_boxes": layout["handwriting_boxes"],
        "question_regions": layout["question_regions"].get(0, []),
        "img_w": img_w,
        "img_h": img_h,
    }


@pytest.fixture(scope="class")
def answers_raw():
    """加载标准答案文本。"""
    from db.crud import get_tasks
    from services.answer_service import AnswerService

    task = get_tasks(limit=1)[0]
    return AnswerService.get_answer(task.class_name, task.date_str) or ""


@pytest.fixture(scope="class")
def model_raw_response(img_array, answers_raw):
    """调用 Qwen3-VL API 获取原始响应文本（约 56 秒）。"""
    from core.qwen_vl_recognizer import _call_qwen_vl_page_level

    response = _call_qwen_vl_page_level(img_array, answers_raw)
    assert response, "Qwen3-VL API 无响应"
    return response


@pytest.fixture(scope="class")
def model_raw_chars(model_raw_response, layout_data):
    """Layer 1: 模型原始 JSON 输出（归一化坐标 → 像素坐标）。

    未经 y-clamp / x-rescale / 等宽切分等后处理。
    返回: [{q_idx, char, bbox_pixel, bbox_norm}]
    """
    from core.qwen_vl_recognizer import _parse_page_level_json, _question_marker_to_q_idx

    img_w = layout_data["img_w"]
    img_h = layout_data["img_h"]

    parsed = _parse_page_level_json(model_raw_response)
    assert parsed, "JSON 解析失败"

    results = []
    for item in parsed:
        if item.get("invalid"):
            continue
        q_marker = item.get("q_marker", "")
        q_idx = _question_marker_to_q_idx(q_marker)
        if q_idx is None:
            continue
        for ch in item.get("chars", []):
            x0, y0, x1, y1 = ch["bbox_norm"]
            # 归一化 → 像素
            px0 = int(round(x0 * img_w))
            py0 = int(round(y0 * img_h))
            px1 = int(round(x1 * img_w))
            py1 = int(round(y1 * img_h))
            results.append({
                "q_idx": q_idx,
                "char": ch["char"],
                "bbox_norm": ch["bbox_norm"],
                "bbox_pixel": (px0, py0, px1, py1),
            })
    return results


@pytest.fixture(scope="class")
def pipeline_chars(img_array, layout_data, answers_raw):
    """Layer 2: 流水线后处理后输出（像素坐标）。

    经 y-clamp + x-rescale + 等宽切分回退等完整后处理。
    """
    from core.qwen_vl_recognizer import recognize_page_level

    results = recognize_page_level(
        img_array,
        layout_data["question_regions"],
        0,
        handwriting_boxes=layout_data["hw_boxes"],
        answers_text=answers_raw,
    )
    assert results, "流水线无输出"
    return results


@pytest.fixture(scope="class")
def golden_chars():
    """Layer 0: golden.json 人工标注（像素坐标）。"""
    with open(GT_JSON, "r", encoding="utf-8") as f:
        return json.load(f)


# ============================================================
# 辅助函数
# ============================================================

def _group_by_q(items, qidx_key="q_idx", mapping=None):
    """按归一化 q_idx 分组，每组按 x 中心排序。"""
    by_q = {}
    for item in items:
        qi = _normalize_qi(item[qidx_key], mapping) if mapping else item[qidx_key]
        by_q.setdefault(qi, []).append(item)
    for items_list in by_q.values():
        bbox_key = "bbox_pixel" if "bbox_pixel" in items_list[0] else None
        if bbox_key:
            bbox = items_list[0][bbox_key]
            if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
                items_list.sort(key=lambda r: (r[bbox_key][0] + r[bbox_key][2]) / 2)
    return by_q


def _match_pairs(list_a, list_b, qidx_key_a, qidx_key_b, mapping_a=None, mapping_b=None):
    """按归一化 q_idx + x 顺序匹配两个列表，返回 [{char, a_cx, a_cy, b_cx, b_cy, dx, dy}]。"""
    a_by_q = _group_by_q(list_a, qidx_key_a, mapping_a)
    b_by_q = _group_by_q(list_b, qidx_key_b, mapping_b)

    pairs = []
    for qi in sorted(set(a_by_q.keys()) & set(b_by_q.keys())):
        a_items = a_by_q[qi]
        b_items = b_by_q[qi]
        n = min(len(a_items), len(b_items))
        for i in range(n):
            a = a_items[i]
            b = b_items[i]
            a_bbox = a.get("bbox_pixel") or a.get("center_pixel")
            b_bbox = b.get("bbox_pixel") or b.get("center_pixel")

            if "center_pixel" in a:
                a_cx, a_cy = a["center_pixel"]
            else:
                a_cx = (a_bbox[0] + a_bbox[2]) / 2
                a_cy = (a_bbox[1] + a_bbox[3]) / 2

            if "center_pixel" in b:
                b_cx, b_cy = b["center_pixel"]
            else:
                b_cx = (b_bbox[0] + b_bbox[2]) / 2
                b_cy = (b_bbox[1] + b_bbox[3]) / 2

            pairs.append({
                "q_idx": qi,
                "char": a.get("char", b.get("char", "?")),
                "a_cx": a_cx, "a_cy": a_cy,
                "b_cx": b_cx, "b_cy": b_cy,
                "dx": b_cx - a_cx,
                "dy": b_cy - a_cy,
            })
    return pairs


def _error_stats(pairs):
    """计算误差统计。"""
    if not pairs:
        return {"avg": 0, "max": 0, "n": 0}
    dys = [abs(p["dy"]) for p in pairs]
    dxs = [abs(p["dx"]) for p in pairs]
    return {
        "n": len(pairs),
        "avg_dy": sum(dys) / len(dys),
        "max_dy": max(dys),
        "avg_dx": sum(dxs) / len(dxs),
        "max_dx": max(dxs),
    }


# ============================================================
# 测试用例
# ============================================================

class TestCoordinateRootCause:
    """坐标错位根因分析 — 三层对比。"""

    def test_model_raw_vs_golden(self, model_raw_chars, golden_chars):
        """Layer 1 vs Layer 0: 模型原始坐标精度。

        判定: 若误差大 → 模型输出的坐标有问题
        """
        pairs = _match_pairs(
            golden_chars, model_raw_chars,
            qidx_key_a="q_idx", qidx_key_b="q_idx",
            mapping_a=GT_QIDX_MAP, mapping_b=PL_QIDX_MAP,
        )
        assert len(pairs) >= 40, f"匹配字符太少: {len(pairs)}"

        stats = _error_stats(pairs)
        print(f"\n=== Model Raw vs Golden (模型原始精度) ===")
        print(f"  n={stats['n']}, avg|Δy|={stats['avg_dy']:.1f}px, max|Δy|={stats['max_dy']:.1f}px")
        print(f"  avg|Δx|={stats['avg_dx']:.1f}px, max|Δx|={stats['max_dx']:.1f}px")

        # 按 q_idx 分组打印
        by_q = {}
        for p in pairs:
            by_q.setdefault(p["q_idx"], []).append(p)
        print(f"  {'q_idx':>5} {'n':>3} {'avg|Δy|':>8} {'max|Δy|':>8} {'avg|Δx|':>8}")
        for qi in sorted(by_q.keys()):
            ps = by_q[qi]
            dys = [abs(p["dy"]) for p in ps]
            dxs = [abs(p["dx"]) for p in ps]
            print(f"  {qi:>5} {len(ps):>3} {sum(dys)/len(dys):>8.1f} {max(dys):>8.1f} {sum(dxs)/len(dxs):>8.1f}")

        # 模型原始坐标误差大是预期内的（已知 y 系统性偏移），
        # 这正是流水线 y-clamp 要修正的目标。此处只记录不 fail。
        # test_root_cause_summary 会断言"流水线有效修正了模型误差"。
        print(f"  → 模型原始 Y 误差 {stats['avg_dy']:.1f}px（预期 > 100px，流水线需修正）")

    def test_pipeline_vs_golden(self, pipeline_chars, golden_chars):
        """Layer 2 vs Layer 0: 流水线最终精度。"""
        pairs = _match_pairs(
            golden_chars, pipeline_chars,
            qidx_key_a="q_idx", qidx_key_b="question_idx",
            mapping_a=GT_QIDX_MAP, mapping_b=PL_QIDX_MAP,
        )
        assert len(pairs) >= 40, f"匹配字符太少: {len(pairs)}"

        stats = _error_stats(pairs)
        print(f"\n=== Pipeline vs Golden (流水线最终精度) ===")
        print(f"  n={stats['n']}, avg|Δy|={stats['avg_dy']:.1f}px, max|Δy|={stats['max_dy']:.1f}px")
        print(f"  avg|Δx|={stats['avg_dx']:.1f}px, max|Δx|={stats['max_dx']:.1f}px")

        by_q = {}
        for p in pairs:
            by_q.setdefault(p["q_idx"], []).append(p)
        print(f"  {'q_idx':>5} {'n':>3} {'avg|Δy|':>8} {'max|Δy|':>8} {'avg|Δx|':>8}")
        for qi in sorted(by_q.keys()):
            ps = by_q[qi]
            dys = [abs(p["dy"]) for p in ps]
            dxs = [abs(p["dx"]) for p in ps]
            print(f"  {qi:>5} {len(ps):>3} {sum(dys)/len(dys):>8.1f} {max(dys):>8.1f} {sum(dxs)/len(dxs):>8.1f}")

    def test_pipeline_vs_model_raw(self, pipeline_chars, model_raw_chars):
        """Layer 2 vs Layer 1: 流水线后处理的增量效果。

        判定:
          - 若 |Δ| 变小 → 流水线修正了模型误差（好）
          - 若 |Δ| 变大 → 流水线后处理有问题（坏）
          - 若 |Δ| ≈ 0 → 流水线未改变坐标
        """
        pairs = _match_pairs(
            model_raw_chars, pipeline_chars,
            qidx_key_a="q_idx", qidx_key_b="question_idx",
            mapping_a=PL_QIDX_MAP, mapping_b=PL_QIDX_MAP,
        )
        assert len(pairs) >= 30, f"匹配字符太少: {len(pairs)}"

        stats = _error_stats(pairs)
        print(f"\n=== Pipeline vs Model Raw (后处理增量效果) ===")
        print(f"  n={stats['n']}, avg|Δy|={stats['avg_dy']:.1f}px, max|Δy|={stats['max_dy']:.1f}px")
        print(f"  avg|Δx|={stats['avg_dx']:.1f}px, max|Δx|={stats['max_dx']:.1f}px")
        print(f"  (Δ = pipeline - model_raw，正值=后处理把坐标往下/右移)")

        by_q = {}
        for p in pairs:
            by_q.setdefault(p["q_idx"], []).append(p)
        print(f"  {'q_idx':>5} {'n':>3} {'avg|Δy|':>8} {'max|Δy|':>8} {'avg|Δx|':>8}")
        for qi in sorted(by_q.keys()):
            ps = by_q[qi]
            dys = [abs(p["dy"]) for p in ps]
            dxs = [abs(p["dx"]) for p in ps]
            print(f"  {qi:>5} {len(ps):>3} {sum(dys)/len(dys):>8.1f} {max(dys):>8.1f} {sum(dxs)/len(dxs):>8.1f}")

    def test_root_cause_summary(self, model_raw_chars, pipeline_chars, golden_chars):
        """根因汇总: 对比三层误差，输出结论。

        判定逻辑:
          - model_raw_error > pipeline_error → 流水线在修正模型误差（正常）
          - model_raw_error < pipeline_error → 流水线在引入误差（有问题）
          - model_raw_error ≈ pipeline_error → 流水线未改变坐标
        """
        pairs_model = _match_pairs(
            golden_chars, model_raw_chars,
            qidx_key_a="q_idx", qidx_key_b="q_idx",
            mapping_a=GT_QIDX_MAP, mapping_b=PL_QIDX_MAP,
        )
        pairs_pipeline = _match_pairs(
            golden_chars, pipeline_chars,
            qidx_key_a="q_idx", qidx_key_b="question_idx",
            mapping_a=GT_QIDX_MAP, mapping_b=PL_QIDX_MAP,
        )

        model_stats = _error_stats(pairs_model)
        pipe_stats = _error_stats(pairs_pipeline)

        print(f"\n{'='*60}")
        print(f"根因分析汇总")
        print(f"{'='*60}")
        print(f"  Model Raw  vs Golden: avg|Δy|={model_stats['avg_dy']:.1f}px, max|Δy|={model_stats['max_dy']:.1f}px")
        print(f"  Pipeline   vs Golden: avg|Δy|={pipe_stats['avg_dy']:.1f}px, max|Δy|={pipe_stats['max_dy']:.1f}px")
        print(f"  改善量: avg|Δy| {model_stats['avg_dy']:.1f} → {pipe_stats['avg_dy']:.1f}px "
              f"({(1 - pipe_stats['avg_dy']/model_stats['avg_dy'])*100:.0f}%)")

        if model_stats["avg_dy"] > pipe_stats["avg_dy"] * 1.5:
            conclusion = "流水线有效修正了模型 Y 误差（模型是主要误差源）"
        elif pipe_stats["avg_dy"] > model_stats["avg_dy"] * 1.5:
            conclusion = "流水线后处理引入了 Y 误差（流水线是主要误差源）"
        else:
            conclusion = "流水线对 Y 误差影响不大（误差源在其他环节）"
        print(f"  结论: {conclusion}")

        if model_stats["avg_dx"] > pipe_stats["avg_dx"] * 1.5:
            conclusion_x = "流水线有效修正了模型 X 误差"
        elif pipe_stats["avg_dx"] > model_stats["avg_dx"] * 1.5:
            conclusion_x = "流水线后处理引入了 X 误差"
        else:
            conclusion_x = "流水线对 X 误差影响不大"
        print(f"  X 结论: {conclusion_x}")
        print(f"{'='*60}")

        # 断言: 流水线应有效改善 Y 误差（至少改善 50%）
        improvement = 1 - pipe_stats["avg_dy"] / model_stats["avg_dy"]
        assert improvement > 0.5, (
            f"流水线 Y 误差改善不足: {improvement*100:.0f}% "
            f"(model={model_stats['avg_dy']:.1f} → pipeline={pipe_stats['avg_dy']:.1f})"
        )
        # 断言: 流水线最终 Y 误差应在可接受范围内（< 100px ≈ 36pt）
        assert pipe_stats["avg_dy"] < 100, (
            f"流水线 Y 误差仍过大: {pipe_stats['avg_dy']:.1f}px"
        )
