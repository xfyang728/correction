"""端到端集成测试 — 文字识别和标注到原位。

针对 data/test_samples/301_2026-06-18_003.pdf 进行完整流水线测试：
  1. Qwen3-VL 整页识别
  2. 坐标映射（y-clamp 修正）
  3. 文字渲染批注到原位

前置条件：
  - Qwen3-VL llama.cpp server 运行在 localhost:8080
  - PaddleOCR 模型已下载
  - 测试 PDF 存在于 data/test_samples/

运行方式：
  python -m pytest tests/test_e2e_annotation.py -v -s
"""

import json
import shutil
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

TEST_PDF = ROOT / "data" / "test_samples" / "301_2026-06-18_003.pdf"
INCOMING_PDF = ROOT / "data" / "incoming" / "301_2026-06-18_003.pdf"
ANNOTATED_PDF = ROOT / "data" / "output" / "301_2026-06-18_003_annotated.pdf"
GOLDEN_PDF = ROOT / "data" / "output" / "301_2026-06-18_003_golden.pdf"
GT_JSON = ROOT / "data" / "test_samples" / "301_2026-06-18_003_ground_truth.json"

# 预期结果（基于历史运行数据）
EXPECTED_QUESTIONS = 10
EXPECTED_CHARS_MIN = 50  # 至少 50 字（历史 57）
EXPECTED_Q_INDICES = {0, 1, 2, 3, 4, 5, 6, 101, 201, 104}  # 题1-7, ①, ②, 下联

# 预期各题 y 范围（基于布局分析，允许 ±60px 容差）
EXPECTED_Y_RANGES = {
    0: (700, 900),    # 题1 左列第1行
    1: (650, 900),    # 题2 右列第1行
    2: (830, 1050),   # 题3 左列第2行
    3: (830, 1050),   # 题4 右列第2行
    4: (950, 1150),   # 题5 左列第3行
    5: (950, 1250),   # 题6 右列第3行
    6: (1080, 1300),  # 题7 左列第4行
    101: (1200, 1450), # ①
    201: (1250, 1700), # ②
    104: (1800, 2050), # 下联
}


def _check_qwen_vl_available() -> bool:
    """检查 Qwen3-VL 服务是否可用。"""
    try:
        import requests
        resp = requests.get("http://localhost:8080/v1/models", timeout=3)
        return resp.status_code == 200
    except Exception:
        return False


def _check_test_pdf_exists() -> bool:
    """检查测试 PDF 是否存在。"""
    return TEST_PDF.exists()


# 跳过条件
pytestmark = pytest.mark.skipif(
    not _check_qwen_vl_available() or not _check_test_pdf_exists(),
    reason="Qwen3-VL 服务不可用或测试 PDF 不存在",
)


@pytest.fixture(scope="module")
def pipeline_result():
    """运行完整流水线，返回数据库中最新任务的结果。

    这是整个测试模块共享的 fixture，只运行一次流水线（约 70 秒）。
    """
    # 确保答案存在
    from db.crud import save_answer
    save_answer("301", "2026-06-18", (
        "（1）随君直到夜郎西\n"
        "（2）海内存知己\n"
        "（3）百般红紫斗芳菲\n"
        "（4）水中藻荇交横\n"
        "（5）人生自古谁无死\n"
        "（6）半竿斜日旧关城\n"
        "（7）采菊东篱下\n"
        "（8）悠然见南山\n"
        "① 质朴；② 绚丽\n"
        "下联：平凡物见证诉说追梦情"
    ))

    # 准备 PDF
    INCOMING_PDF.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(str(TEST_PDF), str(INCOMING_PDF))

    # 清理旧输出
    if ANNOTATED_PDF.exists():
        ANNOTATED_PDF.unlink()

    # 运行流水线
    from monitor.processor import process_pdf
    process_pdf(str(INCOMING_PDF))

    # 从数据库获取最新任务
    from db.crud import get_tasks
    tasks = get_tasks(limit=1)
    assert tasks, "流水线未产生任务"
    return tasks[0]


class TestPipelineOutput:
    """验证流水线基本输出。"""

    def test_task_created(self, pipeline_result):
        """任务已保存到数据库。"""
        assert pipeline_result.filename == "301_2026-06-18_003.pdf"
        assert pipeline_result.total_chars >= EXPECTED_CHARS_MIN, \
            f"字符数 {pipeline_result.total_chars} < {EXPECTED_CHARS_MIN}"

    def test_annotated_pdf_generated(self, pipeline_result):
        """批注 PDF 已生成且大小合理。"""
        assert ANNOTATED_PDF.exists(), "批注 PDF 不存在"
        size_kb = ANNOTATED_PDF.stat().st_size / 1024
        assert size_kb > 100, f"批注 PDF 太小: {size_kb:.1f} KB"

    def test_result_json_not_empty(self, pipeline_result):
        """result_json 包含完整的逐字结果。"""
        results = pipeline_result.result_json
        assert results is not None
        assert len(results) >= EXPECTED_CHARS_MIN


class TestQuestionMatching:
    """验证题目匹配完整性。"""

    def test_all_questions_matched(self, pipeline_result):
        """10 道题全部匹配（题1-7, ①, ②, 下联）。"""
        results = pipeline_result.result_json
        q_indices = set(r.get("question_idx", -1) for r in results)
        matched = q_indices & EXPECTED_Q_INDICES
        missing = EXPECTED_Q_INDICES - q_indices
        assert len(matched) >= EXPECTED_QUESTIONS, \
            f"只匹配了 {len(matched)} 题，缺失: {missing}"

    def test_no_unexpected_q_idx(self, pipeline_result):
        """不应有预期之外的 q_idx（排除噪声区域）。"""
        results = pipeline_result.result_json
        q_indices = set(r.get("question_idx", -1) for r in results)
        unexpected = q_indices - EXPECTED_Q_INDICES
        # 允许少量意外匹配，但不应太多
        assert len(unexpected) <= 2, \
            f"意外 q_idx 过多: {unexpected}"


class TestCoordinateMapping:
    """验证坐标映射 — 文字标注到原位的核心。"""

    def test_y_coordinates_distinct_per_question(self, pipeline_result):
        """每道题的 y 坐标范围不同（y-clamp 修正有效）。"""
        results = pipeline_result.result_json

        # 按 q_idx 分组，提取 y 范围
        y_ranges = {}
        for r in results:
            qi = r.get("question_idx", -1)
            bbox = r.get("bbox_pixel", (0, 0, 0, 0))
            if isinstance(bbox, (str, list)):
                bbox = tuple(json.loads(bbox)) if isinstance(bbox, str) else tuple(bbox)
            if qi not in y_ranges:
                y_ranges[qi] = [bbox[1], bbox[3]]
            else:
                y_ranges[qi][0] = min(y_ranges[qi][0], bbox[1])
                y_ranges[qi][1] = max(y_ranges[qi][1], bbox[3])

        # 检查每道题的 y 范围是否在预期区域内
        for qi, (y0, y1) in y_ranges.items():
            if qi in EXPECTED_Y_RANGES:
                exp_y0, exp_y1 = EXPECTED_Y_RANGES[qi]
                y_center = (y0 + y1) / 2
                assert exp_y0 <= y_center <= exp_y1, \
                    f"q_idx={qi} y_center={y_center} 不在预期范围 [{exp_y0}, {exp_y1}]"

    def test_no_shared_y_between_different_rows(self, pipeline_result):
        """不同行的题目不应共享完全相同的 y 范围（修复前的 bug）。"""
        results = pipeline_result.result_json

        # 收集每题的 y 中心
        y_centers = {}
        for r in results:
            qi = r.get("question_idx", -1)
            bbox = r.get("bbox_pixel", (0, 0, 0, 0))
            if isinstance(bbox, (str, list)):
                bbox = tuple(json.loads(bbox)) if isinstance(bbox, str) else tuple(bbox)
            if qi not in y_centers:
                y_centers[qi] = (bbox[1] + bbox[3]) / 2

        # 左列题目（q_idx 0,2,4,6）应在不同行
        left_col = [0, 2, 4, 6]
        left_ys = [y_centers[qi] for qi in left_col if qi in y_centers]
        if len(left_ys) >= 2:
            unique_ys = len(set(round(y, 1) for y in left_ys))
            assert unique_ys >= 3, \
                f"左列题目 y 中心不够多样: {left_ys}（应至少 3 个不同值）"

        # 右列题目（q_idx 1,3,5）应在不同行
        right_col = [1, 3, 5]
        right_ys = [y_centers[qi] for qi in right_col if qi in y_centers]
        if len(right_ys) >= 2:
            unique_ys = len(set(round(y, 1) for y in right_ys))
            assert unique_ys >= 2, \
                f"右列题目 y 中心不够多样: {right_ys}（应至少 2 个不同值）"

    def test_x_coordinates_preserved(self, pipeline_result):
        """x 坐标保留模型原始值（未被 clamp 修改）。

        注：A4 横向排版下，左列题目的长答案会从中线左侧延伸到右半页
        （如题1"随君直到夜郎西"7字从 x=281 延伸到 x=1058），
        因此用首字 x_center 判断起始列，而非整道题 x_center。
        """
        results = pipeline_result.result_json

        # 按 q_idx 分组，取首字 bbox
        from collections import defaultdict
        first_chars = {}
        for r in results:
            qi = r.get("question_idx", -1)
            if qi not in first_chars:
                first_chars[qi] = r

        for qi, r in first_chars.items():
            bbox = r.get("bbox_pixel", (0, 0, 0, 0))
            if isinstance(bbox, (str, list)):
                bbox = tuple(json.loads(bbox)) if isinstance(bbox, str) else tuple(bbox)
            img_w = r.get("img_pixel_w", 1653)
            x_center = (bbox[0] + bbox[2]) / 2
            page_mid = img_w / 2

            if qi in (0, 2, 4, 6):  # 左列起始
                assert x_center < page_mid, \
                    f"q_idx={qi} 首字 x_center={x_center} 应在左半页 (< {page_mid})"
            elif qi in (1, 3, 5):  # 右列起始
                assert x_center > page_mid, \
                    f"q_idx={qi} 首字 x_center={x_center} 应在右半页 (> {page_mid})"

    def test_bbox_within_image_bounds(self, pipeline_result):
        """所有 bbox 在图像边界内。"""
        results = pipeline_result.result_json
        for r in results:
            bbox = r.get("bbox_pixel", (0, 0, 0, 0))
            if isinstance(bbox, (str, list)):
                bbox = tuple(json.loads(bbox)) if isinstance(bbox, str) else tuple(bbox)
            img_w = r.get("img_pixel_w", 1653)
            img_h = r.get("img_pixel_h", 2312)
            x0, y0, x1, y1 = bbox
            assert 0 <= x0 <= img_w, f"x0={x0} 越界 (img_w={img_w})"
            assert 0 <= x1 <= img_w, f"x1={x1} 越界 (img_w={img_w})"
            assert 0 <= y0 <= img_h, f"y0={y0} 越界 (img_h={img_h})"
            assert 0 <= y1 <= img_h, f"y1={y1} 越界 (img_h={img_h})"


class TestRecognitionAccuracy:
    """验证识别准确性。"""

    def test_known_characters_present(self, pipeline_result):
        """已知字符应出现在识别结果中。"""
        results = pipeline_result.result_json
        chars = set(r.get("char", "") for r in results)

        # 题1 前 3 字
        for ch in "随君直":
            assert ch in chars, f"题1 字符 '{ch}' 未识别到"

        # ① 题
        for ch in "质朴":
            assert ch in chars, f"① 字符 '{ch}' 未识别到"

        # 下联首字
        assert "平" in chars, "下联首字 '平' 未识别到"

    def test_correct_count_reasonable(self, pipeline_result):
        """正确字数应占总字数的 60% 以上。"""
        total = pipeline_result.total_chars
        correct = pipeline_result.correct_count
        ratio = correct / total if total > 0 else 0
        assert ratio >= 0.6, f"正确率 {ratio:.1%} < 60%"


class TestAnnotationModes:
    """验证批注模式配置。"""

    def test_render_text_mode_enabled(self):
        """RENDER_TEXT_MODE 应为 True（文字渲染模式）。"""
        from core.recognition_config import RENDER_TEXT_MODE
        assert RENDER_TEXT_MODE is True, "RENDER_TEXT_MODE 应为 True"

    def test_page_level_engine(self):
        """识别引擎应为 page_level。"""
        from core.recognition_config import RECOGNITION_ENGINE
        assert RECOGNITION_ENGINE == "page_level", \
            f"RECOGNITION_ENGINE={RECOGNITION_ENGINE}，应为 page_level"


class TestAgainstGolden:
    """对照人工标注的黄金标准，量化坐标映射误差。

    黄金标准来源: data/test_samples/301_2026-06-18_003-人工标注box.pdf
        人工标注 53 个 Square 注释 (题1-8 + ①②，未含下联)
    生成脚本: tools/_build_golden_pdf.py

    验收阈值 (基于首次对比基线，留出 10% 改进空间):
        X 误差 p90 < 420px (基线 380px)
        Y 误差 p90 < 180px (基线 163px，含 q_idx=201 异常)
        误差 < 1 字符高 (101px) 的占比 ≥ 30% (基线 32%)
    未来优化目标: 阈值逐步收紧，最终 X/Y p90 < 100px, 占比 > 80%
    """

    @pytest.fixture(scope="module")
    def gt_data(self):
        """加载 ground truth JSON。"""
        if not GT_JSON.exists():
            pytest.skip(f"黄金标准 JSON 不存在: {GT_JSON}，请先运行 tools/_build_golden_pdf.py")
        with open(GT_JSON, "r", encoding="utf-8") as f:
            return json.load(f)

    def _match_pipeline_to_gt(self, pipeline_results, gt_data):
        """按 (q_idx, x 中心升序) 匹配流水线结果到 gt。"""
        # 按 q_idx 分组
        pl_by_q = {}
        for r in pipeline_results:
            qi = r.get("question_idx", -1)
            pl_by_q.setdefault(qi, []).append(r)
        for items in pl_by_q.values():
            items.sort(key=lambda r: (r["bbox_pixel"][0] + r["bbox_pixel"][2]) / 2)

        gt_by_q = {}
        for gt in gt_data:
            gt_by_q.setdefault(gt["q_idx"], []).append(gt)

        # 匹配
        pairs = []
        for qi in sorted(set(gt_by_q.keys()) & set(pl_by_q.keys())):
            gt_items = gt_by_q[qi]
            pl_items = pl_by_q[qi]
            n = min(len(gt_items), len(pl_items))
            for i in range(n):
                gt = gt_items[i]
                pl = pl_items[i]
                gt_cx, gt_cy = gt["center_pixel"]
                pl_bbox = pl["bbox_pixel"]
                pl_cx = (pl_bbox[0] + pl_bbox[2]) / 2
                pl_cy = (pl_bbox[1] + pl_bbox[3]) / 2
                pairs.append({
                    "q_idx": qi,
                    "char": gt["char"],
                    "gt_x": gt_cx, "gt_y": gt_cy,
                    "pl_x": pl_cx, "pl_y": pl_cy,
                    "dx": pl_cx - gt_cx,
                    "dy": pl_cy - gt_cy,
                })
        return pairs

    def test_golden_pdf_exists(self):
        """黄金标准 PDF 应已生成。"""
        if not GOLDEN_PDF.exists():
            pytest.skip(f"黄金标准 PDF 不存在: {GOLDEN_PDF}，请先运行 tools/_build_golden_pdf.py")

    def test_x_error_within_threshold(self, pipeline_result, gt_data):
        """X 坐标误差 p90 < 420px。"""
        pairs = self._match_pipeline_to_gt(pipeline_result.result_json, gt_data)
        assert len(pairs) >= 40, f"匹配字数过少: {len(pairs)}"

        x_errors = sorted(abs(p["dx"]) for p in pairs)
        p90 = x_errors[int(len(x_errors) * 0.9)]
        assert p90 < 420, \
            f"X 误差 p90={p90:.1f}px 超过阈值 420px"

    def test_y_error_within_threshold(self, pipeline_result, gt_data):
        """Y 坐标误差 p90 < 180px。"""
        pairs = self._match_pipeline_to_gt(pipeline_result.result_json, gt_data)
        assert len(pairs) >= 40, f"匹配字数过少: {len(pairs)}"

        y_errors = sorted(abs(p["dy"]) for p in pairs)
        p90 = y_errors[int(len(y_errors) * 0.9)]
        assert p90 < 180, \
            f"Y 误差 p90={p90:.1f}px 超过阈值 180px"

    def test_within_one_char_height_ratio(self, pipeline_result, gt_data):
        """至少 30% 的字符坐标误差 < 101px (1 个字符高)。

        基线 32% 偏低，主要受两个问题影响:
          - q_idx=201 (题8) y 偏差 +425px (layout_analyzer 检测错误)
          - q_idx=5 (题6) y 偏差 +159px (y-clamp 使用了错误的手写框)
        修复这两个问题后，占比应提升到 50%+。
        """
        pairs = self._match_pipeline_to_gt(pipeline_result.result_json, gt_data)
        assert len(pairs) >= 40, f"匹配字数过少: {len(pairs)}"

        within = sum(1 for p in pairs
                     if abs(p["dx"]) < 101 and abs(p["dy"]) < 101)
        ratio = within / len(pairs)
        assert ratio >= 0.30, \
            f"误差 < 101px 的字符仅占 {ratio:.1%}，应 ≥ 30%"

    def test_error_report(self, pipeline_result, gt_data):
        """输出详细的坐标误差报告（信息性测试，不 assert）。"""
        pairs = self._match_pipeline_to_gt(pipeline_result.result_json, gt_data)
        if not pairs:
            pytest.skip("无匹配数据")

        x_errors = [abs(p["dx"]) for p in pairs]
        y_errors = [abs(p["dy"]) for p in pairs]
        x_errors_sorted = sorted(x_errors)
        y_errors_sorted = sorted(y_errors)

        print(f"\n=== 坐标误差报告 (n={len(pairs)}) ===")
        print(f"  X: avg={sum(x_errors) / len(x_errors):.1f}px, "
              f"p50={x_errors_sorted[len(x_errors_sorted) // 2]:.1f}px, "
              f"p90={x_errors_sorted[int(len(x_errors_sorted) * 0.9)]:.1f}px, "
              f"max={max(x_errors):.1f}px")
        print(f"  Y: avg={sum(y_errors) / len(y_errors):.1f}px, "
              f"p50={y_errors_sorted[len(y_errors_sorted) // 2]:.1f}px, "
              f"p90={y_errors_sorted[int(len(y_errors_sorted) * 0.9)]:.1f}px, "
              f"max={max(y_errors):.1f}px")

        # 按 q_idx 分组打印
        by_q = {}
        for p in pairs:
            by_q.setdefault(p["q_idx"], []).append(p)
        print(f"\n  按 q_idx 分组 (avg |Δx|, avg |Δy|):")
        for qi in sorted(by_q):
            ps = by_q[qi]
            avg_dx = sum(abs(p["dx"]) for p in ps) / len(ps)
            avg_dy = sum(abs(p["dy"]) for p in ps) / len(ps)
            print(f"    q_idx={qi}: n={len(ps)}, "
                  f"avg|Δx|={avg_dx:.0f}px, avg|Δy|={avg_dy:.0f}px")

    def test_no_q_idx_mismatch(self, pipeline_result, gt_data):
        """gt 中所有 q_idx 应在 pipeline 中有对应（除已知的 ② 题遗漏）。

        已知遗漏: q_idx=102 (② 题"绚丽") 在 pipeline 中未识别
        """
        gt_q_indices = set(gt["q_idx"] for gt in gt_data)
        pl_q_indices = set(r.get("question_idx", -1) for r in pipeline_result.result_json)
        missing = gt_q_indices - pl_q_indices

        # 已知 ② 题 (q_idx=102) 在 pipeline 中未识别，允许
        known_missing = {102}
        unexpected_missing = missing - known_missing
        assert not unexpected_missing, \
            f"pipeline 缺失了意外的 q_idx: {unexpected_missing}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
