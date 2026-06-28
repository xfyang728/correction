"""坐标映射单元测试 — 防止 y-clamp 和 q_idx 回退匹配回归。

覆盖核心函数:
  - _norm_to_pixel_bbox: 归一化坐标 → 像素坐标
  - _estimate_question_y_range: 同列最近手写框 y 范围查找
  - _clamp_chars_y_to_region: 模型 y 替换为手写框 y
  - _q_idx_format_range: q_idx 格式判断
  - _question_marker_to_q_idx: 题号文本 → q_idx
"""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core.qwen_vl_recognizer import (
    _clamp_chars_y_to_region,
    _estimate_question_x_range,
    _estimate_question_y_range,
    _find_answer_start_in_marker,
    _norm_to_pixel_bbox,
    _q_idx_format_range,
    _question_marker_to_q_idx,
    _rescale_chars_x_to_region,
    _split_merged_json_chars,
    recognize_page_level,
)


# ============================================================
# _norm_to_pixel_bbox
# ============================================================

class TestNormToPixelBbox:
    """归一化坐标 [0-1] → 像素坐标转换。"""

    def test_basic_conversion(self):
        """正常坐标转换。"""
        result = _norm_to_pixel_bbox((0.5, 0.5, 0.6, 0.6), 1000, 2000)
        assert result == (500, 1000, 600, 1200)

    def test_origin(self):
        """左上角 (0,0)。"""
        result = _norm_to_pixel_bbox((0.0, 0.0, 0.1, 0.1), 1000, 2000)
        assert result == (0, 0, 100, 200)

    def test_bottom_right(self):
        """右下角 (1,1)。"""
        result = _norm_to_pixel_bbox((0.9, 0.9, 1.0, 1.0), 1000, 2000)
        assert result == (900, 1800, 1000, 2000)

    def test_zero_width_clamped(self):
        """零宽 bbox 自动补 1px。"""
        result = _norm_to_pixel_bbox((0.5, 0.5, 0.5, 0.6), 1000, 2000)
        assert result[2] == result[0] + 1  # x1 = x0 + 1

    def test_zero_height_clamped(self):
        """零高 bbox 自动补 1px。"""
        result = _norm_to_pixel_bbox((0.5, 0.5, 0.6, 0.5), 1000, 2000)
        assert result[3] == result[1] + 1  # y1 = y0 + 1


# ============================================================
# _q_idx_format_range
# ============================================================

class TestQIdxFormatRange:
    """q_idx 格式判断（paren/dot/circle）。"""

    def test_paren_range(self):
        """0-99 → paren。"""
        assert _q_idx_format_range(0) == "paren"
        assert _q_idx_format_range(50) == "paren"
        assert _q_idx_format_range(99) == "paren"

    def test_dot_range(self):
        """100-199 → dot。"""
        assert _q_idx_format_range(100) == "dot"
        assert _q_idx_format_range(150) == "dot"
        assert _q_idx_format_range(199) == "dot"

    def test_circle_range(self):
        """200-209 → circle。"""
        assert _q_idx_format_range(200) == "circle"
        assert _q_idx_format_range(205) == "circle"
        assert _q_idx_format_range(209) == "circle"


# ============================================================
# _question_marker_to_q_idx
# ============================================================

class TestQuestionMarkerToQIdx:
    """题号文本 → q_idx 转换。"""

    def test_paren_marker(self):
        """括号题号 (1) → q_idx=0。"""
        assert _question_marker_to_q_idx("（1）") == 0
        assert _question_marker_to_q_idx("(2)") == 1

    def test_dot_marker(self):
        """点号题号 1. → q_idx=100。"""
        assert _question_marker_to_q_idx("1.") == 100
        assert _question_marker_to_q_idx("2.") == 101

    def test_circle_marker(self):
        """带圈数字 ① → q_idx=200。"""
        assert _question_marker_to_q_idx("①") == 200
        assert _question_marker_to_q_idx("②") == 201

    def test_no_match(self):
        """无题号格式 → None。"""
        assert _question_marker_to_q_idx("下联") is None
        assert _question_marker_to_q_idx("") is None


# ============================================================
# _estimate_question_y_range
# ============================================================

class TestEstimateQuestionYRange:
    """同列最近手写框 y 范围查找。"""

    # 模拟图像尺寸
    IMG_W = 1653
    IMG_H = 2312

    def test_finds_same_column_box(self):
        """能找到同列（左半页）最近的手写框。"""
        region = {
            "y_start": 746,
            "y_end": 874,  # region 中心 y = 810
            "marker_bbox": (17, 675, 655, 817),  # marker 在左列
        }
        hw_boxes = [
            {"bbox_pixel": (919, 666, 1368, 804)},   # 右列，应被排除
            {"bbox_pixel": (17, 738, 655, 817)},      # 左列，y_center=777，最近
            {"bbox_pixel": (18, 866, 707, 949)},      # 左列，y_center=907，较远
        ]
        y0, y1 = _estimate_question_y_range(region, hw_boxes, self.IMG_H, self.IMG_W)
        assert y0 == 738
        assert y1 == 817

    def test_excludes_opposite_column(self):
        """排除右列手写框，即使 y 更近。"""
        region = {
            "y_start": 746,
            "y_end": 874,
            "marker_bbox": (17, 675, 655, 817),  # 左列
        }
        hw_boxes = [
            {"bbox_pixel": (919, 750, 1368, 820)},   # 右列，y_center=785，更近但不同列
            {"bbox_pixel": (17, 738, 655, 817)},      # 左列，y_center=777
        ]
        y0, y1 = _estimate_question_y_range(region, hw_boxes, self.IMG_H, self.IMG_W)
        # 应选左列的框，不是右列的
        assert y0 == 738
        assert y1 == 817

    def test_right_column_marker(self):
        """右列 marker 匹配右列手写框（手写框比 region 更紧凑时才使用）。"""
        region = {
            "y_start": 746,
            "y_end": 1004,  # height=258（较大）
            "marker_bbox": (844, 722, 959, 804),  # 右列
        }
        hw_boxes = [
            {"bbox_pixel": (17, 738, 655, 817)},      # 左列，排除
            {"bbox_pixel": (919, 666, 1368, 804)},   # 右列，height=138 < 258
        ]
        y0, y1 = _estimate_question_y_range(region, hw_boxes, self.IMG_H, self.IMG_W)
        assert y0 == 666
        assert y1 == 804

    def test_no_handwriting_boxes(self):
        """无手写框时返回 region 原始范围。"""
        region = {"y_start": 746, "y_end": 874, "marker_bbox": (17, 675, 655, 817)}
        y0, y1 = _estimate_question_y_range(region, None, self.IMG_H, self.IMG_W)
        assert y0 == 746
        assert y1 == 874

    def test_no_img_w(self):
        """img_w=0 时返回 region 原始范围（无法做列判断）。"""
        region = {"y_start": 746, "y_end": 874, "marker_bbox": (17, 675, 655, 817)}
        hw_boxes = [{"bbox_pixel": (17, 738, 655, 817)}]
        y0, y1 = _estimate_question_y_range(region, hw_boxes, self.IMG_H, img_w=0)
        assert y0 == 746
        assert y1 == 874

    def test_hw_box_larger_than_region(self):
        """手写框比 region 更大时不使用（只取更紧凑的）。"""
        region = {
            "y_start": 746,
            "y_end": 874,  # height=128
            "marker_bbox": (17, 675, 655, 817),
        }
        hw_boxes = [
            {"bbox_pixel": (17, 600, 655, 1000)},  # height=400，比 region 大
        ]
        y0, y1 = _estimate_question_y_range(region, hw_boxes, self.IMG_H, self.IMG_W)
        # 手写框更大，应返回 region 原始范围
        assert y0 == 746
        assert y1 == 874

    def test_no_marker_bbox(self):
        """无 marker_bbox 时用 img_w/2 做列判断（边界情况）。"""
        region = {
            "y_start": 746,
            "y_end": 874,
            "marker_bbox": None,
        }
        # marker_cx 默认 img_w/2 = 826.5，hw_box cx=336 < 826.5
        # 但 826.5 < 826.5 = False，336 < 826.5 = True，False != True → 排除
        # 所以无匹配，返回 region 原始范围
        hw_boxes = [
            {"bbox_pixel": (17, 738, 655, 817)},  # 左列
        ]
        y0, y1 = _estimate_question_y_range(region, hw_boxes, self.IMG_H, self.IMG_W)
        assert y0 == 746  # 回退到 region 原始范围
        assert y1 == 874


# ============================================================
# _clamp_chars_y_to_region
# ============================================================

class TestClampCharsYToRegion:
    """模型 y 坐标混合修正：完全偏移用 replace，部分重叠用 clamp。"""

    IMG_W = 1653
    IMG_H = 2312

    def test_replaces_y_with_hw_box(self):
        """模型 y 完全在 region 外（系统性偏移）→ replace 为手写框 y 范围。"""
        # 模型 y=0.45-0.52（错误，偏移到题5位置），手写框 y=738-817（0.319-0.353）
        # 0.45 > 0.353 → 完全在 region 外 → replace
        char_items = [
            {"char": "随", "bbox_norm": (0.12, 0.45, 0.16, 0.52)},
            {"char": "君", "bbox_norm": (0.16, 0.45, 0.20, 0.52)},
        ]
        region = {
            "y_start": 746,
            "y_end": 874,
            "marker_bbox": (17, 675, 655, 817),
        }
        hw_boxes = [
            {"bbox_pixel": (17, 738, 655, 817)},  # 正确的题1 y 范围
        ]
        result = _clamp_chars_y_to_region(char_items, region, self.IMG_H, hw_boxes, self.IMG_W)

        expected_y0 = 738 / self.IMG_H  # ≈ 0.319
        expected_y1 = 817 / self.IMG_H  # ≈ 0.353
        for ci in result:
            assert abs(ci["bbox_norm"][1] - expected_y0) < 0.001
            assert abs(ci["bbox_norm"][3] - expected_y1) < 0.001
            # x 坐标应保留模型原始值
            assert ci["bbox_norm"][0] == char_items[0]["bbox_norm"][0] or \
                   ci["bbox_norm"][0] == char_items[1]["bbox_norm"][0]

    def test_preserves_x_coordinates(self):
        """x 坐标完全保留，不被修改。"""
        char_items = [
            {"char": "随", "bbox_norm": (0.12, 0.45, 0.16, 0.52)},
        ]
        region = {"y_start": 746, "y_end": 874, "marker_bbox": (17, 675, 655, 817)}
        hw_boxes = [{"bbox_pixel": (17, 738, 655, 817)}]
        result = _clamp_chars_y_to_region(char_items, region, self.IMG_H, hw_boxes, self.IMG_W)

        assert result[0]["bbox_norm"][0] == 0.12  # x0 不变
        assert result[0]["bbox_norm"][2] == 0.16  # x1 不变

    def test_different_questions_get_different_y(self):
        """不同题目（不同 region + 手写框）获得不同 y 范围。"""
        # 题1 和 题5 模型返回相同的 y=0.45-0.52
        char_items_q1 = [
            {"char": "随", "bbox_norm": (0.12, 0.45, 0.16, 0.52)},
        ]
        char_items_q5 = [
            {"char": "人", "bbox_norm": (0.12, 0.45, 0.16, 0.52)},
        ]
        region_q1 = {"y_start": 746, "y_end": 874, "marker_bbox": (17, 675, 655, 817)}
        region_q5 = {"y_start": 1004, "y_end": 1134, "marker_bbox": (15, 995, 679, 1088)}
        hw_boxes = [
            {"bbox_pixel": (17, 738, 655, 817)},   # 题1 手写框
            {"bbox_pixel": (15, 995, 679, 1088)},  # 题5 手写框
        ]
        result_q1 = _clamp_chars_y_to_region(char_items_q1, region_q1, self.IMG_H, hw_boxes, self.IMG_W)
        result_q5 = _clamp_chars_y_to_region(char_items_q5, region_q5, self.IMG_H, hw_boxes, self.IMG_W)

        y0_q1 = result_q1[0]["bbox_norm"][1]
        y0_q5 = result_q5[0]["bbox_norm"][1]
        # 题1 和 题5 的 y 应该不同
        assert y0_q1 != y0_q5, f"题1 y={y0_q1} 不应等于 题5 y={y0_q5}"

    def test_no_handwriting_boxes_returns_original_y(self):
        """无手写框时用 region y 范围。"""
        char_items = [
            {"char": "随", "bbox_norm": (0.12, 0.45, 0.16, 0.52)},
        ]
        region = {"y_start": 746, "y_end": 874, "marker_bbox": (17, 675, 655, 817)}
        result = _clamp_chars_y_to_region(char_items, region, self.IMG_H, None, self.IMG_W)

        expected_y0 = 746 / self.IMG_H
        expected_y1 = 874 / self.IMG_H
        assert abs(result[0]["bbox_norm"][1] - expected_y0) < 0.001
        assert abs(result[0]["bbox_norm"][3] - expected_y1) < 0.001

    def test_partial_overlap_clamps_to_boundary(self):
        """模型 y 部分重叠 region → clamp 到边界（保留模型 y0 相对信息）。"""
        # region y=995-1088（0.430-0.471），模型 y=0.45-0.52
        # y0=0.45 在 [0.430, 0.471] 内，y1=0.52 > 0.471 → 部分重叠 → clamp
        char_items = [
            {"char": "人", "bbox_norm": (0.12, 0.45, 0.16, 0.52)},
        ]
        region = {"y_start": 1004, "y_end": 1134, "marker_bbox": (15, 995, 679, 1088)}
        hw_boxes = [{"bbox_pixel": (15, 995, 679, 1088)}]
        result = _clamp_chars_y_to_region(char_items, region, self.IMG_H, hw_boxes, self.IMG_W)

        norm_y0 = 995 / self.IMG_H  # 0.430
        norm_y1 = 1088 / self.IMG_H  # 0.471
        # clamp：y0 = max(0.45, 0.430) = 0.45（保留模型 y0），y1 = min(0.52, 0.471) = 0.471
        assert abs(result[0]["bbox_norm"][1] - 0.45) < 0.001, \
            f"部分重叠时应保留模型 y0=0.45, 实际={result[0]['bbox_norm'][1]}"
        assert abs(result[0]["bbox_norm"][3] - norm_y1) < 0.001, \
            f"部分重叠时 y1 应 clamp 到 {norm_y1}, 实际={result[0]['bbox_norm'][3]}"

    def test_fully_inside_preserves_model_y(self):
        """模型 y 完全在 region 内 → 保留模型 y（不修改）。"""
        # region y=738-817（0.319-0.353），模型 y=0.33-0.34（完全在内）
        char_items = [
            {"char": "随", "bbox_norm": (0.12, 0.33, 0.16, 0.34)},
        ]
        region = {"y_start": 746, "y_end": 874, "marker_bbox": (17, 675, 655, 817)}
        hw_boxes = [{"bbox_pixel": (17, 738, 655, 817)}]
        result = _clamp_chars_y_to_region(char_items, region, self.IMG_H, hw_boxes, self.IMG_W)

        # 完全在内 → clamp 不改变（max(0.33, 0.319)=0.33, min(0.34, 0.353)=0.34）
        assert result[0]["bbox_norm"][1] == 0.33, "完全在内时应保留模型 y0"
        assert result[0]["bbox_norm"][3] == 0.34, "完全在内时应保留模型 y1"

    def test_y_clamp_disabled_preserves_model_y(self, monkeypatch):
        """Y_CLAMP_ENABLED=False 时完全保留模型 y（A/B 测试开关）。"""
        import core.recognition_config as cfg
        monkeypatch.setattr(cfg, "Y_CLAMP_ENABLED", False)

        # 模型 y=0.45-0.52（明显偏移），但开关关闭 → 保留
        char_items = [
            {"char": "随", "bbox_norm": (0.12, 0.45, 0.16, 0.52)},
        ]
        region = {"y_start": 746, "y_end": 874, "marker_bbox": (17, 675, 655, 817)}
        hw_boxes = [{"bbox_pixel": (17, 738, 655, 817)}]
        result = _clamp_chars_y_to_region(char_items, region, self.IMG_H, hw_boxes, self.IMG_W)

        assert result[0]["bbox_norm"][1] == 0.45, "开关关闭时应保留模型 y0"
        assert result[0]["bbox_norm"][3] == 0.52, "开关关闭时应保留模型 y1"


# ============================================================
# _find_answer_start_in_marker
# ============================================================

class TestFindAnswerStartInMarker:
    """在 marker_text 中查找答案起始索引。"""

    def test_exact_match(self):
        """精确匹配: marker_text 含完整答案字符串。"""
        # marker_text = "(1) 随君直到夜郎西", answer = "随君直到夜郎西"
        idx = _find_answer_start_in_marker("(1) 随君直到夜郎西", "随君直到夜郎西")
        assert idx == 4  # 答案首字在索引 4

    def test_fuzzy_match_by_first_char(self):
        """模糊匹配: 用答案首字定位（marker_text 含 OCR 误差）。"""
        # marker_text 中答案被 OCR 改变，但首字相同
        idx = _find_answer_start_in_marker("(1) 随君直到夜郎", "随君直到夜郎西")
        assert idx == 4

    def test_answer_at_start(self):
        """marker_text 无题号（无题号格式如'下联'），答案在开头。"""
        idx = _find_answer_start_in_marker("平凡物见证", "平凡物见证诉说追梦情")
        # 首字 "平" 在索引 0
        assert idx == 0

    def test_circle_marker(self):
        """带圈题号 ② 答案起始位置。"""
        # marker_text = "② 染绚丽", answer = "染绚丽"
        idx = _find_answer_start_in_marker("② 染绚丽", "染绚丽")
        assert idx == 2  # "②" 占 1 个字符位 + 1 空格

    def test_not_found(self):
        """答案首字不在 marker_text 中 → 返回 None。"""
        idx = _find_answer_start_in_marker("(1) 甲乙丙", "XYZ")
        assert idx is None

    def test_empty_inputs(self):
        """空输入 → 返回 None。"""
        assert _find_answer_start_in_marker("", "abc") is None
        assert _find_answer_start_in_marker("abc", "") is None
        assert _find_answer_start_in_marker(None, "abc") is None


# ============================================================
# _estimate_question_x_range
# ============================================================

class TestEstimateQuestionXRange:
    """估算答案 x 范围 — 重点验证 answer_text 比例估算逻辑。"""

    IMG_W = 1653

    def test_proportional_estimate_with_answer_text(self):
        """Step 2: 用 marker_text + answer_text 比例裁掉题号区域。

        场景: marker_text="(1) 随君直到夜郎西", marker_bbox=(17, _, 655, _)
              答案在 marker_text 索引 4（11 字中第 5 字起）
              预期 x_start = 17 + 4/11 * (655-17) ≈ 249
        """
        region = {
            "y_start": 746,
            "y_end": 874,
            "marker_bbox": (17, 675, 655, 817),
            "marker_text": "(1) 随君直到夜郎西",  # 11 字符
        }
        x_start, x_end = _estimate_question_x_range(
            region, None, self.IMG_W, answer_text="随君直到夜郎西")
        # 答案从索引 4 开始，char_w = 638/11 ≈ 58
        # x_start = 17 + 4*58 = 249
        # x_end = 249 + 7*58 = 655 (但 min(655, marker[2]=655) = 655)
        assert x_start == 249, f"x_start={x_start}, 预期 249"
        assert x_end == 655, f"x_end={x_end}, 预期 655"

    def test_proportional_estimate_right_column(self):
        """右列题目比例估算。"""
        # marker_text = "(6) 半竿斜日旧关城", marker_bbox=(823, _, 1523, _)
        # 答案 "半竿斜日旧关城" 在索引 4（11 字中第 5 字起）
        region = {
            "y_start": 1000,
            "y_end": 1266,
            "marker_bbox": (823, 990, 1523, 1086),
            "marker_text": "(6) 半竿斜日旧关城",
        }
        x_start, x_end = _estimate_question_x_range(
            region, None, self.IMG_W, answer_text="半竿斜日旧关城")
        # char_w = 700/11 ≈ 63.6
        # x_start = 823 + 4*63.6 = 1077
        # x_end = 1077 + 7*63.6 = 1522 (min(1522, 1523) = 1522)
        assert x_start == 1077, f"x_start={x_start}, 预期 1077"
        assert x_end == 1522, f"x_end={x_end}, 预期 1522"

    def test_answer_at_marker_start_no_trim(self):
        """答案在 marker_text 开头时（answer_start=0），不裁题号，回退到 Step 3/4。

        场景: 无题号格式题目，marker_text == answer_text（答案在索引 0）
              answer_start=0 → Step 2 条件 (>0) 不满足，跳过
              marker 宽度 > 200 → Step 3 跳过
              无手写框 → Step 4 回退 (marker_bbox[2], img_w)
        """
        region = {
            "y_start": 1800,
            "y_end": 2050,
            "marker_bbox": (50, 1750, 1600, 2050),  # 宽 1550 > 200
            "marker_text": "平凡物见证",  # 与 answer_text 完全相同
        }
        x_start, x_end = _estimate_question_x_range(
            region, None, self.IMG_W, answer_text="平凡物见证")
        # Step 2 不触发（answer_start=0），Step 3 不触发（宽 > 200）
        # Step 4: 无手写框，返回 (marker_bbox[2], img_w)
        assert x_start == 1600
        assert x_end == self.IMG_W

    def test_narrow_marker_fallback(self):
        """Step 3: 窄 marker (< 200px) 时，x_start = marker_bbox[2]。"""
        # 无 answer_text，marker 宽度 100 < 200
        region = {
            "y_start": 746,
            "y_end": 874,
            "marker_bbox": (17, 675, 117, 817),  # 宽 100
            "marker_text": "(2)",
        }
        x_start, x_end = _estimate_question_x_range(
            region, None, self.IMG_W, answer_text=None)
        # Step 3: x_start = 117, x_end = img_w (无手写框)
        assert x_start == 117
        assert x_end == self.IMG_W

    def test_narrow_marker_with_handwriting_box(self):
        """Step 3: 窄 marker + 同列手写框 → x_end = 手写框 x_end。"""
        region = {
            "y_start": 746,
            "y_end": 874,
            "marker_bbox": (17, 675, 117, 817),  # 宽 100，左列
            "marker_text": "(2)",
        }
        hw_boxes = [
            {"bbox_pixel": (150, 738, 655, 817)},  # 左列，y_center 在 [746, 874)
        ]
        x_start, x_end = _estimate_question_x_range(
            region, hw_boxes, self.IMG_W, answer_text=None)
        # Step 1: hw_x_start=150, hw_x_end=655
        # Step 3: x_start = 117, x_end = hw_x_end = 655
        assert x_start == 117
        assert x_end == 655

    def test_fallback_to_handwriting_box_range(self):
        """Step 4: 无 answer_text + 宽 marker → 回退到手写框 x 范围。"""
        region = {
            "y_start": 746,
            "y_end": 874,
            "marker_bbox": (17, 675, 655, 817),  # 宽 638 > 200，左列
            "marker_text": "(1) 随君直到夜郎西",
        }
        hw_boxes = [
            {"bbox_pixel": (289, 738, 733, 817)},  # 左列，y_center 在范围内
        ]
        x_start, x_end = _estimate_question_x_range(
            region, hw_boxes, self.IMG_W, answer_text=None)
        # Step 4: 返回手写框 x 范围
        assert x_start == 289
        assert x_end == 733

    def test_no_marker_bbox_no_handwriting(self):
        """边界: 无 marker_bbox 且无手写框 → 返回 (0, img_w)。"""
        region = {
            "y_start": 746,
            "y_end": 874,
            "marker_bbox": None,
            "marker_text": "",
        }
        x_start, x_end = _estimate_question_x_range(
            region, None, self.IMG_W, answer_text=None)
        assert x_start == 0
        assert x_end == self.IMG_W

    def test_answer_text_not_in_marker_falls_back(self):
        """answer_text 不在 marker_text 中 → 回退到 Step 3/4。"""
        region = {
            "y_start": 746,
            "y_end": 874,
            "marker_bbox": (17, 675, 655, 817),  # 宽 638 > 200
            "marker_text": "(1) 甲乙丙丁",
        }
        hw_boxes = [
            {"bbox_pixel": (289, 738, 733, 817)},
        ]
        x_start, x_end = _estimate_question_x_range(
            region, hw_boxes, self.IMG_W, answer_text="XYZ不存在")
        # _find_answer_start_in_marker 返回 None → Step 2 跳过
        # Step 3: 宽 638 > 200，跳过
        # Step 4: 回退到手写框
        assert x_start == 289
        assert x_end == 733

    def test_excludes_opposite_column_handwriting(self):
        """Step 1: 排除对列手写框（左列 marker 不匹配右列手写框）。"""
        region = {
            "y_start": 746,
            "y_end": 874,
            "marker_bbox": (17, 675, 655, 817),  # 左列
            "marker_text": "(1) 答案",
        }
        hw_boxes = [
            {"bbox_pixel": (919, 738, 1368, 817)},  # 右列，y_center 在范围内但不同列
            {"bbox_pixel": (289, 738, 733, 817)},  # 左列
        ]
        x_start, x_end = _estimate_question_x_range(
            region, hw_boxes, self.IMG_W, answer_text=None)
        # Step 4: 只用左列手写框 → (289, 733)
        assert x_start == 289
        assert x_end == 733


# ============================================================
# _rescale_chars_x_to_region
# ============================================================

class TestRescaleCharsXToRegion:
    """model bbox 路径 X 重缩放 — 双重触发判断 + answer_text 启用 Step 2。"""

    IMG_W = 1653

    def _make_region(self, marker_bbox=(17, 675, 655, 817), marker_text="(1) 随君直到夜郎西"):
        """构造左列题目区域。"""
        return {
            "y_start": 746,
            "y_end": 874,
            "marker_bbox": marker_bbox,
            "marker_text": marker_text,
        }

    def test_span_exceed_triggers_rescale(self):
        """跨度偏大（>1.3×目标）触发线性映射。"""
        # 模型跨度 744px（0.05~0.5 → 83~827px），目标跨度 449px（289~738）
        # 744 > 449*1.3=584 → 触发
        region = self._make_region()
        hw_boxes = [{"bbox_pixel": (289, 738, 738, 817)}]  # 目标范围 449px
        char_items = [
            {"char": "随", "bbox_norm": (0.05, 0.32, 0.10, 0.35)},  # 83px
            {"char": "西", "bbox_norm": (0.45, 0.32, 0.50, 0.35)},  # 827px
        ]
        result = _rescale_chars_x_to_region(char_items, region, self.IMG_W, hw_boxes)
        # 映射后首字 x0 应接近目标 x_start=289
        new_x0 = result[0]["bbox_norm"][0] * self.IMG_W
        assert abs(new_x0 - 289) < 2, f"首字 x0={new_x0}, 预期 ≈289"
        # 末字 x1 应接近目标 x_end=738
        new_x1 = result[1]["bbox_norm"][2] * self.IMG_W
        assert abs(new_x1 - 738) < 2, f"末字 x1={new_x1}, 预期 ≈738"

    def test_offset_exceed_triggers_rescale(self):
        """跨度正常但首字偏移 >80px 触发线性映射（退化为平移）。"""
        # 目标范围 [289, 738]，跨度 449px
        # 模型跨度 449px（正常），但首字在 100px（偏移 189px > 80）→ 触发
        region = self._make_region()
        hw_boxes = [{"bbox_pixel": (289, 738, 738, 817)}]
        char_items = [
            {"char": "随", "bbox_norm": (100/1653, 0.32, 150/1653, 0.35)},  # 100~150px
            {"char": "西", "bbox_norm": (499/1653, 0.32, 549/1653, 0.35)},  # 499~549px
        ]
        result = _rescale_chars_x_to_region(char_items, region, self.IMG_W, hw_boxes)
        # 模型跨度 449 == 目标跨度 449，线性映射退化为平移
        # 首字 x0 应从 100 → 289（平移 +189）
        new_x0 = result[0]["bbox_norm"][0] * self.IMG_W
        assert abs(new_x0 - 289) < 2, f"首字 x0={new_x0}, 预期 ≈289（平移修正）"

    def test_both_ok_preserves_model_x(self):
        """跨度正常且偏移 <80px → 保留模型 x 原值。"""
        # 目标范围 [289, 738]，模型首字在 300px（偏移 11px < 80），跨度 400px < 449*1.3
        region = self._make_region()
        hw_boxes = [{"bbox_pixel": (289, 738, 738, 817)}]
        char_items = [
            {"char": "随", "bbox_norm": (300/1653, 0.32, 350/1653, 0.35)},
            {"char": "西", "bbox_norm": (650/1653, 0.32, 700/1653, 0.35)},
        ]
        result = _rescale_chars_x_to_region(char_items, region, self.IMG_W, hw_boxes)
        # 应保留模型原值
        assert result[0]["bbox_norm"][0] == char_items[0]["bbox_norm"][0]
        assert result[1]["bbox_norm"][2] == char_items[1]["bbox_norm"][2]

    def test_answer_text_enables_step2_target(self):
        """传 answer_text 启用 Step 2，目标范围用 marker_text 比例裁剪。"""
        # marker_text="(1) 随君直到夜郎西"，marker_bbox=(17,_,655,_)
        # Step 2: answer_start=4, char_w=638/11≈58, x_start=17+4*58=249, x_end=655
        # 不传 answer_text 时走 Step 4（手写框回退），目标范围=(289,738)
        region = self._make_region()
        hw_boxes = [{"bbox_pixel": (289, 738, 738, 817)}]
        # 模型首字在 100px（偏移 149px > 80），触发重缩放
        char_items = [
            {"char": "随", "bbox_norm": (100/1653, 0.32, 150/1653, 0.35)},
        ]
        # 传 answer_text → Step 2 目标范围 (249, 655)
        result_with = _rescale_chars_x_to_region(
            char_items, region, self.IMG_W, hw_boxes,
            answer_text="随君直到夜郎西")
        new_x0_with = result_with[0]["bbox_norm"][0] * self.IMG_W
        # 应映射到 Step 2 的 x_start=249（而非 Step 4 的 289）
        assert abs(new_x0_with - 249) < 2, f"Step2 启用时 x0={new_x0_with}, 预期 ≈249"

        # 不传 answer_text → Step 4 目标范围 (289, 738)
        result_without = _rescale_chars_x_to_region(
            char_items, region, self.IMG_W, hw_boxes)
        new_x0_without = result_without[0]["bbox_norm"][0] * self.IMG_W
        assert abs(new_x0_without - 289) < 2, f"Step4 回退时 x0={new_x0_without}, 预期 ≈289"

    def test_no_answer_text_falls_back(self):
        """不传 answer_text 时走 Step 3/4，行为同现状（向后兼容）。"""
        region = self._make_region()
        hw_boxes = [{"bbox_pixel": (289, 738, 738, 817)}]
        # 模型跨度偏大触发
        char_items = [
            {"char": "随", "bbox_norm": (0.05, 0.32, 0.10, 0.35)},
            {"char": "西", "bbox_norm": (0.45, 0.32, 0.50, 0.35)},
        ]
        result = _rescale_chars_x_to_region(char_items, region, self.IMG_W, hw_boxes)
        # 应映射到手写框范围 (289, 738)
        new_x0 = result[0]["bbox_norm"][0] * self.IMG_W
        assert abs(new_x0 - 289) < 2

    def test_single_char_skipped(self):
        """单字（model_span_norm=0）时不报错，保留原值。"""
        # 单字 bbox x0==x1 → model_span_norm=0，应跳过
        region = self._make_region()
        hw_boxes = [{"bbox_pixel": (289, 738, 738, 817)}]
        char_items = [
            {"char": "随", "bbox_norm": (0.05, 0.32, 0.05, 0.35)},  # x0==x1
        ]
        result = _rescale_chars_x_to_region(char_items, region, self.IMG_W, hw_boxes)
        # 虽然偏移超阈，但 model_span_norm=0 → 跳过，保留原值
        assert result[0]["bbox_norm"][0] == 0.05

    def test_multi_row_rescale(self):
        """跨行题，每行独立判断和映射。"""
        # 行1: y=0.32, 模型 x 跨度偏大 → 映射
        # 行2: y=0.50, 模型 x 合理 → 保留
        region = self._make_region()
        hw_boxes = [{"bbox_pixel": (289, 738, 738, 817)}]
        char_items = [
            # 行1 (y=0.32): 跨度 744px > 449*1.3 → 映射
            {"char": "随", "bbox_norm": (0.05, 0.32, 0.10, 0.35)},
            {"char": "西", "bbox_norm": (0.45, 0.32, 0.50, 0.35)},
            # 行2 (y=0.50): 跨度 100px < 449*1.3, 偏移 11px < 80 → 保留
            {"char": "君", "bbox_norm": (300/1653, 0.50, 350/1653, 0.53)},
            {"char": "到", "bbox_norm": (380/1653, 0.50, 430/1653, 0.53)},
        ]
        result = _rescale_chars_x_to_region(char_items, region, self.IMG_W, hw_boxes)
        # 行1 首字应被映射到 ≈289
        assert abs(result[0]["bbox_norm"][0] * self.IMG_W - 289) < 2
        # 行2 首字应保留原值 300
        assert abs(result[2]["bbox_norm"][0] * self.IMG_W - 300) < 2

    def test_empty_chars_returns_empty(self):
        """空输入返回空列表。"""
        region = self._make_region()
        result = _rescale_chars_x_to_region([], region, self.IMG_W, None)
        assert result == []


# ============================================================
# 集成场景：模拟多题坐标映射
# ============================================================

class TestMultiQuestionScenario:
    """模拟真实场景：8 道诗题共享 3 个 y 值，y-clamp 后应各不相同。"""

    IMG_W = 1653
    IMG_H = 2312

    def test_eight_questions_distinct_y(self):
        """8 道题模型返回共享 y 值，clamp 后每题 y 唯一。"""
        # 模型返回的 y 只有 3 个值（系统性偏移）
        model_y_ranges = [
            (0.45, 0.52),  # 题1
            (0.38, 0.45),  # 题2
            (0.37, 0.44),  # 题3
            (0.38, 0.45),  # 题4
            (0.45, 0.52),  # 题5
            (0.38, 0.45),  # 题6
            (0.45, 0.52),  # 题7
        ]

        # 布局分析检测的题目区域和手写框
        regions = [
            {"y_start": 746, "y_end": 874, "marker_bbox": (17, 675, 655, 817)},     # 题1 左
            {"y_start": 763, "y_end": 871, "marker_bbox": (844, 722, 959, 804)},    # 题2 右
            {"y_start": 874, "y_end": 1004, "marker_bbox": (18, 866, 707, 949)},    # 题3 左
            {"y_start": 871, "y_end": 1000, "marker_bbox": (820, 862, 1464, 954)},  # 题4 右
            {"y_start": 1004, "y_end": 1134, "marker_bbox": (15, 995, 679, 1088)},  # 题5 左
            {"y_start": 1000, "y_end": 1266, "marker_bbox": (823, 990, 1523, 1086)},# 题6 右
            {"y_start": 1134, "y_end": 1281, "marker_bbox": (14, 1125, 556, 1219)}, # 题7 左
        ]
        hw_boxes = [
            {"bbox_pixel": (17, 738, 655, 817)},    # 题1
            {"bbox_pixel": (919, 666, 1368, 804)},  # 题2
            {"bbox_pixel": (18, 866, 707, 949)},    # 题3
            {"bbox_pixel": (820, 862, 1464, 954)},  # 题4
            {"bbox_pixel": (15, 995, 679, 1088)},   # 题5
            {"bbox_pixel": (823, 990, 1523, 1086)}, # 题6
            {"bbox_pixel": (14, 1125, 556, 1219)},  # 题7
        ]

        y_values = set()
        for i, (y0_m, y1_m) in enumerate(model_y_ranges):
            char_items = [{"char": "x", "bbox_norm": (0.12, y0_m, 0.16, y1_m)}]
            result = _clamp_chars_y_to_region(
                char_items, regions[i], self.IMG_H, hw_boxes, self.IMG_W)
            y_key = (round(result[0]["bbox_norm"][1], 4), round(result[0]["bbox_norm"][3], 4))
            y_values.add(y_key)

        # 8 道题应有至少 5 个不同的 y 值（模型只有 3 个，clamp 后应更多）
        assert len(y_values) >= 5, f"y 值不够多样: {y_values}"

    def test_left_right_column_separation(self):
        """左右列题目不会互相干扰 y 范围。"""
        # 题1（左列）和 题2（右列）区域较大，手写框更紧凑
        region_q1 = {"y_start": 746, "y_end": 1004, "marker_bbox": (17, 675, 655, 817)}
        region_q2 = {"y_start": 746, "y_end": 1004, "marker_bbox": (844, 722, 959, 804)}
        hw_boxes = [
            {"bbox_pixel": (17, 738, 655, 817)},    # 左列，height=79
            {"bbox_pixel": (919, 666, 1368, 804)},  # 右列，height=138
        ]
        char_items = [{"char": "x", "bbox_norm": (0.12, 0.45, 0.16, 0.52)}]

        r1 = _clamp_chars_y_to_region(char_items, region_q1, self.IMG_H, hw_boxes, self.IMG_W)
        r2 = _clamp_chars_y_to_region(char_items, region_q2, self.IMG_H, hw_boxes, self.IMG_W)

        # 题1 应匹配左列手写框，题2 应匹配右列手写框
        y1 = r1[0]["bbox_norm"][1] * self.IMG_H
        y2 = r2[0]["bbox_norm"][1] * self.IMG_H
        assert abs(y1 - 738) < 5, f"题1 应匹配左列 y=738, got {y1}"
        assert abs(y2 - 666) < 5, f"题2 应匹配右列 y=666, got {y2}"


# ============================================================
# _split_merged_json_chars（改进3：合并题拆分 + rescale）
# ============================================================

class TestSplitMergedJsonChars:
    """合并题拆分 — 验证改进3：传 handwriting_boxes + 调用 rescale。

    改进3 的改动：
    1. _clamp_chars_to_column 传 handwriting_boxes（而非 None）
    2. 在 clamp_column 之后、clamp_y 之前，插入 _rescale_chars_x_to_region
    """

    IMG_W = 1653
    IMG_H = 2312

    def _make_merged_char_items(self, n_chars: int, x_offset_norm: float = 0.0):
        """构造合并题的 char_items（模型输出，归一化坐标）。"""
        items = []
        # 模拟模型把两题合并输出，x 从 0.2 开始，每字间隔 0.03
        for i in range(n_chars):
            x0 = 0.2 + i * 0.03 + x_offset_norm
            x1 = x0 + 0.025
            items.append({
                "char": f"字{i}",
                "bbox_norm": (x0, 0.45, x1, 0.52),  # 模型 y 偏移（典型）
            })
        return items

    def test_split_into_two_questions(self):
        """合并题按标准答案字数拆分为两题。"""
        # 题7（q_idx=6）5字 + 题8（q_idx=7）5字 = 10字合并
        char_items = self._make_merged_char_items(10)
        std_answers = {
            6: ["随", "君", "直", "到", "夜"],
            7: ["人", "闲", "桂", "花", "落"],
        }
        region_by_qidx = {
            6: {
                "y_start": 1134, "y_end": 1281,
                "marker_bbox": (14, 1125, 556, 1219),
                "marker_text": "(7) 随君直到夜",
                "q_idx": 6,
            },
            7: {
                "y_start": 1281, "y_end": 1428,
                "marker_bbox": (14, 1272, 556, 1366),
                "marker_text": "(8) 人闲桂花落",
                "q_idx": 7,
            },
        }
        hw_boxes = [
            {"bbox_pixel": (150, 1138, 655, 1217)},  # 题7 手写框
            {"bbox_pixel": (150, 1285, 655, 1364)},  # 题8 手写框
        ]

        results = _split_merged_json_chars(
            char_items, 6, std_answers, region_by_qidx,
            page_idx=0, img_w=self.IMG_W, img_h=self.IMG_H,
            used_regions=set(), handwriting_boxes=hw_boxes,
        )

        # 应拆分为 10 字，前 5 字归 q_idx=6，后 5 字归 q_idx=7
        assert len(results) == 10, f"应拆分 10 字，实际={len(results)}"
        q6_chars = [r for r in results if r["question_idx"] == 6]
        q7_chars = [r for r in results if r["question_idx"] == 7]
        assert len(q6_chars) == 5
        assert len(q7_chars) == 5

    def test_rescale_applied_to_split_chunks(self):
        """合并题拆分后，各 chunk 的 x 经过 rescale 修正（偏移超阈时）。"""
        # 模型 x 偏移很大（整体偏移 200px）
        char_items = self._make_merged_char_items(10, x_offset_norm=0.15)
        std_answers = {
            6: ["随", "君", "直", "到", "夜"],
            7: ["人", "闲", "桂", "花", "落"],
        }
        region_by_qidx = {
            6: {
                "y_start": 1134, "y_end": 1281,
                "marker_bbox": (14, 1125, 556, 1219),
                "marker_text": "(7) 随君直到夜",
                "q_idx": 6,
            },
        }
        hw_boxes = [
            {"bbox_pixel": (150, 1138, 655, 1217)},
        ]

        results = _split_merged_json_chars(
            char_items, 6, std_answers, region_by_qidx,
            page_idx=0, img_w=self.IMG_W, img_h=self.IMG_H,
            used_regions=set(), handwriting_boxes=hw_boxes,
        )

        # 题6 的 5 字应经过 rescale，x 偏移应被修正
        q6_results = [r for r in results if r["question_idx"] == 6]
        assert len(q6_results) == 5
        # 验证 x 在合理范围内（手写框 x=150~655）
        for r in q6_results:
            x_center = (r["bbox_pixel"][0] + r["bbox_pixel"][2]) / 2
            # rescale 后 x 应在 [150, 655] 附近（允许边界误差）
            assert 100 < x_center < 700, \
                f"rescale 后 x_center 应在 [150, 655] 附近，实际={x_center}"

    def test_handwriting_boxes_not_none(self):
        """传 handwriting_boxes（非 None）时不崩溃，且能用于 x 范围估算。"""
        char_items = self._make_merged_char_items(5)
        std_answers = {6: ["随", "君", "直", "到", "夜"]}
        region_by_qidx = {
            6: {
                "y_start": 1134, "y_end": 1281,
                "marker_bbox": (14, 1125, 556, 1219),
                "marker_text": "(7) 随君直到夜",
                "q_idx": 6,
            },
        }
        hw_boxes = [{"bbox_pixel": (150, 1138, 655, 1217)}]

        # 不应抛出异常
        results = _split_merged_json_chars(
            char_items, 6, std_answers, region_by_qidx,
            page_idx=0, img_w=self.IMG_W, img_h=self.IMG_H,
            used_regions=set(), handwriting_boxes=hw_boxes,
        )
        assert len(results) == 5

    def test_none_handwriting_boxes_no_crash(self):
        """handwriting_boxes=None 时不崩溃（向后兼容）。"""
        char_items = self._make_merged_char_items(5)
        std_answers = {6: ["随", "君", "直", "到", "夜"]}
        region_by_qidx = {
            6: {
                "y_start": 1134, "y_end": 1281,
                "marker_bbox": (14, 1125, 556, 1219),
                "marker_text": "(7) 随君直到夜",
                "q_idx": 6,
            },
        }

        # handwriting_boxes=None 不应崩溃
        results = _split_merged_json_chars(
            char_items, 6, std_answers, region_by_qidx,
            page_idx=0, img_w=self.IMG_W, img_h=self.IMG_H,
            used_regions=set(), handwriting_boxes=None,
        )
        assert len(results) == 5

    def test_no_region_for_second_question(self):
        """找不到下一题标准答案时，剩余全部归当前题（符合 docstring 约定）。"""
        char_items = self._make_merged_char_items(10)
        std_answers = {
            6: ["随", "君", "直", "到", "夜"],
            # 题7 无标准答案
        }
        region_by_qidx = {
            6: {
                "y_start": 1134, "y_end": 1281,
                "marker_bbox": (14, 1125, 556, 1219),
                "marker_text": "(7) 随君直到夜",
                "q_idx": 6,
            },
            # 题7 无 region
        }
        hw_boxes = [{"bbox_pixel": (150, 1138, 655, 1217)}]

        results = _split_merged_json_chars(
            char_items, 6, std_answers, region_by_qidx,
            page_idx=0, img_w=self.IMG_W, img_h=self.IMG_H,
            used_regions=set(), handwriting_boxes=hw_boxes,
        )
        # 找不到下一题（std_answers 无 key 7）→ 剩余全部归题6
        assert len(results) == 10
        q6_results = [r for r in results if r["question_idx"] == 6]
        assert len(q6_results) == 10, f"剩余应全部归题6，实际 q6={len(q6_results)}"

    def test_merged_split_sets_coord_source(self):
        """路径 B: 合并题拆分设置 coord_source='model_merged_split'。"""
        char_items = self._make_merged_char_items(10)
        std_answers = {
            6: ["随", "君", "直", "到", "夜"],
            7: ["人", "闲", "桂", "花", "落"],
        }
        region_by_qidx = {
            6: {
                "y_start": 1134, "y_end": 1281,
                "marker_bbox": (14, 1125, 556, 1219),
                "marker_text": "(7) 随君直到夜",
                "q_idx": 6,
            },
            7: {
                "y_start": 1281, "y_end": 1428,
                "marker_bbox": (844, 1272, 1368, 1366),
                "marker_text": "(8) 人闲桂花落",
                "q_idx": 7,
            },
        }
        hw_boxes = [{"bbox_pixel": (150, 1138, 655, 1217)}]

        results = _split_merged_json_chars(
            char_items, 6, std_answers, region_by_qidx,
            page_idx=0, img_w=self.IMG_W, img_h=self.IMG_H,
            used_regions=set(), handwriting_boxes=hw_boxes,
        )
        assert len(results) == 10
        # 所有结果应标记为 model_merged_split
        for r in results:
            assert r["coord_source"] == "model_merged_split", \
                f"合并题拆分应标记 model_merged_split，实际={r.get('coord_source')}"


# ============================================================
# coord_source 字段追踪（4 条坐标来源路径）
# ============================================================

class TestCoordSourceTracking:
    """验证 recognize_page_level 的 4 条路径正确设置 coord_source 字段。

    路径 A (model_json): JSON 模型坐标，经 rescale + clamp
    路径 B (model_merged_split): 合并题拆分（在 TestSplitMergedJsonChars 中测试）
    路径 C (fallback_invalid): 坐标无效，用标准答案等宽切分
    路径 D (fallback_text_parse): JSON 解析失败，旧文本解析 + 等宽切分
    """

    IMG_W = 1653
    IMG_H = 2312

    def _make_img(self):
        """构造测试用 numpy 图像。"""
        import numpy as np
        return np.full((self.IMG_H, self.IMG_W, 3), 255, dtype=np.uint8)

    def _make_question_regions(self):
        """构造题目区域列表（题1，q_idx=0）。"""
        return [{
            "page": 0,
            "q_idx": 0,
            "y_start": 746,
            "y_end": 874,
            "marker_bbox": (17, 675, 655, 817),
            "marker_text": "（1）随君直到夜郎西",
        }]

    def _make_hw_boxes(self):
        """构造手写区域列表。"""
        return [{"bbox_pixel": (17, 738, 655, 817)}]

    def _make_std_answers_text(self):
        """构造标准答案文本。"""
        return "（1）随君直到夜郎西"

    def test_model_json_path_sets_coord_source(self, monkeypatch):
        """路径 A: JSON 模型坐标设置 coord_source='model_json'。"""
        # mock 返回有效 JSON（含合法 bbox）
        valid_json = (
            '[{"q":"（1）","chars":['
            '{"c":"随","bbox":[0.12,0.45,0.16,0.52]},'
            '{"c":"君","bbox":[0.16,0.45,0.20,0.52]}'
            ']}]'
        )
        monkeypatch.setattr(
            "core.qwen_vl_recognizer._call_qwen_vl_page_level",
            lambda img, answers_text="": valid_json,
        )

        results = recognize_page_level(
            self._make_img(),
            self._make_question_regions(),
            page_idx=0,
            handwriting_boxes=self._make_hw_boxes(),
            answers_text=self._make_std_answers_text(),
        )

        assert len(results) > 0, "路径 A 应有产出"
        for r in results:
            assert r["coord_source"] == "model_json", \
                f"路径 A 应标记 model_json，实际={r.get('coord_source')}"

    def test_invalid_path_sets_coord_source(self, monkeypatch):
        """路径 C: 坐标无效（零宽 bbox）设置 coord_source='fallback_invalid'。"""
        # mock 返回 invalid JSON（bbox 零宽 → is_invalid=True）
        invalid_json = (
            '[{"q":"（1）","chars":['
            '{"c":"随","bbox":[0.5,0.5,0.5,0.6]},'
            '{"c":"君","bbox":[0.5,0.5,0.5,0.6]}'
            ']}]'
        )
        monkeypatch.setattr(
            "core.qwen_vl_recognizer._call_qwen_vl_page_level",
            lambda img, answers_text="": invalid_json,
        )

        results = recognize_page_level(
            self._make_img(),
            self._make_question_regions(),
            page_idx=0,
            handwriting_boxes=self._make_hw_boxes(),
            answers_text=self._make_std_answers_text(),
        )

        assert len(results) > 0, "路径 C 应有产出（用标准答案等宽切分）"
        for r in results:
            assert r["coord_source"] == "fallback_invalid", \
                f"路径 C 应标记 fallback_invalid，实际={r.get('coord_source')}"

    def test_fallback_text_parse_sets_coord_source(self, monkeypatch):
        """路径 D: JSON 解析失败设置 coord_source='fallback_text_parse'。"""
        # mock 返回非 JSON 文本（触发 _fallback_text_path）
        plain_text = "（1）随君直到夜郎西"
        monkeypatch.setattr(
            "core.qwen_vl_recognizer._call_qwen_vl_page_level",
            lambda img, answers_text="": plain_text,
        )

        results = recognize_page_level(
            self._make_img(),
            self._make_question_regions(),
            page_idx=0,
            handwriting_boxes=self._make_hw_boxes(),
            answers_text=self._make_std_answers_text(),
        )

        assert len(results) > 0, "路径 D 应有产出（旧文本解析 + 等宽切分）"
        for r in results:
            assert r["coord_source"] == "fallback_text_parse", \
                f"路径 D 应标记 fallback_text_parse，实际={r.get('coord_source')}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
