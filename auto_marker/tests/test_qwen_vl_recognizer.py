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
    _estimate_question_y_range,
    _norm_to_pixel_bbox,
    _q_idx_format_range,
    _question_marker_to_q_idx,
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
    """模型 y 坐标替换为手写框 y 范围。"""

    IMG_W = 1653
    IMG_H = 2312

    def test_replaces_y_with_hw_box(self):
        """y 被替换为手写框 y 范围，x 保留。"""
        # 模型 y=0.45-0.52（错误，偏移到题5位置）
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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
