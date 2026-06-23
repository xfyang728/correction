"""layout_analyzer 模块单元测试 — 测试题号检测、列分割、区间构建。"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.layout_analyzer import (
    _QIDX_OFFSET_CIRCLE,
    _QIDX_OFFSET_DOT,
    _QIDX_OFFSET_PAREN,
    _build_column_intervals,
    _question_match,
    _split_into_columns,
)


class TestQuestionMatch:
    """测试 _question_match() 题号正则匹配。"""

    def test_paren_half_width(self):
        """半角括号 (1) → q_idx=0。"""
        assert _question_match("(1) 随君直到夜郎西") == 0 + _QIDX_OFFSET_PAREN
        assert _question_match("(2) 海内存知己") == 1 + _QIDX_OFFSET_PAREN

    def test_paren_full_width(self):
        """全角括号 （1） → q_idx=0。"""
        assert _question_match("（1）测试") == 0 + _QIDX_OFFSET_PAREN
        assert _question_match("（3）百般红紫") == 2 + _QIDX_OFFSET_PAREN

    def test_dot_number(self):
        """1. 格式 → q_idx=0 + DOT offset。"""
        assert _question_match("1. (8分）") == 0 + _QIDX_OFFSET_DOT
        assert _question_match("2.(2分)") == 1 + _QIDX_OFFSET_DOT

    def test_circled_number(self):
        """① ② 格式 → CIRCLE offset。"""
        assert _question_match("①(xuàn li) 染绚丽") == 0 + _QIDX_OFFSET_CIRCLE
        assert _question_match("②测试") == 1 + _QIDX_OFFSET_CIRCLE

    def test_no_match(self):
        """无题号的文本 → None。"""
        assert _question_match("悠然见南山") is None
        assert _question_match("以下为非选择题答题区") is None
        assert _question_match("") is None

    def test_q_idx_uniqueness(self):
        """不同格式题号的 q_idx 不冲突。"""
        paren = _question_match("(1)")
        dot = _question_match("1.")
        circle = _question_match("①")
        assert paren != dot
        assert paren != circle
        assert dot != circle


class TestSplitIntoColumns:
    """测试 _split_into_columns() 列检测。"""

    def test_single_column(self):
        """所有 marker x 坐标接近 → 单列。"""
        markers = [
            {"x_center": 100, "y_center": 200, "q_idx": 0, "marker_bbox": (90, 190, 110, 210), "marker_text": "(1)"},
            {"x_center": 105, "y_center": 300, "q_idx": 1, "marker_bbox": (95, 290, 115, 310), "marker_text": "(2)"},
        ]
        cols = _split_into_columns(markers)
        assert len(cols) == 1
        assert len(cols[0]) == 2

    def test_two_columns(self):
        """x 坐标间隙 > 150px → 双列。"""
        markers = [
            {"x_center": 100, "y_center": 200, "q_idx": 0, "marker_bbox": (90, 190, 110, 210), "marker_text": "(1)"},
            {"x_center": 800, "y_center": 200, "q_idx": 1, "marker_bbox": (790, 190, 810, 210), "marker_text": "(2)"},
        ]
        cols = _split_into_columns(markers)
        assert len(cols) == 2
        assert len(cols[0]) == 1
        assert len(cols[1]) == 1

    def test_single_marker(self):
        """单个 marker → 单列。"""
        markers = [{"x_center": 100, "y_center": 200, "q_idx": 0,
                     "marker_bbox": (90, 190, 110, 210), "marker_text": "(1)"}]
        cols = _split_into_columns(markers)
        assert len(cols) == 1

    def test_empty_markers(self):
        """空列表 → 单空列。"""
        cols = _split_into_columns([])
        assert len(cols) == 1
        assert cols[0] == []


class TestBuildColumnIntervals:
    """测试 _build_column_intervals() y 区间构建。"""

    def test_single_column_intervals(self):
        """单列 markers → y 区间连续。"""
        columns = [[
            {"q_idx": 0, "y_center": 100, "marker_bbox": (0, 90, 50, 110), "marker_text": "(1)"},
            {"q_idx": 1, "y_center": 300, "marker_bbox": (0, 290, 50, 310), "marker_text": "(2)"},
        ]]
        regions = _build_column_intervals(columns, img_h=1000)
        assert len(regions) == 2
        assert regions[0]["y_start"] == 100
        assert regions[0]["y_end"] == 300
        assert regions[1]["y_start"] == 300
        assert regions[1]["y_end"] == 1000  # 最后一个区间到图片底部

    def test_two_column_independent(self):
        """双列区间独立，不互相干扰。"""
        columns = [
            [{"q_idx": 0, "y_center": 100, "marker_bbox": (0, 90, 50, 110), "marker_text": "(1)"},
             {"q_idx": 2, "y_center": 300, "marker_bbox": (0, 290, 50, 310), "marker_text": "(3)"}],
            [{"q_idx": 1, "y_center": 100, "marker_bbox": (600, 90, 650, 110), "marker_text": "(2)"},
             {"q_idx": 3, "y_center": 300, "marker_bbox": (600, 290, 650, 310), "marker_text": "(4)"}],
        ]
        regions = _build_column_intervals(columns, img_h=1000)
        assert len(regions) == 4
        # 列 0 的区间
        col0 = [r for r in regions if r["column"] == 0]
        assert len(col0) == 2
        # 列 1 的区间
        col1 = [r for r in regions if r["column"] == 1]
        assert len(col1) == 2

    def test_single_marker_interval(self):
        """单个 marker → 区间从 marker y 到 img_h。"""
        columns = [[
            {"q_idx": 0, "y_center": 500, "marker_bbox": (0, 490, 50, 510), "marker_text": "(1)"},
        ]]
        regions = _build_column_intervals(columns, img_h=2000)
        assert len(regions) == 1
        assert regions[0]["y_start"] == 500
        assert regions[0]["y_end"] == 2000
