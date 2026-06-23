"""grader 模块单元测试 — 测试 DP 对齐算法和评分逻辑。"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.grader import _align, grade


class TestAlign:
    """测试 Needleman-Wunsch 对齐算法。"""

    def test_perfect_match(self):
        """完全匹配的序列应对齐为 1:1。"""
        ocr = ["春", "眠", "不", "觉", "晓"]
        ans = ["春", "眠", "不", "觉", "晓"]
        result = _align(ocr, ans)
        assert len(result) == 5
        for ocr_idx, ans_idx in result:
            assert ocr_idx is not None
            assert ans_idx is not None

    def test_extra_ocr_char(self):
        """OCR 多识别的字应被对齐为 (idx, None)。"""
        ocr = ["春", "眠", "X", "不", "觉", "晓"]
        ans = ["春", "眠", "不", "觉", "晓"]
        result = _align(ocr, ans)
        # 应有 6 个对齐项
        assert len(result) == 6
        # 其中一个 ocr_idx 对应 None ans_idx
        none_ans = [r for r in result if r[1] is None]
        assert len(none_ans) == 1

    def test_missing_ocr_char(self):
        """答案中多余的字（漏写）应对齐为 (None, idx)。"""
        ocr = ["春", "眠", "晓"]
        ans = ["春", "眠", "不", "觉", "晓"]
        result = _align(ocr, ans)
        assert len(result) == 5
        none_ocr = [r for r in result if r[0] is None]
        assert len(none_ocr) == 2

    def test_empty_ocr(self):
        """空 OCR 序列应对齐为全 (None, idx)。"""
        result = _align([], ["春", "眠"])
        assert len(result) == 2
        for ocr_idx, ans_idx in result:
            assert ocr_idx is None
            assert ans_idx is not None

    def test_empty_answers(self):
        """空答案序列应对齐为全 (idx, None)。"""
        result = _align(["春", "眠"], [])
        assert len(result) == 2
        for ocr_idx, ans_idx in result:
            assert ocr_idx is not None
            assert ans_idx is None

    def test_confidence_weighting(self):
        """高置信度的匹配应获得更高分。"""
        ocr = ["春", "X"]
        ans = ["春", "眠"]
        # 高置信度：X 更可能被当作错误字（wrong）
        result_high = _align(ocr, ans, [0.95, 0.95])
        # 低置信度：X 更可能被跳过（uncertain）
        result_low = _align(ocr, ans, [0.95, 0.30])
        # 两种情况都应产生 2 个对齐项
        assert len(result_high) == 2
        assert len(result_low) == 2


class TestGrade:
    """测试 grade() 评分逻辑。"""

    def test_all_correct(self):
        """全部正确且高置信度 → 全部 correct。"""
        ocr_results = [
            {"page": 0, "bbox_pixel": (0, 0, 10, 10), "char": "春",
             "confidence": 0.95, "img_pixel_w": 100, "img_pixel_h": 100},
            {"page": 0, "bbox_pixel": (10, 0, 20, 10), "char": "眠",
             "confidence": 0.95, "img_pixel_w": 100, "img_pixel_h": 100},
        ]
        answers = ["春", "眠"]
        graded = grade(ocr_results, answers)
        assert len(graded) == 2
        assert all(g["status"] == "correct" for g in graded)

    def test_wrong_char(self):
        """字符不匹配且高置信度 → wrong。"""
        ocr_results = [
            {"page": 0, "bbox_pixel": (0, 0, 10, 10), "char": "X",
             "confidence": 0.95, "img_pixel_w": 100, "img_pixel_h": 100},
        ]
        answers = ["春"]
        graded = grade(ocr_results, answers)
        assert len(graded) == 1
        assert graded[0]["status"] == "wrong"

    def test_uncertain_low_confidence(self):
        """字符匹配但低置信度 → uncertain。"""
        ocr_results = [
            {"page": 0, "bbox_pixel": (0, 0, 10, 10), "char": "春",
             "confidence": 0.65, "img_pixel_w": 100, "img_pixel_h": 100},
        ]
        answers = ["春"]
        graded = grade(ocr_results, answers)
        assert len(graded) == 1
        assert graded[0]["status"] == "uncertain"

    def test_skip_low_confidence(self):
        """极低置信度的字应被跳过。"""
        ocr_results = [
            {"page": 0, "bbox_pixel": (0, 0, 10, 10), "char": "春",
             "confidence": 0.95, "img_pixel_w": 100, "img_pixel_h": 100},
            {"page": 0, "bbox_pixel": (10, 0, 20, 10), "char": "眠",
             "confidence": 0.10, "img_pixel_w": 100, "img_pixel_h": 100},
        ]
        answers = ["春", "眠"]
        graded = grade(ocr_results, answers, skip_threshold=0.30)
        # 低置信度字被跳过，只剩 1 个
        assert len(graded) == 1

    def test_empty_ocr_results(self):
        """空 OCR 结果应返回空列表。"""
        graded = grade([], ["春"])
        assert graded == []
