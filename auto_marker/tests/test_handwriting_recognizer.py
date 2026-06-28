"""手写识别单元测试 — 验证多 OCR 记录匹配与拼接。

覆盖核心场景:
  - 单 OCR 记录匹配单手写框（基本功能）
  - 多 OCR 记录匹配同一手写框（改进2核心：移除 pop，允许重复匹配）
  - 多 OCR 记录按 x 坐标排序拼接
  - 未匹配手写框追踪
  - 无中文字符的 OCR 记录被跳过
"""

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core.handwriting_recognizer import recognize_handwriting


# ============================================================
# 测试辅助函数
# ============================================================

def _make_img(width: int = 1000, height: int = 1000) -> np.ndarray:
    """创建白色背景的测试图像。"""
    return np.full((height, width, 3), 255, dtype=np.uint8)


def _make_hw_box(x0: int, y0: int, x2: int, y2: int,
                 question_idx: int = 0) -> dict:
    """构造手写框。"""
    return {
        "bbox_pixel": (x0, y0, x2, y2),
        "question_idx": question_idx,
        "confidence": 0.9,
    }


def _make_ocr_record(text: str, x0: int, y0: int, x1: int, y1: int,
                     score: float = 0.9, page: int = 0) -> dict:
    """构造 OCR 记录。rec_bbox 中心点应落在某个手写框内。"""
    return {
        "rec_text": text,
        "rec_score": score,
        "rec_bbox": (x0, y0, x1, y1),
        "page": page,
    }


# ============================================================
# 基本匹配功能
# ============================================================

class TestBasicMatching:
    """单 OCR 记录匹配单手写框。"""

    def test_single_record_matches_box(self):
        """单个 OCR 记录中心点落在手写框内 → 匹配成功。"""
        img = _make_img()
        hw_boxes = [_make_hw_box(100, 100, 500, 200, question_idx=0)]
        ocr_records = [_make_ocr_record("随君", 150, 120, 250, 180)]

        results = recognize_handwriting(img, hw_boxes, ocr_records, page_idx=0)

        assert len(results) == 2  # "随君" → 2 字
        assert results[0]["char"] == "随"
        assert results[1]["char"] == "君"
        assert results[0]["question_idx"] == 0

    def test_no_matching_record(self):
        """OCR 记录中心点不在任何手写框内 → 无匹配。"""
        img = _make_img()
        hw_boxes = [_make_hw_box(100, 100, 500, 200)]
        # OCR 记录中心点在 (600, 300)，不在手写框 (100,100,500,200) 内
        ocr_records = [_make_ocr_record("随", 550, 250, 650, 350)]

        results = recognize_handwriting(img, hw_boxes, ocr_records, page_idx=0)
        assert len(results) == 0  # 无匹配

    def test_wrong_page_skipped(self):
        """OCR 记录页码不匹配 → 跳过。"""
        img = _make_img()
        hw_boxes = [_make_hw_box(100, 100, 500, 200)]
        ocr_records = [_make_ocr_record("随", 150, 120, 250, 180, page=1)]

        results = recognize_handwriting(img, hw_boxes, ocr_records, page_idx=0)
        assert len(results) == 0


# ============================================================
# 改进2核心：多 OCR 记录匹配同一手写框
# ============================================================

class TestMultipleRecordsMatchSameBox:
    """验证移除 pop 后，同一手写框可匹配多个 OCR 记录。"""

    def test_two_records_match_same_box(self):
        """两个 OCR 记录中心点都落在同一手写框内 → 都匹配。"""
        img = _make_img()
        hw_boxes = [_make_hw_box(100, 100, 800, 200, question_idx=0)]
        # 两个 OCR 记录，中心点都在手写框内
        ocr_records = [
            _make_ocr_record("随君", 150, 120, 350, 180),   # 中心 (250, 150)
            _make_ocr_record("直到", 400, 120, 600, 180),   # 中心 (500, 150)
        ]

        results = recognize_handwriting(img, hw_boxes, ocr_records, page_idx=0)

        # 改进2前：只有第一个 OCR 记录匹配（2字）
        # 改进2后：两个 OCR 记录都匹配并拼接（4字）
        assert len(results) == 4, f"应匹配 4 字（两个 OCR 记录拼接），实际={len(results)}"
        chars = [r["char"] for r in results]
        assert chars == ["随", "君", "直", "到"], f"字符顺序错误: {chars}"

    def test_three_records_match_same_box(self):
        """三个 OCR 记录都落在同一手写框内 → 都匹配并拼接。"""
        img = _make_img()
        hw_boxes = [_make_hw_box(100, 100, 900, 200, question_idx=0)]
        ocr_records = [
            _make_ocr_record("随君", 150, 120, 250, 180),
            _make_ocr_record("直到", 300, 120, 400, 180),
            _make_ocr_record("夜郎", 450, 120, 550, 180),
        ]

        results = recognize_handwriting(img, hw_boxes, ocr_records, page_idx=0)
        assert len(results) == 6, f"应匹配 6 字，实际={len(results)}"
        chars = [r["char"] for r in results]
        assert chars == ["随", "君", "直", "到", "夜", "郎"]

    def test_records_merged_by_x_order(self):
        """多个 OCR 记录按 rec_bbox x_center 排序拼接（非 ocr_records 输入顺序）。"""
        img = _make_img()
        hw_boxes = [_make_hw_box(100, 100, 900, 200, question_idx=0)]
        # 故意打乱顺序：第三个记录的 x 最小
        ocr_records = [
            _make_ocr_record("直到", 400, 120, 500, 180),   # x_center=450
            _make_ocr_record("夜郎", 600, 120, 700, 180),   # x_center=650
            _make_ocr_record("随君", 150, 120, 250, 180),   # x_center=200（最小）
        ]

        results = recognize_handwriting(img, hw_boxes, ocr_records, page_idx=0)
        chars = [r["char"] for r in results]
        # 应按 x 排序：随君 → 直到 → 夜郎
        assert chars == ["随", "君", "直", "到", "夜", "郎"], \
            f"应按 x 坐标排序拼接，实际={chars}"

    def test_merged_score_takes_minimum(self):
        """拼接后的置信度取各 OCR 记录中的最低值。"""
        img = _make_img()
        hw_boxes = [_make_hw_box(100, 100, 800, 200, question_idx=0)]
        ocr_records = [
            _make_ocr_record("随君", 150, 120, 350, 180, score=0.95),
            _make_ocr_record("直到", 400, 120, 600, 180, score=0.80),
        ]

        results = recognize_handwriting(img, hw_boxes, ocr_records, page_idx=0)
        # 所有字的置信度应为 min(0.95, 0.80) = 0.80
        for r in results:
            assert r["confidence"] == 0.80, \
                f"置信度应为 0.80（最低值），实际={r['confidence']}"


# ============================================================
# 多手写框场景
# ============================================================

class TestMultipleHandwritingBoxes:
    """多个手写框，每个框匹配各自的 OCR 记录。"""

    def test_two_boxes_each_match_one_record(self):
        """两个手写框，各匹配一个 OCR 记录。"""
        img = _make_img()
        hw_boxes = [
            _make_hw_box(100, 100, 500, 200, question_idx=0),
            _make_hw_box(100, 300, 500, 400, question_idx=1),
        ]
        ocr_records = [
            _make_ocr_record("随君", 150, 120, 250, 180),   # 匹配框1
            _make_ocr_record("人闲", 150, 320, 250, 380),   # 匹配框2
        ]

        results = recognize_handwriting(img, hw_boxes, ocr_records, page_idx=0)
        assert len(results) == 4
        q0_chars = [r["char"] for r in results if r["question_idx"] == 0]
        q1_chars = [r["char"] for r in results if r["question_idx"] == 1]
        assert q0_chars == ["随", "君"]
        assert q1_chars == ["人", "闲"]

    def test_one_box_multiple_records_other_box_single(self):
        """一个框匹配多记录，另一个框匹配单记录。"""
        img = _make_img()
        hw_boxes = [
            _make_hw_box(100, 100, 800, 200, question_idx=0),
            _make_hw_box(100, 300, 500, 400, question_idx=1),
        ]
        ocr_records = [
            _make_ocr_record("随君", 150, 120, 250, 180),   # 匹配框1
            _make_ocr_record("直到", 300, 120, 400, 180),   # 匹配框1（改进2前会被遗漏）
            _make_ocr_record("人闲", 150, 320, 250, 380),   # 匹配框2
        ]

        results = recognize_handwriting(img, hw_boxes, ocr_records, page_idx=0)
        assert len(results) == 6, f"应匹配 6 字（框1: 4字 + 框2: 2字），实际={len(results)}"
        q0_chars = [r["char"] for r in results if r["question_idx"] == 0]
        q1_chars = [r["char"] for r in results if r["question_idx"] == 1]
        assert q0_chars == ["随", "君", "直", "到"]
        assert q1_chars == ["人", "闲"]


# ============================================================
# 无中文字符处理
# ============================================================

class TestNonChineseChars:
    """无中文字符的 OCR 记录被跳过。"""

    def test_pinyin_only_skipped(self):
        """纯拼音 OCR 记录被跳过（无中文字符）。"""
        img = _make_img()
        hw_boxes = [_make_hw_box(100, 100, 500, 200, question_idx=0)]
        ocr_records = [_make_ocr_record("suí jūn", 150, 120, 350, 180)]

        results = recognize_handwriting(img, hw_boxes, ocr_records, page_idx=0)
        assert len(results) == 0

    def test_mixed_chinese_pinyin(self):
        """中英混合的 OCR 记录只保留中文字符。"""
        img = _make_img()
        hw_boxes = [_make_hw_box(100, 100, 500, 200, question_idx=0)]
        ocr_records = [_make_ocr_record("随jun君", 150, 120, 350, 180)]

        results = recognize_handwriting(img, hw_boxes, ocr_records, page_idx=0)
        chars = [r["char"] for r in results]
        assert chars == ["随", "君"], f"应只保留中文字符，实际={chars}"


# ============================================================
# img_pixel_w/h 字段验证
# ============================================================

class TestImgPixelFields:
    """验证结果中的 img_pixel_w/h 字段与输入图像尺寸一致。"""

    def test_img_pixel_fields_match_image_size(self):
        """img_pixel_w/h 与输入图像尺寸一致。"""
        img = _make_img(width=1653, height=2312)
        hw_boxes = [_make_hw_box(100, 100, 500, 200, question_idx=0)]
        ocr_records = [_make_ocr_record("随", 150, 120, 250, 180)]

        results = recognize_handwriting(img, hw_boxes, ocr_records, page_idx=0)
        assert len(results) == 1
        assert results[0]["img_pixel_w"] == 1653
        assert results[0]["img_pixel_h"] == 2312
