"""PDF 批注器单元测试 — 验证文字渲染基线垂直居中。

覆盖核心函数:
  - _draw_text_at_bbox: 文字渲染基线计算（垂直居中公式）
"""
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


class TestDrawTextAtBbox:
    """验证 _draw_text_at_bbox 的基线垂直居中公式。

    修复 Y 偏下问题：当 font_size 被 24pt 上限截断且 bh 较大时，
    旧公式 text_y = bbox_bottom + bh * 0.2 使文字偏下；
    新公式 text_y = bbox_bottom + (bh - font_size * 0.8) / 2 使文字垂直居中。
    """

    def test_text_vertical_centering(self):
        """文字视觉中心 = bbox 中心（垂直居中）。

        数学验证:
        - 文字视觉中心 = text_y + 0.4 * font_size（基线 + 半字高）
        - bbox 中心 = bbox_bottom + bh / 2
        - 代入新公式: text_y = bbox_bottom + (bh - 0.8*font_size)/2
          → text_y + 0.4*font_size = bbox_bottom + (bh - 0.8*fs)/2 + 0.4*fs
                                    = bbox_bottom + bh/2 - 0.4*fs + 0.4*fs
                                    = bbox_bottom + bh/2 ✓
        """
        from core.pdf_annotator import _draw_text_at_bbox

        # 模拟 canvas
        can = MagicMock()
        can.saveState = MagicMock()
        can.restoreState = MagicMock()
        can.setFont = MagicMock()
        can.setFillColorRGB = MagicMock()
        can.rect = MagicMock()
        can.drawString = MagicMock()

        # bbox: 100x100pt，font_size 会被截断到 24pt（bh*0.9=90 > 24）
        bbox_pixel = (100, 100, 200, 200)  # x0, y0, x1, y1（像素）
        ocr_img_w, ocr_img_h = 1000, 1000
        page_width, page_height = 500, 500  # PDF 点

        _draw_text_at_bbox(can, "随", bbox_pixel, ocr_img_w, ocr_img_h,
                           page_width, page_height, color=(0, 1, 0), alpha=0.7)

        # 验证 drawString 被调用
        assert can.drawString.called, "drawString 应被调用"
        call_args = can.drawString.call_args
        text_x, text_y = call_args[0][0], call_args[0][1]
        char = call_args[0][2]

        assert char == "随"

        # 计算 bbox 在 PDF 坐标系的位置
        # _pixel_to_page: x_page = x_pixel / img_w * page_w
        #                 y_page = page_h - y_pixel / img_h * page_h
        x0_page = 100 / 1000 * 500  # 50
        x1_page = 200 / 1000 * 500  # 100
        y0_page = 500 - 100 / 1000 * 500  # 450 (top)
        y1_page = 500 - 200 / 1000 * 500  # 400 (bottom)
        bh = abs(y1_page - y0_page)  # 50pt
        font_size = max(min(bh * 0.9, 24), 8)  # min(45, 24) = 24

        # 验证垂直居中：text_y = bbox_bottom + (bh - font_size * 0.8) / 2
        bbox_bottom = min(y0_page, y1_page)  # 400
        expected_text_y = bbox_bottom + (bh - font_size * 0.8) / 2
        # = 400 + (50 - 19.2) / 2 = 400 + 15.4 = 415.4

        assert abs(text_y - expected_text_y) < 0.01, \
            f"text_y={text_y}, 预期 {expected_text_y} (垂直居中)"

        # 验证文字视觉中心 = bbox 中心
        text_visual_center = text_y + 0.4 * font_size
        bbox_center = bbox_bottom + bh / 2
        assert abs(text_visual_center - bbox_center) < 0.01, \
            f"文字视觉中心 {text_visual_center} 应等于 bbox 中心 {bbox_center}"

    def test_font_size_capped_vertical_centering(self):
        """bh 较大时 font_size=24（被截断），text_y 仍能垂直居中。

        场景: bh=80pt（大 bbox），font_size=24pt
              旧公式: text_y = bbox_bottom + 0.2*80 = bbox_bottom + 16
                      文字位于 bbox 20%-44% 区间（偏下）
              新公式: text_y = bbox_bottom + (80 - 19.2)/2 = bbox_bottom + 30.4
                      文字视觉中心 = 30.4 + 9.6 = 40 = bbox 中心 (80/2) ✓
        """
        from core.pdf_annotator import _draw_text_at_bbox

        can = MagicMock()
        can.saveState = MagicMock()
        can.restoreState = MagicMock()
        can.setFont = MagicMock()
        can.setFillColorRGB = MagicMock()
        can.rect = MagicMock()
        can.drawString = MagicMock()

        # bbox: 200x160 像素 → 在 PDF 中 bh 较大
        bbox_pixel = (100, 100, 300, 260)
        ocr_img_w, ocr_img_h = 1000, 1000
        page_width, page_height = 500, 500

        _draw_text_at_bbox(can, "随", bbox_pixel, ocr_img_w, ocr_img_h,
                           page_width, page_height, color=(1, 0, 0), alpha=0.7)

        call_args = can.drawString.call_args
        text_y = call_args[0][1]

        # 计算
        y0_page = 500 - 100 / 1000 * 500  # 450
        y1_page = 500 - 260 / 1000 * 500  # 370
        bh = abs(y1_page - y0_page)  # 80pt
        font_size = max(min(bh * 0.9, 24), 8)  # 24（被截断）
        bbox_bottom = min(y0_page, y1_page)  # 370

        # 验证 font_size 被截断到 24
        assert font_size == 24, f"font_size 应被截断到 24, 实际 {font_size}"

        # 验证新公式：text_y = bbox_bottom + (bh - font_size * 0.8) / 2
        expected_text_y = bbox_bottom + (bh - font_size * 0.8) / 2
        # = 370 + (80 - 19.2) / 2 = 370 + 30.4 = 400.4

        assert abs(text_y - expected_text_y) < 0.01, \
            f"大 bbox 时 text_y={text_y}, 预期 {expected_text_y} (垂直居中)"

        # 关键验证：文字视觉中心 = bbox 中心（不再偏下）
        text_visual_center = text_y + 0.4 * font_size
        bbox_center = bbox_bottom + bh / 2
        assert abs(text_visual_center - bbox_center) < 0.01, \
            f"大 bbox 文字视觉中心 {text_visual_center} 应等于 bbox 中心 {bbox_center}（不再偏下）"

    def test_small_bbox_font_size_not_capped(self):
        """小 bbox 时 font_size 不被截断，垂直居中仍成立。"""
        from core.pdf_annotator import _draw_text_at_bbox

        can = MagicMock()
        can.saveState = MagicMock()
        can.restoreState = MagicMock()
        can.setFont = MagicMock()
        can.setFillColorRGB = MagicMock()
        can.rect = MagicMock()
        can.drawString = MagicMock()

        # bbox: 50x30 像素 → 小 bbox，font_size 不被截断
        bbox_pixel = (100, 100, 150, 130)
        ocr_img_w, ocr_img_h = 1000, 1000
        page_width, page_height = 500, 500

        _draw_text_at_bbox(can, "随", bbox_pixel, ocr_img_w, ocr_img_h,
                           page_width, page_height, color=(0, 1, 0), alpha=0.7)

        call_args = can.drawString.call_args
        text_y = call_args[0][1]

        y0_page = 500 - 100 / 1000 * 500  # 450
        y1_page = 500 - 130 / 1000 * 500  # 435
        bh = abs(y1_page - y0_page)  # 15pt
        font_size = max(min(bh * 0.9, 24), 8)  # min(13.5, 24) = 13.5
        bbox_bottom = min(y0_page, y1_page)  # 435

        # 验证 font_size 未被截断
        assert font_size == 13.5, f"小 bbox font_size={font_size}, 预期 13.5"

        # 验证垂直居中
        expected_text_y = bbox_bottom + (bh - font_size * 0.8) / 2
        assert abs(text_y - expected_text_y) < 0.01, \
            f"小 bbox text_y={text_y}, 预期 {expected_text_y}"

        text_visual_center = text_y + 0.4 * font_size
        bbox_center = bbox_bottom + bh / 2
        assert abs(text_visual_center - bbox_center) < 0.01, \
            f"小 bbox 文字视觉中心 {text_visual_center} 应等于 bbox 中心 {bbox_center}"
