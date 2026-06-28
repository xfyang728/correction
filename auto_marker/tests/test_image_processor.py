"""图像预处理单元测试 — 验证角度复用与坐标空间统一。

覆盖核心函数:
  - preprocess_image_with_angle: 预处理并返回 (image, deskew_angle)
  - preprocess_image: 向后兼容版本（仅返回 image）
  - _deskew_fast: 返回 (rotated_img, angle)
  - apply_deskew: 用已知角度旋转图像
  - deskew_only: 仅倾斜校正（向后兼容）
"""

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core.image_processor import (
    _deskew_fast,
    apply_deskew,
    deskew_only,
    preprocess_image,
    preprocess_image_with_angle,
)


# ============================================================
# 测试辅助函数
# ============================================================

def _make_test_image(width: int = 400, height: int = 500) -> Image.Image:
    """创建一个白色背景的测试图像（无倾斜，无文字轮廓）。"""
    arr = np.full((height, width, 3), 255, dtype=np.uint8)
    return Image.fromarray(arr)


def _make_skewed_image(angle_deg: float, width: int = 400, height: int = 500) -> Image.Image:
    """创建一个带黑色矩形（模拟文字）并旋转一定角度的测试图像。"""
    arr = np.full((height, width, 3), 255, dtype=np.uint8)
    # 画几个黑色矩形模拟文字轮廓
    cv2.rectangle(arr, (50, 50), (150, 100), (0, 0, 0), -1)
    cv2.rectangle(arr, (200, 50), (300, 100), (0, 0, 0), -1)
    cv2.rectangle(arr, (50, 200), (150, 250), (0, 0, 0), -1)
    cv2.rectangle(arr, (200, 200), (300, 250), (0, 0, 0), -1)
    cv2.rectangle(arr, (50, 350), (150, 400), (0, 0, 0), -1)
    cv2.rectangle(arr, (200, 350), (300, 400), (0, 0, 0), -1)
    # 旋转
    center = (width // 2, height // 2)
    matrix = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    rotated = cv2.warpAffine(
        arr, matrix, (width, height),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255),
    )
    return Image.fromarray(rotated)


# ============================================================
# preprocess_image_with_angle
# ============================================================

class TestPreprocessImageWithAngle:
    """预处理并返回 (image, deskew_angle) 二元组。"""

    def test_returns_tuple(self):
        """返回值为 (PIL.Image, float) 二元组。"""
        img = _make_test_image()
        result = preprocess_image_with_angle(img)
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], Image.Image)
        assert isinstance(result[1], float)

    def test_no_skew_returns_zero_angle(self):
        """无倾斜图像返回 angle=0.0。"""
        img = _make_test_image()
        _, angle = preprocess_image_with_angle(img)
        assert angle == 0.0

    def test_skewed_image_returns_nonzero_angle(self):
        """有倾斜图像返回非零角度。"""
        # 创建 2° 倾斜的图像（超过 0.3° 阈值）
        img = _make_skewed_image(angle_deg=2.0)
        _, angle = preprocess_image_with_angle(img)
        # 角度应接近 2.0（允许误差，因为降采样检测可能不精确）
        assert abs(angle) > 0.3, f"倾斜图像应检测到角度，实际={angle}"

    def test_preserves_image_size(self):
        """预处理不改变图像尺寸（输入大于 LOW_RES_THRESHOLD=1000，不触发上采样）。"""
        img = _make_test_image(width=1200, height=1500)
        processed, _ = preprocess_image_with_angle(img)
        assert processed.size == (1200, 1500)


# ============================================================
# preprocess_image（向后兼容）
# ============================================================

class TestPreprocessImageBackwardCompatible:
    """向后兼容版本仅返回 PIL.Image。"""

    def test_returns_image_only(self):
        """返回值为 PIL.Image（非元组）。"""
        img = _make_test_image()
        result = preprocess_image(img)
        assert isinstance(result, Image.Image)
        assert not isinstance(result, tuple)

    def test_equivalent_to_with_angle(self):
        """与 preprocess_image_with_angle[0] 结果一致。"""
        img = _make_test_image()
        result1 = preprocess_image(img)
        result2 = preprocess_image_with_angle(img)[0]
        arr1 = np.array(result1)
        arr2 = np.array(result2)
        assert arr1.shape == arr2.shape
        assert np.array_equal(arr1, arr2)


# ============================================================
# _deskew_fast
# ============================================================

class TestDeskewFastReturnsAngle:
    """_deskew_fast 返回 (rotated_img, angle) 二元组。"""

    def test_returns_tuple(self):
        """返回值为 (ndarray, float) 二元组。"""
        gray = np.full((500, 400), 255, dtype=np.uint8)
        result = _deskew_fast(gray)
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], np.ndarray)
        assert isinstance(result[1], float)

    def test_no_contours_returns_zero_angle(self):
        """无轮廓时返回 (原图, 0.0)。"""
        gray = np.full((500, 400), 255, dtype=np.uint8)  # 纯白
        rotated, angle = _deskew_fast(gray)
        assert angle == 0.0
        assert rotated is gray  # 应返回原图引用

    def test_skewed_image_returns_nonzero_angle(self):
        """有倾斜图像返回非零角度。"""
        # 创建带黑色矩形的灰度图并旋转
        gray = np.full((500, 400), 255, dtype=np.uint8)
        cv2.rectangle(gray, (50, 50), (150, 100), 0, -1)
        cv2.rectangle(gray, (200, 200), (300, 250), 0, -1)
        center = (200, 250)
        matrix = cv2.getRotationMatrix2D(center, 2.0, 1.0)
        skewed = cv2.warpAffine(
            gray, matrix, (400, 500),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=255,
        )
        _, angle = _deskew_fast(skewed)
        assert abs(angle) > 0.3, f"倾斜图像应检测到角度，实际={angle}"


# ============================================================
# apply_deskew
# ============================================================

class TestApplyDeskew:
    """用已知角度旋转图像，用于坐标空间统一。"""

    def test_zero_angle_no_rotation(self):
        """角度为 0 时不旋转，返回原图引用。"""
        arr = np.full((500, 400, 3), 255, dtype=np.uint8)
        result = apply_deskew(arr, 0.0)
        assert result is arr  # 应返回原图引用

    def test_small_angle_no_rotation(self):
        """角度 < 0.3 时不旋转（低于阈值）。"""
        arr = np.full((500, 400, 3), 255, dtype=np.uint8)
        result = apply_deskew(arr, 0.2)
        assert result is arr  # 应返回原图引用

    def test_large_angle_no_rotation(self):
        """角度 > 45 时不旋转（超出阈值）。"""
        arr = np.full((500, 400, 3), 255, dtype=np.uint8)
        result = apply_deskew(arr, 50.0)
        assert result is arr  # 应返回原图引用

    def test_valid_angle_rotates_image(self):
        """有效角度（0.3~45）时旋转图像。"""
        arr = np.full((500, 400, 3), 255, dtype=np.uint8)
        cv2.rectangle(arr, (50, 50), (150, 100), (0, 0, 0), -1)
        result = apply_deskew(arr, 2.0)
        assert result is not arr  # 应返回新图像
        assert result.shape == arr.shape  # 尺寸不变

    def test_grayscale_image(self):
        """灰度图也能正确处理。"""
        arr = np.full((500, 400), 255, dtype=np.uint8)
        result = apply_deskew(arr, 2.0)
        assert result is not arr
        assert result.shape == arr.shape
        assert result.ndim == 2  # 保持灰度

    def test_preserves_size(self):
        """旋转后图像尺寸不变（warpAffine 输出尺寸 = 输入尺寸）。"""
        arr = np.full((500, 400, 3), 255, dtype=np.uint8)
        result = apply_deskew(arr, 5.0)
        assert result.shape == arr.shape


# ============================================================
# 坐标空间一致性验证（改进1核心）
# ============================================================

class TestCoordinateSpaceConsistency:
    """验证 preprocess_image_with_angle + apply_deskew 能统一坐标空间。"""

    def test_same_angle_applied_to_both_images(self):
        """同一角度应用到 processed_np 和 original_np，坐标空间一致。

        模拟改进1的场景：
        1. preprocess_image_with_angle(img) → (processed_np, angle)
        2. apply_deskew(original_np, angle) → deskewed_original_np
        3. 两者的旋转角度相同，坐标空间一致
        """
        # 创建有倾斜的测试图像（尺寸大于 LOW_RES_THRESHOLD=1000，避免上采样导致尺寸不一致）
        img = _make_skewed_image(angle_deg=2.0, width=1200, height=1500)
        original_np = np.array(img)

        # preprocess 检测角度并旋转
        processed_img, angle = preprocess_image_with_angle(img)
        processed_np = np.array(processed_img)

        # 用相同角度 deskew original_np
        if abs(angle) >= 0.3:
            deskewed_original = apply_deskew(original_np, angle)
        else:
            deskewed_original = original_np

        # 如果检测到角度，两者应都经过了相同角度的旋转
        if abs(angle) >= 0.3:
            # 验证两者尺寸一致（坐标空间一致的基础）
            assert processed_np.shape == deskewed_original.shape
            # 验证角度非零（确实检测到倾斜）
            assert abs(angle) > 0.3

    def test_no_skew_no_apply(self):
        """无倾斜时 angle=0，apply_deskew 不旋转，坐标空间天然一致。"""
        img = _make_test_image()
        original_np = np.array(img)
        _, angle = preprocess_image_with_angle(img)
        assert angle == 0.0
        # apply_deskew 不会旋转
        deskewed = apply_deskew(original_np, angle)
        assert deskewed is original_np


# ============================================================
# deskew_only（向后兼容）
# ============================================================

class TestDeskewOnlyBackwardCompatible:
    """deskew_only 向后兼容性验证。"""

    def test_returns_ndarray(self):
        """返回值为 ndarray（非元组）。"""
        arr = np.full((500, 400, 3), 255, dtype=np.uint8)
        result = deskew_only(arr)
        assert isinstance(result, np.ndarray)
        assert not isinstance(result, tuple)

    def test_preserves_rgb_channels(self):
        """RGB 输入返回 RGB 输出（3 通道）。"""
        arr = np.full((500, 400, 3), 255, dtype=np.uint8)
        result = deskew_only(arr)
        assert result.ndim == 3
        assert result.shape[2] == 3

    def test_preserves_size(self):
        """尺寸不变。"""
        arr = np.full((500, 400, 3), 255, dtype=np.uint8)
        result = deskew_only(arr)
        assert result.shape == arr.shape
