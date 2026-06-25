"""
图像预处理 — 对扫描页面进行去噪、二值化、增强、倾斜校正、质量筛查。
"""

import logging

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger("image_processor")

# 模糊检测阈值（拉普拉斯方差），低于此值认为图片模糊
# 手拍试卷通常偏模糊，适当降低阈值
BLUR_THRESHOLD = 60

# 低分辨率检测阈值（最小边长低于此值则上采样）
LOW_RES_THRESHOLD = 1000


def preprocess_image(img: Image.Image) -> Image.Image:
    """对扫描页面图片进行预处理，提升 OCR 质量。

    处理流程：
        1. 灰度化
        2. 低分辨率检测 + 上采样
        3. 中值滤波去噪
        4. CLAHE 对比度增强（clipLimit=3.0，适应手拍光照不均）
        5. 倾斜校正（阈值 0.3°，比默认更积极）
        6. 模糊质量检测（日志警告，不阻断）

    参数:
        img: PIL Image (RGB)

    返回:
        预处理后的 PIL Image (RGB)
    """
    # PIL → numpy
    img_rgb = np.array(img)
    if len(img_rgb.shape) == 2:
        gray = img_rgb
    else:
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)

    # 0. 低分辨率检测 & 上采样
    h, w = gray.shape[:2]
    if min(h, w) < LOW_RES_THRESHOLD:
        h_orig, w_orig = h, w
        scale = LOW_RES_THRESHOLD / min(h, w)
        new_w, new_h = int(w * scale), int(h * scale)
        gray = cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        h, w = gray.shape[:2]
        logger.info("低分辨率图像 %dx%d → 上采样到 %dx%d", w_orig, h_orig, new_w, new_h)

    # 1. 去噪 — 中值滤波
    denoised = cv2.medianBlur(gray, 3)

    # 2. 对比度增强 — CLAHE（clipLimit 提升到 3.0，手拍光照不均更严重）
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(denoised)

    # 3. 倾斜校正（降采样加速 — 对大图先缩小再检测角度）
    deskewed = _deskew_fast(enhanced)

    # 4. 质量筛查 — 模糊检测
    blur_score = cv2.Laplacian(deskewed, cv2.CV_64F).var()
    if blur_score < BLUR_THRESHOLD:
        logger.warning("图片模糊度 %.1f（阈值 %.0f），可能影响 OCR 质量", blur_score, BLUR_THRESHOLD)
    else:
        logger.debug("模糊度评分: %.1f", blur_score)

    # numpy → PIL（转回 RGB 三通道，保持接口一致）
    result = cv2.cvtColor(deskewed, cv2.COLOR_GRAY2RGB)
    return Image.fromarray(result)


def _deskew_fast(img: np.ndarray) -> np.ndarray:
    """快速倾斜校正 — 降采样检测角度，全分辨率校正。

    对大图（如 2481×3508）先缩小到 800px 检测角度，
    避免在大图上做 findContours 耗时过长。
    """
    h, w = img.shape[:2]
    max_dim = max(h, w)

    # 大图降采样到 800px 检测角度
    if max_dim > 800:
        scale = 800 / max_dim
        small = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    else:
        small = img

    # 反转以检测文字轮廓
    if np.mean(small) > 127:
        inv = cv2.bitwise_not(small)
    else:
        inv = small.copy()

    contours, _ = cv2.findContours(inv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return img

    min_area = small.shape[0] * small.shape[1] * 0.0005
    valid_contours = [c for c in contours if cv2.contourArea(c) > min_area]
    if not valid_contours:
        return img

    angles = []
    for c in valid_contours:
        rect = cv2.minAreaRect(c)
        angle = rect[2]
        if angle < -45:
            angle = 90 + angle
        angles.append(angle)

    if not angles:
        return img

    median_angle = float(np.median(angles))
    # PP-OCRv6 增强：阈值从 0.5° 降低到 0.3°，对手拍轻微倾斜更积极校正
    if abs(median_angle) > 45 or abs(median_angle) < 0.3:
        return img

    logger.info("检测到倾斜角度: %.2f°，执行校正", median_angle)
    center = (w // 2, h // 2)
    matrix = cv2.getRotationMatrix2D(center, median_angle, 1.0)
    rotated = cv2.warpAffine(
        img, matrix, (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255),
    )
    return rotated


def deskew_only(img_array: np.ndarray) -> np.ndarray:
    """P1-5: 仅做倾斜校正，不含 CLAHE/去噪，保留原始像素分布。

    用于 Qwen3-VL 路径（可配置开关），避免 CLAHE 损害浅色铅笔字。
    输入/输出均为 RGB numpy array (H, W, 3)。
    """
    if img_array.ndim == 3:
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    else:
        gray = img_array
    deskewed_gray = _deskew_fast(gray)
    if deskewed_gray.ndim == 2 and img_array.ndim == 3:
        return cv2.cvtColor(deskewed_gray, cv2.COLOR_GRAY2RGB)
    return deskewed_gray
