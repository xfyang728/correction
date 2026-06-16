"""
图像预处理 — 对扫描页面进行去噪、二值化、增强、倾斜校正、质量筛查。
"""

import logging
import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger("image_processor")

# 模糊检测阈值（拉普拉斯方差），低于此值认为图片模糊
BLUR_THRESHOLD = 80


def preprocess_image(img: Image.Image) -> Image.Image:
    """对扫描页面图片进行预处理，提升 OCR 质量。

    处理流程：
        1. 灰度化
        2. 中值滤波去噪
        3. OTSU 二值化
        4. CLAHE 对比度增强
        5. 倾斜校正
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

    # 1. 去噪 — 中值滤波
    denoised = cv2.medianBlur(gray, 3)

    # 2. 对比度增强 — CLAHE（保留手写笔画浓淡信息，不做二值化）
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(denoised)

    # 3. 倾斜校正（降采样加速 — 对大图先缩小再检测角度）
    deskewed = _deskew_fast(enhanced)

    # 5. 质量筛查 — 模糊检测
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
    if abs(median_angle) > 45 or abs(median_angle) < 0.5:
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