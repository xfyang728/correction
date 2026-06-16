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

    # 2. 二值化 — OTSU 自适应阈值
    _, binary = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 3. 对比度增强 — CLAHE（对二值图效果有限，但对灰度图有效）
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(denoised)

    # 合并二值化与增强：取两者优势
    # 对于清晰文本区域用二值化结果，对于过渡区域用增强结果
    combined = cv2.addWeighted(binary, 0.6, enhanced, 0.4, 0)

    # 4. 倾斜校正
    deskewed = _deskew(combined)

    # 5. 质量筛查 — 模糊检测
    blur_score = cv2.Laplacian(deskewed, cv2.CV_64F).var()
    if blur_score < BLUR_THRESHOLD:
        logger.warning("图片模糊度 %.1f（阈值 %.0f），可能影响 OCR 质量", blur_score, BLUR_THRESHOLD)
    else:
        logger.debug("模糊度评分: %.1f", blur_score)

    # numpy → PIL（转回 RGB 三通道，保持接口一致）
    result = cv2.cvtColor(deskewed, cv2.COLOR_GRAY2RGB)
    return Image.fromarray(result)


def _deskew(img: np.ndarray) -> np.ndarray:
    """检测图片倾斜角度并校正。

    通过查找所有文本轮廓的最小外接矩形，计算平均倾斜角。
    """
    # 反转（白字黑底 → 黑字白底）以检测文字轮廓
    if np.mean(img) > 127:
        inv = cv2.bitwise_not(img)
    else:
        inv = img.copy()

    # 查找轮廓
    contours, _ = cv2.findContours(inv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        logger.debug("未检测到轮廓，跳过倾斜校正")
        return img

    # 过滤小轮廓，只保留有意义文本区域
    min_area = img.shape[0] * img.shape[1] * 0.0005
    valid_contours = [c for c in contours if cv2.contourArea(c) > min_area]
    if not valid_contours:
        return img

    # 计算所有有效轮廓的最小外接矩形角度
    angles = []
    for c in valid_contours:
        rect = cv2.minAreaRect(c)
        angle = rect[2]
        # OpenCV 返回角度范围 [-90, 0)，需要转换
        if angle < -45:
            angle = 90 + angle
        angles.append(angle)

    if not angles:
        return img

    # 取中位数角度作为整体倾斜角（抗离群点）
    median_angle = np.median(angles)

    # 角度过大（> 45°）可能是假检测，跳过校正
    if abs(median_angle) > 45:
        logger.debug("检测到异常角度 %.2f°，跳过校正", median_angle)
        return img

    # 只有角度足够大时才校正（避免小幅抖动）
    if abs(median_angle) < 0.5:
        return img

    logger.info("检测到倾斜角度: %.2f°，执行校正", median_angle)
    h, w = img.shape[:2]
    center = (w // 2, h // 2)
    matrix = cv2.getRotationMatrix2D(center, median_angle, 1.0)
    # 旋转后保持原尺寸，自动填充黑色背景
    rotated = cv2.warpAffine(
        img, matrix, (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255),
    )
    return rotated