"""集中配置读取 — 从 config.ini 加载配置，供各模块统一使用。"""

import configparser
import logging
from pathlib import Path

logger = logging.getLogger("config")

_cfg: configparser.ConfigParser | None = None


def get_config() -> configparser.ConfigParser:
    """获取全局配置单例（线程安全的 lazy 初始化）。"""
    global _cfg
    if _cfg is None:
        _cfg = configparser.ConfigParser()
        cfg_path = Path(__file__).resolve().parent.parent / "config.ini"
        if cfg_path.exists():
            _cfg.read(cfg_path, encoding="utf-8")
            logger.debug("配置已加载: %s", cfg_path)
        else:
            logger.warning("config.ini 不存在: %s，使用默认值", cfg_path)
    return _cfg


# ---- OCR 配置 ----

def ocr_use_gpu() -> bool:
    return get_config().getboolean("ocr", "use_gpu", fallback=False)


def ocr_confidence_threshold() -> float:
    return get_config().getfloat("ocr", "confidence_threshold", fallback=0.6)


def ocr_dpi() -> int:
    return get_config().getint("ocr", "dpi", fallback=200)


# ---- Grader 配置 ----

def grader_green_threshold() -> float:
    return get_config().getfloat("grader", "green_threshold", fallback=0.85)


def grader_orange_threshold() -> float:
    return get_config().getfloat("grader", "orange_threshold", fallback=0.60)


def grader_skip_threshold() -> float:
    return get_config().getfloat("grader", "skip_threshold", fallback=0.30)


# ---- 版面分析配置 ----

def layout_handwritten_height_min() -> int:
    return get_config().getint("layout", "handwritten_height_min", fallback=90)


def layout_pinyin_height_max() -> int:
    return get_config().getint("layout", "pinyin_height_max", fallback=55)


def layout_column_gap_threshold() -> int:
    return get_config().getint("layout", "column_gap_threshold", fallback=150)


def layout_y_proximity_threshold() -> int:
    return get_config().getint("layout", "y_proximity_threshold", fallback=30)


def layout_sub_region_height_min() -> int:
    return get_config().getint("layout", "sub_region_height_min", fallback=130)


def layout_sub_region_ratio() -> float:
    return get_config().getfloat("layout", "sub_region_ratio", fallback=0.55)


# ---- 图像处理配置 ----

def image_blur_threshold() -> float:
    return get_config().getfloat("image", "blur_threshold", fallback=60)


def image_low_res_threshold() -> int:
    return get_config().getint("image", "low_res_threshold", fallback=1000)


# ---- 监控配置 ----

def monitor_max_workers() -> int:
    return get_config().getint("monitor", "max_workers", fallback=4)


def monitor_file_stable_timeout() -> int:
    return get_config().getint("monitor", "file_stable_timeout", fallback=15)


# ---- 打印配置 ----

def printer_name() -> str:
    return get_config().get("printer", "name", fallback="")
