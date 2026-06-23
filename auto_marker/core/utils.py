"""共享工具函数 — bbox 操作和字符判断，供各模块复用。"""


def is_chinese_char(ch: str) -> bool:
    """判断是否为中文字符。"""
    code = ord(ch)
    return (
        0x4E00 <= code <= 0x9FFF
        or 0x3400 <= code <= 0x4DBF
        or 0xF900 <= code <= 0xFAFF
    )


def bbox_center(bbox: tuple) -> tuple[float, float]:
    """返回 bbox 中心点 (cx, cy)。"""
    return ((bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0)


def bbox_center_x(bbox: tuple) -> float:
    """返回 bbox 中心 x。"""
    return (bbox[0] + bbox[2]) / 2.0


def bbox_center_y(bbox: tuple) -> float:
    """返回 bbox 中心 y。"""
    return (bbox[1] + bbox[3]) / 2.0


def bbox_h(bbox: tuple) -> int:
    """返回 bbox 高度。"""
    return bbox[3] - bbox[1]


def bbox_w(bbox: tuple) -> int:
    """返回 bbox 宽度。"""
    return bbox[2] - bbox[0]


def point_in_bbox(px: float, py: float, bbox: tuple) -> bool:
    """判断点 (px, py) 是否在 bbox (x0,y0,x2,y2) 内。"""
    return bbox[0] <= px <= bbox[2] and bbox[1] <= py <= bbox[3]
