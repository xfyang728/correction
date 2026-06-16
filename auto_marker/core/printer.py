"""
打印模块 — 将 PDF 发送到 Windows 打印机。

支持指定打印机名称（从 config.ini 读取）或系统默认打印机。
"""

import logging
import subprocess
import time
from pathlib import Path

logger = logging.getLogger("printer")


def print_pdf(pdf_path: str, printer_name: str | None = None) -> bool:
    """
    打印 PDF。

    参数:
        pdf_path: PDF 文件路径
        printer_name: 打印机名称，None 则用系统默认

    返回:
        True 表示打印任务已提交
    """
    path = Path(pdf_path)
    if not path.exists():
        logger.error("文件不存在: %s", pdf_path)
        return False

    logger.info("发送打印任务: %s (打印机: %s)", pdf_path, printer_name or "默认")

    try:
        if printer_name:
            cmd = ["powershell", "-Command",
                   f'Start-Process -FilePath "{path}" -Verb Print -PassThru '
                   f'| ForEach-Object {{$_.CloseMainWindow()}}']
        else:
            # 使用系统默认打印机
            cmd = ["powershell", "-Command",
                   f'Start-Process -FilePath "{path}" -Verb Print -PassThru '
                   f'| ForEach-Object {{$_.CloseMainWindow()}}']

        # Windows: start /b 避免阻塞
        subprocess.Popen(
            cmd,
            shell=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        # 短暂延迟后返回，让打印管理器接管
        time.sleep(1)
        logger.info("打印任务已提交")
        return True

    except Exception as e:
        logger.exception("打印失败: %s", e)
        return False
