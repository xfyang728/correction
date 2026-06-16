"""
监控服务 — 监听 incoming 目录，发现 PDF 后提交到流水线。

核心设计：ThreadPoolExecutor + 文件稳定检测。
"""

import os
import sys
import time
import logging
import configparser
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# 确保项目根目录在 sys.path 中
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from monitor.processor import process_pdf

logger = logging.getLogger("watcher")

# ---- 文件稳定检测 ----

def wait_for_file_stable(filepath: str, timeout: int = 15, interval: float = 0.5) -> bool:
    """等待文件写入完成：大小和最后修改时间不再变化。"""
    stable_count = 0
    needed = 3
    prev_size = -1
    prev_mtime = 0
    elapsed = 0.0
    while elapsed < timeout:
        try:
            stat = os.stat(filepath)
            cur_size = stat.st_size
            cur_mtime = stat.st_mtime
        except FileNotFoundError:
            return False
        if cur_size == prev_size and cur_mtime == prev_mtime:
            stable_count += 1
            if stable_count >= needed:
                return True
        else:
            stable_count = 0
        prev_size, prev_mtime = cur_size, cur_mtime
        time.sleep(interval)
        elapsed += interval
    logger.warning("文件在 %ss 内未稳定，跳过: %s", timeout, filepath)
    return False


# ---- Watchdog Handler ----

class PDFHandler(FileSystemEventHandler):
    """监听 PDF 文件创建/修改事件，提交到线程池处理。"""

    def __init__(self, executor: ThreadPoolExecutor):
        self.executor = executor

    def on_created(self, event):
        if not event.src_path.lower().endswith(".pdf"):
            return
        self.executor.submit(self._safe_process, event.src_path)

    def on_modified(self, event):
        # 部分客户端用修改事件而非创建事件
        if not event.src_path.lower().endswith(".pdf"):
            return
        self.executor.submit(self._safe_process, event.src_path)

    def _safe_process(self, filepath: str):
        try:
            logger.info("检测到新文件: %s", filepath)
            if not wait_for_file_stable(filepath):
                logger.error("文件不稳定，跳过: %s", filepath)
                return
            process_pdf(filepath)
        except Exception:
            logger.exception("处理文件时出错: %s", filepath)


# ---- 配置加载 ----

def load_config() -> configparser.ConfigParser:
    cfg = configparser.ConfigParser()
    cfg_path = ROOT / "config.ini"
    if cfg_path.exists():
        cfg.read(str(cfg_path))
    return cfg


# ---- 日志设置 ----

def setup_logging(cfg: configparser.ConfigParser):
    level = getattr(logging, cfg.get("log", "level", fallback="INFO").upper(), logging.INFO)
    log_dir = ROOT / (cfg.get("log", "dir", fallback="data/logs"))
    log_dir.mkdir(parents=True, exist_ok=True)

    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)

    # 文件 handler（按天滚动）
    from logging.handlers import TimedRotatingFileHandler
    fh = TimedRotatingFileHandler(str(log_dir / "monitor.log"), when="midnight",
                                  backupCount=7, encoding="utf-8")
    fh.setLevel(level)
    fh.setFormatter(fmt)
    root.addHandler(fh)

    # 控制台 handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(level)
    ch.setFormatter(fmt)
    root.addHandler(ch)

    return level


# ---- 入口 ----

def main():
    cfg = load_config()
    setup_logging(cfg)

    max_workers = cfg.getint("monitor", "max_workers", fallback=4)
    incoming_dir = ROOT / cfg.get("monitor", "incoming_dir", fallback="data/incoming")
    incoming_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 50)
    logger.info("监控服务启动")
    logger.info("  监听目录: %s", incoming_dir)
    logger.info("  最大并发: %s", max_workers)
    logger.info("=" * 50)

    executor = ThreadPoolExecutor(max_workers=max_workers)
    event_handler = PDFHandler(executor)
    observer = Observer()
    observer.schedule(event_handler, str(incoming_dir), recursive=False)
    observer.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("收到中断信号，正在关闭...")
        observer.stop()
        executor.shutdown(wait=False)
    observer.join()
    logger.info("监控服务已关闭")


if __name__ == "__main__":
    main()
