#!/usr/bin/env python3
"""
简易 FTP 服务器 —— 基于 pyftpdlib

用法:
    python ftp_server.py                        # 匿名模式，端口 2121
    python ftp_server.py --port=21               # 自定义端口
    python ftp_server.py --user=admin --passwd=123456  # 用户认证模式
    python ftp_server.py --dir=/srv/ftp          # 共享目录
    python ftp_server.py --max-conn=50           # 最大连接数
"""

import argparse
import logging
import os
import sys

from pyftpdlib.authorizers import DummyAuthorizer
from pyftpdlib.handlers import FTPHandler
from pyftpdlib.servers import FTPServer

logger = logging.getLogger(__name__)


class ColoredFormatter(logging.Formatter):
    """简易的控制台颜色输出"""
    grey = "\x1b[90m"
    blue = "\x1b[94m"
    yellow = "\x1b[93m"
    red = "\x1b[91m"
    reset = "\x1b[0m"

    FORMATS = {
        logging.DEBUG: grey,
        logging.INFO: blue,
        logging.WARNING: yellow,
        logging.ERROR: red,
    }

    def format(self, record):
        color = self.FORMATS.get(record.levelno, self.grey)
        record.levelname = f"{color}{record.levelname}{self.reset}"
        return super().format(record)


class EveryoneAwareAuthorizer(DummyAuthorizer):
    """增强版 DummyAuthorizer，兼容 Windows 资源管理等客户端
    发送的 'Everyone' / 'guest' 等匿名用户名。

    将这些别名统一映射为 'anonymous' 处理。
    """
    ANONYMOUS_ALIASES = {"anonymous", "ftp", "everyone", "guest"}

    def _resolve(self, username: str) -> str:
        if username.lower() in self.ANONYMOUS_ALIASES:
            return "anonymous"
        return username

    def validate_authentication(self, username, password, handler):
        super().validate_authentication(self._resolve(username), password, handler)

    def has_user(self, username):
        return super().has_user(self._resolve(username))

    def get_home_dir(self, username):
        return super().get_home_dir(self._resolve(username))

    def get_perms(self, username):
        return super().get_perms(self._resolve(username))

    def has_perm(self, username, perm, path=None):
        return super().has_perm(self._resolve(username), perm, path)

    def get_msg_login(self, username):
        return super().get_msg_login(self._resolve(username))

    def get_msg_quit(self, username):
        return super().get_msg_quit(self._resolve(username))

    def impersonate_user(self, username, password):
        return super().impersonate_user(self._resolve(username), password)

    def terminate_impersonation(self, username):
        return super().terminate_impersonation(self._resolve(username))


class CustomFTPHandler(FTPHandler):
    """可扩展的自定义 Handler"""

    def on_connect(self):
        logger.info(f"[连入] {self.remote_ip}:{self.remote_port}")

    def on_disconnect(self):
        logger.info(f"[断开] {self.remote_ip}:{self.remote_port}")

    def on_login(self, username):
        logger.info(f"[登录] {username} 来自 {self.remote_ip}")

    def on_logout(self, username):
        logger.info(f"[登出] {username}")

    def on_file_sent(self, file):
        logger.info(f"[下载] {file}  -> {self.remote_ip}")

    def on_file_received(self, file):
        logger.info(f"[上传] {file}  <- {self.remote_ip}")

    def on_incomplete_file_sent(self, file):
        logger.warning(f"[下载中断] {file}")

    def on_incomplete_file_received(self, file):
        logger.warning(f"[上传中断] {file}")


def build_authorizer(anonymous: bool, user: str | None, passwd: str | None,
                     share_dir: str) -> EveryoneAwareAuthorizer:
    """构建授权器：匿名模式 或 用户认证模式"""
    authorizer = EveryoneAwareAuthorizer()

    if anonymous:
        # 匿名用户：可读可写
        authorizer.add_anonymous(share_dir, perm="elradfmwMT")
        logger.info("已开启匿名访问")
    else:
        username = user or "ftpuser"
        password = passwd or "ftp123"
        authorizer.add_user(username, password, share_dir, perm="elradfmwMT")
        logger.info(f"用户模式 — {username}:{password}")
    return authorizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="简易 FTP 服务器",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--host", default="0.0.0.0",
                        help="监听地址 (默认 0.0.0.0)")
    parser.add_argument("--port", type=int, default=2121,
                        help="监听端口 (默认 2121)")
    parser.add_argument("--dir", default=os.getcwd(),
                        help="共享根目录 (默认当前目录)")
    parser.add_argument("--user", default=None,
                        help="用户名 (不指定则匿名)")
    parser.add_argument("--passwd", default=None,
                        help="密码")
    parser.add_argument("--max-conn", type=int, default=100,
                        help="最大并发连接数 (默认 100)")
    parser.add_argument("--max-login", type=int, default=10,
                        help="同一 IP 最大登录数 (默认 10)")
    parser.add_argument("--banner", default="Welcome to Simple FTP Server",
                        help="FTP 欢迎标语")
    parser.add_argument("--passive-ports", default=None,
                        help="被动模式端口范围, 如 6000-7000")
    parser.add_argument("--masquerade", default=None,
                        help="NAT 后的公网地址 (masquerade)")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="详细日志输出 (DEBUG 级别)")
    return parser.parse_args()


def setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(ColoredFormatter(
        "%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    ))
    logging.basicConfig(level=level, handlers=[handler])


def main() -> None:
    args = parse_args()
    setup_logging(args.verbose)

    share_dir = os.path.abspath(args.dir)
    if not os.path.isdir(share_dir):
        logger.error(f"目录不存在: {share_dir}")
        sys.exit(1)

    # 授权
    anonymous = args.user is None
    authorizer = build_authorizer(anonymous, args.user, args.passwd, share_dir)

    # Handler
    handler = CustomFTPHandler
    handler.authorizer = authorizer
    handler.banner = args.banner

    # 被动模式端口范围
    if args.passive_ports:
        try:
            low, high = args.passive_ports.split("-")
            handler.passive_ports = list(range(int(low), int(high) + 1))  # type: ignore[assignment]
        except (ValueError, TypeError):
            logger.error(f"被动端口格式错误, 应为 6000-7000: {args.passive_ports}")
            sys.exit(1)

    # NAT masquerade
    if args.masquerade:
        handler.masquerade_address = args.masquerade

    # 服务器
    server = FTPServer((args.host, args.port), handler)

    server.max_cons = args.max_conn
    server.max_cons_per_ip = args.max_login

    # 打印启动信息
    local_ip = args.host if args.host != "0.0.0.0" else "127.0.0.1"
    print("-" * 50)
    print(f"  FTP 服务器已启动")
    print(f"  地址:      {args.host}:{args.port}")
    print(f"  共享目录:  {share_dir}")
    print(f"  匿名模式:  {'是' if anonymous else '否'}")
    print(f"  最大连接:  {args.max_conn}")
    print(f"  被动端口:  {args.passive_ports or '系统自动分配'}")
    print(f"  本地测试:  ftp://{local_ip}:{args.port}")
    print("-" * 50)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("收到中断信号，正在关闭服务器...")
        server.close_all()
        logger.info("服务器已关闭")


def run_ftp_server(
    host="0.0.0.0",
    port=2121,
    directory=None,
    anonymous=True,
    username=None,
    password=None,
    max_conn=100,
    max_login=10,
    banner=None,
    passive_ports=None,
    masquerade=None,
    verbose=False,
):
    """供其他模块嵌入调用的入口，返回 (FTPServer, stop_fn) 元组。

    调用方可通过 stop_fn() 关闭服务，或直接 server.close_all()。
    """
    directory = directory or os.getcwd()

    if not os.path.isdir(directory):
        raise NotADirectoryError(f"目录不存在: {directory}")

    authorizer = build_authorizer(anonymous, username, password, directory)

    handler = CustomFTPHandler
    handler.authorizer = authorizer
    handler.banner = banner or f"Welcome to Simple FTP Server"

    if passive_ports:
        low, high = passive_ports
        handler.passive_ports = list(range(low, high + 1))  # type: ignore[assignment]

    if masquerade:
        handler.masquerade_address = masquerade

    server = FTPServer((host, port), handler)
    server.max_cons = max_conn
    server.max_cons_per_ip = max_login

    logger.info("FTP 服务器已就绪 %s:%s → %s", host, port, directory)
    return server


if __name__ == "__main__":
    main()
