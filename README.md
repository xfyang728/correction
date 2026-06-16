# Simple FTP Server

基于 [pyftpdlib](https://github.com/giampaolo/pyftpdlib) 的简易 FTP 服务器，单文件即用。

## 安装

```bash
pip install pyftpdlib
```

## 快速启动

```bash
# 匿名模式，共享当前目录，端口 2121
python ftp_server.py

# 指定共享目录
python ftp_server.py --dir=D:\scan

# 用户认证模式（指定 --user 自动启用）
python ftp_server.py --port=21 --user=admin --passwd=123456 --dir=/srv/ftp
```

## 参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--host` | `0.0.0.0` | 监听地址 |
| `--port` | `2121` | 监听端口 |
| `--dir` | 当前目录 | 共享根目录 |
| `--user` | — | 用户名（不指定则为匿名模式） |
| `--passwd` | — | 密码（不指定则默认 `ftp123`） |
| `--max-conn` | `100` | 最大并发连接数 |
| `--max-login` | `10` | 同一 IP 最大登录数 |
| `--passive-ports` | — | 被动模式端口范围，如 `6000-7000` |
| `--masquerade` | — | NAT 后的公网地址 |
| `--banner` | `Welcome...` | FTP 欢迎标语 |
| `-v` / `--verbose` | — | 详细日志（DEBUG） |

## 兼容 Windows 资源管理器

Windows 资源管理器的 FTP 客户端使用 `Everyone` 作为匿名登录用户名，本服务器已内置兼容，无需额外配置。

## 文件操作

```bash
# 列出文件
ftp 127.0.0.1 2121

# 上传
put local.txt

# 下载
get remote.txt
```
