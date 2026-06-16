@echo off
cd /d "%~dp0"
start /B pythonw monitor\watcher.py
echo 监控服务已启动（后台运行）
pause
