@echo off
chcp 65001 >nul
title YOLOv8 批量处理工具
echo ============================================
echo    YOLOv8m 批量图片视频处理程序
echo ============================================
echo.

:: 检查 Python
python --version >nul 2>&1
if errorlevel 1 (
    echo [错误] 未检测到 Python！
    pause
    exit /b
)

:: 检查依赖
echo 正在检查依赖...
pip show ultralytics >nul 2>&1
if errorlevel 1 (
    echo 正在安装 ultralytics...
    pip install ultralytics -i https://pypi.tuna.tsinghua.edu.cn/simple
)

pip show opencv-python >nul 2>&1
if errorlevel 1 (
    echo 正在安装 opencv-python...
    pip install opencv-python -i https://pypi.tuna.tsinghua.edu.cn/simple
)

echo.
echo 正在启动程序...
echo.
python yolo_batch.py

echo.
echo 程序已退出。
pause