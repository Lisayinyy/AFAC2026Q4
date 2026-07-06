#!/usr/bin/env bash
# AFAC2026 环境初始化 —— 安装可复现依赖 (满足赛题代码审核要求)
set -e

echo "[init_env] installing python packages..."
pip install -r requirements.txt

echo "[init_env] done. run:  python main.py"
