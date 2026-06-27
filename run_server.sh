#!/bin/sh
# 启动 S600 端侧大模型 OpenAI 兼容服务。
# 用法：./run_server.sh            # 前台运行（Ctrl-C 退出）
#       setsid ./run_server.sh </dev/null >>server.log 2>&1 &   # 后台常驻
cd "$(dirname "$0")"
export PATH="$HOME/.local/bin:$PATH"
export HOST="${HOST:-0.0.0.0}"
export PORT="${PORT:-8000}"
exec uv run uvicorn server.app:app --host "$HOST" --port "$PORT"
