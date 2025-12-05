#!/bin/bash

# ArkTS Code Repair Service 启动脚本

set -e

echo "Starting ArkTS Code Repair Service..."

# 增加文件监控限制
echo "Setting file watch limits..."
echo fs.inotify.max_user_watches=524288 | sudo tee -a /etc/sysctl.conf > /dev/null || true
sudo sysctl -p > /dev/null 2>&1 || true

# 检查Python版本
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Python version: $python_version"

# 检查是否存在虚拟环境
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# 激活虚拟环境
echo "Activating virtual environment..."
source venv/bin/activate

# 安装依赖
echo "Installing dependencies..."
#pip install -r requirements.txt

# 检查环境变量
if [ ! -f ".env" ]; then
    echo "Warning: .env file not found. Copying from .env.example..."
    cp .env.example .env
    echo "Please edit .env file with your API keys before running the service."
    exit 1
fi

# 检查必需的环境变量
source .env
if [ -z "$OPENAI_API_KEY" ]; then
    echo "Error: OPENAI_API_KEY is not set in .env file"
    exit 1
fi

# 创建必要的目录
mkdir -p logs
mkdir -p models

# 启动服务
echo "Starting FastAPI server..."
if [ "$1" = "dev" ]; then
    echo "Running in development mode..."
    export DEBUG=True
    python main.py
elif [ "$1" = "prod" ]; then
    echo "Running in production mode..."
    export DEBUG=False
    uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000} --workers 4
else
    echo "Usage: $0 [dev|prod]"
    echo "  dev  - Development mode with auto-reload"
    echo "  prod - Production mode with multiple workers"
    exit 1
fi