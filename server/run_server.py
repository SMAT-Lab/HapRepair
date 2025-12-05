#!/usr/bin/env python3
"""
简单的服务器启动脚本，避免文件监控问题
"""

import uvicorn
import os
import sys
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

def main():
    # 获取配置
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    debug = os.getenv("DEBUG", "False").lower() == "true"
    
    print(f"Starting ArkTS Code Repair Service...")
    print(f"Host: {host}")
    print(f"Port: {port}")
    print(f"Debug mode: {debug}")
    
    # 检查必需的环境变量
    if not os.getenv("OPENAI_API_KEY"):
        print("Warning: OPENAI_API_KEY not set")
    
    # 启动服务器
    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        reload=True,  # 禁用自动重载以避免文件监控问题
        log_level="info",
        workers=1 if debug else 4
    )

if __name__ == "__main__":
    main()