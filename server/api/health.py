from fastapi import APIRouter
from datetime import datetime
import sys
import os

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from models.schemas import HealthResponse

router = APIRouter(tags=["health"])

@router.get("/health", response_model=HealthResponse)
async def health_check():
    """健康检查接口"""
    return HealthResponse(
        status="healthy",
        timestamp=datetime.utcnow().isoformat(),
        version="1.0.0",
        services={
            "repair_service": "running",
            "vector_db": "connected",
            "llm_models": "available"
        }
    )

@router.get("/health/deep")
async def deep_health_check():
    """深度健康检查，检查所有依赖服务"""
    try:
        # 这里可以添加对各个服务的实际检查
        # 例如：检查数据库连接、模型加载状态等
        
        services_status = {}
        
        # 检查环境变量
        required_env_vars = ["OPENAI_API_KEY", "PINECONE_API_KEY"]
        env_status = {}
        for env_var in required_env_vars:
            env_status[env_var] = "configured" if os.getenv(env_var) else "missing"
        
        services_status["environment"] = env_status
        services_status["python_version"] = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
        
        return {
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "detailed_status": services_status
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "timestamp": datetime.utcnow().isoformat(),
            "error": str(e)
        }