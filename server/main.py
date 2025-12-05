from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import logging
import sys
import os

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api.repair import router as repair_router
from api.health import router as health_router
from api.rag import router as rag_router
from services.repair_service import RepairService
from services.rag_service import RAGService

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 全局服务实例
repair_service = None
rag_service = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    # 启动时初始化服务
    global repair_service, rag_service
    try:
        logger.info("Initializing services...")
        
        # 初始化修复服务
        repair_service = RepairService()
        await repair_service.initialize()
        logger.info("Repair service initialized successfully")
        
        # 初始化RAG服务
        try:
            rag_service = RAGService()
            await rag_service.initialize()
            logger.info("RAG service initialized successfully")
        except Exception as e:
            logger.warning(f"RAG service initialization failed: {e}")
            logger.info("Application will continue without RAG functionality")
        
        yield
    except Exception as e:
        logger.error(f"Failed to initialize services: {e}")
        raise
    finally:
        # 清理资源
        if repair_service:
            await repair_service.cleanup()
        if rag_service:
            await rag_service.cleanup()
        logger.info("Application shutdown complete")

app = FastAPI(
    title="ArkTS Code Repair Service",
    description="Automated ArkTS code defect repair system based on large language models",
    version="1.0.0",
    lifespan=lifespan
)

# 配置CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 生产环境中应该限制具体域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 注册路由
app.include_router(health_router, prefix="/api/v1")
app.include_router(repair_router, prefix="/api/v1")
app.include_router(rag_router, prefix="/api/v1")

@app.get("/")
async def root():
    return {
        "message": "ArkTS Code Repair Service",
        "version": "1.0.0",
        "docs": "/docs"
    }

if __name__ == "__main__":
    import uvicorn
    import os
    
    # 获取配置
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    debug = os.getenv("DEBUG", "False").lower() == "true"
    
    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        reload=debug,  # 只在调试模式下启用自动重载
        log_level="info"
    )