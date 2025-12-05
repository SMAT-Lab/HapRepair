import os
from dotenv import load_dotenv
from typing import Optional

# 加载环境变量
load_dotenv()

class Settings:
    # 应用基本配置
    APP_NAME: str = "ArkTS Code Repair Service"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = os.getenv("DEBUG", "False").lower() == "true"
    
    # 服务器配置
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", "8000"))
    
    # API配置
    API_PREFIX: str = "/api/v1"
    
    # OpenAI配置
    OPENAI_API_KEY: Optional[str] = os.getenv("OPENAI_API_KEY")
    OPENAI_API_BASE: Optional[str] = os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1")
    
    # Deepseek配置
    DEEPSEEK_API_KEY: Optional[str] = os.getenv("DEEPSEEK_API_KEY")
    DEEPSEEK_API_BASE: Optional[str] = os.getenv("DEEPSEEK_API_BASE", "https://api.deepseek.com/v1")
    
    # Pinecone配置
    PINECONE_API_KEY: Optional[str] = os.getenv("PINECONE_API_KEY")
    PINECONE_INDEX_NAME: str = os.getenv("PINECONE_INDEX_NAME", "arkts-1536")
    
    # 模型配置
    DEFAULT_MODEL: str = os.getenv("DEFAULT_MODEL", "gpt-4o-mini")
    MODEL_CACHE_DIR: str = os.getenv("MODEL_CACHE_DIR", "/home/models")
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "dunzhang/stella_en_1.5B_v5")
    
    # 修复配置
    MAX_REPAIR_ROUNDS: int = int(os.getenv("MAX_REPAIR_ROUNDS", "5"))
    DEFAULT_REPAIR_MODE: str = os.getenv("DEFAULT_REPAIR_MODE", "single")
    ENABLE_RAG: bool = os.getenv("ENABLE_RAG", "True").lower() == "true"
    
    # 批处理配置
    MAX_BATCH_SIZE: int = int(os.getenv("MAX_BATCH_SIZE", "100"))
    BATCH_TIMEOUT: int = int(os.getenv("BATCH_TIMEOUT", "3600"))  # 1小时
    
    # 日志配置
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    LOG_FILE: Optional[str] = os.getenv("LOG_FILE")
    
    # CORS配置
    CORS_ORIGINS: list = os.getenv("CORS_ORIGINS", "*").split(",")
    
    # 限流配置
    RATE_LIMIT_PER_MINUTE: int = int(os.getenv("RATE_LIMIT_PER_MINUTE", "60"))
    
    def validate_required_settings(self):
        """验证必需的配置项"""
        required_settings = []
        
        if not self.OPENAI_API_KEY:
            required_settings.append("OPENAI_API_KEY")
        
        if self.ENABLE_RAG and not self.PINECONE_API_KEY:
            required_settings.append("PINECONE_API_KEY (required when RAG is enabled)")
        
        if required_settings:
            raise ValueError(f"Missing required environment variables: {', '.join(required_settings)}")
        
        return True

# 创建全局设置实例
settings = Settings()

# 在导入时验证设置
try:
    settings.validate_required_settings()
except ValueError as e:
    print(f"Configuration Error: {e}")
    print("Please check your environment variables or .env file")
    # 在开发环境中可以继续运行，生产环境中应该退出
    if not settings.DEBUG:
        exit(1)