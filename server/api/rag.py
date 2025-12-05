from fastapi import APIRouter, HTTPException
import logging
import time
import sys
import os

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from models.schemas import (
    RAGSearchRequest, RAGSearchResponse,
    CodeSimilarityRequest, CodeSimilarityResponse,
    ContextExtractionRequest, ContextExtractionResponse
)
from services.rag_service import RAGService

router = APIRouter(tags=["rag"])
logger = logging.getLogger(__name__)

# 全局服务实例
rag_service = None

def get_rag_service():
    global rag_service
    if rag_service is None:
        rag_service = RAGService()
    return rag_service

@router.post("/rag/search", response_model=RAGSearchResponse)
async def search_similar_code(request: RAGSearchRequest):
    """RAG检索 - 搜索相似的代码模式和修复建议"""
    try:
        start_time = time.time()
        service = get_rag_service()
        await service.initialize()
        
        # 执行RAG搜索
        result = await service.search_similar_patterns(
            code=request.code,
            query=request.query,
            top_k=request.top_k,
            include_metadata=request.include_metadata,
            rule=request.rule
        )
        
        search_time = time.time() - start_time
        
        return RAGSearchResponse(
            success=result["success"],
            query_embedding_created=result.get("embedding_created", False),
            matches=result.get("matches", []),
            total_matches=len(result.get("matches", [])),
            search_time=search_time,
            message=result.get("message", f"Found {len(result.get('matches', []))} similar patterns")
        )
    
    except Exception as e:
        logger.error(f"Error in RAG search: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/rag/similarity", response_model=CodeSimilarityResponse)
async def compare_code_similarity(request: CodeSimilarityRequest):
    """代码相似度比较"""
    try:
        service = get_rag_service()
        await service.initialize()
        
        # 计算代码相似度
        result = await service.compare_code_similarity(
            source_code=request.source_code,
            target_codes=request.target_codes,
            threshold=request.similarity_threshold
        )
        
        return CodeSimilarityResponse(
            success=result["success"],
            similarities=result.get("similarities", []),
            most_similar=result.get("most_similar"),
            message=result.get("message", "Similarity comparison completed")
        )
    
    except Exception as e:
        logger.error(f"Error in code similarity comparison: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/rag/status")
async def get_rag_status():
    """获取RAG服务状态"""
    try:
        service = get_rag_service()
        await service.initialize()
        status = await service.get_service_status()
        
        return {
            "success": True,
            "rag_available": status.get("available", False),
            "vector_db_connected": status.get("vector_db_connected", False),
            "embedding_model_loaded": status.get("embedding_model_loaded", False),
            "index_info": status.get("index_info", {}),
            "message": status.get("message", "RAG service status retrieved")
        }
    
    except Exception as e:
        logger.error(f"Error getting RAG status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/rag/embed")
async def create_embedding(request: dict):
    """创建代码嵌入向量"""
    try:
        code = request.get("code")
        if not code:
            raise HTTPException(status_code=400, detail="Code is required")
        
        service = get_rag_service()
        await service.initialize()
        
        # 创建嵌入向量
        result = await service.create_embedding(code)
        
        return {
            "success": result["success"],
            "embedding_created": result.get("embedding_created", False),
            "embedding_dimension": result.get("embedding_dimension", 0),
            "message": result.get("message", "Embedding created successfully")
        }
    
    except Exception as e:
        logger.error(f"Error creating embedding: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/rag/repair-suggestions")
async def get_repair_suggestions(request: dict):
    """基于RAG获取修复建议"""
    try:
        code = request.get("code")
        defects = request.get("defects", [])
        top_k = request.get("top_k", 3)
        
        if not code:
            raise HTTPException(status_code=400, detail="Code is required")
        
        service = get_rag_service()
        await service.initialize()
        
        # 获取RAG修复建议
        result = await service.get_repair_suggestions(
            code=code,
            defects=defects,
            top_k=top_k
        )
        
        return {
            "success": result["success"],
            "suggestions": result.get("suggestions", []),
            "similar_patterns": result.get("similar_patterns", []),
            "confidence_score": result.get("confidence_score", 0.0),
            "message": result.get("message", "Repair suggestions generated")
        }
    
    except Exception as e:
        logger.error(f"Error getting repair suggestions: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/rag/extract-context", response_model=ContextExtractionResponse)
async def extract_context_from_defects(request: ContextExtractionRequest):
    """基于缺陷检测结果提取上下文信息"""
    try:
        service = get_rag_service()
        await service.initialize()
        
        # 提取上下文
        result = await service.extract_context_from_defects(
            project_path=request.project_path,
            code=request.code,
            defects=request.defects,
            context_window=request.context_window or 0,  # 保留兼容性，实际使用代码结构
            include_similar_patterns=request.include_similar_patterns,
            auto_detect=request.auto_detect
        )
        
        return ContextExtractionResponse(
            success=result["success"],
            context_groups=result.get("context_groups", []),
            total_context_groups=result.get("total_context_groups", 0),
            project_defects=result.get("project_defects", []),
            total_project_defects=result.get("total_project_defects", 0),
            similar_patterns=result.get("similar_patterns", []),
            message=result.get("message", "Context extraction completed")
        )
    
    except Exception as e:
        logger.error(f"Error extracting context from defects: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/rag/generate-repair-prompts")
async def generate_repair_prompts(request: ContextExtractionRequest):
    """生成完整的缺陷修复prompt"""
    try:
        service = get_rag_service()
        await service.initialize()
        
        # 提取上下文
        context_result = await service.extract_context_from_defects(
            project_path=request.project_path,
            code=request.code,
            defects=request.defects,
            context_window=request.context_window,
            include_similar_patterns=True,
            auto_detect=request.auto_detect
        )
        
        if not context_result["success"]:
            return {
                "success": False,
                "repair_prompts": [],
                "message": "Failed to extract context for repair prompts"
            }
        
        # 生成修复prompt
        prompt_result = await service.generate_repair_prompts(
            context_groups=context_result.get("context_groups", [])
        )
        
        return {
            **prompt_result,
            "context_groups": context_result.get("context_groups", []),
            "project_defects": context_result.get("project_defects", [])
        }
    
    except Exception as e:
        logger.error(f"Error generating repair prompts: {e}")
        raise HTTPException(status_code=500, detail=str(e))