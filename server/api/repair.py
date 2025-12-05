from fastapi import APIRouter, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
import logging
import time
import sys
import os

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from models.schemas import (
    CodeRepairRequest, CodeRepairResponse,
    DefectDetectionRequest, DefectDetectionResponse,
    FunctionalityCheckRequest, FunctionalityCheckResponse
)
from services.repair_service import RepairService

router = APIRouter(tags=["repair"])
logger = logging.getLogger(__name__)

# 全局服务实例（在生产环境中应该使用依赖注入）
repair_service = None

def get_repair_service():
    global repair_service
    if repair_service is None:
        repair_service = RepairService()
    return repair_service

@router.post("/detect", response_model=DefectDetectionResponse)
async def detect_defects(request: DefectDetectionRequest):
    """检测代码缺陷 - 支持单文件代码或整个harmony项目"""
    try:
        start_time = time.time()
        service = get_repair_service()
        
        # 调用缺陷检测服务
        defects = await service.detect_defects(
            code=request.code,
            detection_rules=request.detection_rules,
            project_path=request.project_path
        )
        
        # 计算代码质量评分（简单实现）
        quality_score = max(0, 100 - len(defects) * 10)
        
        execution_time = time.time() - start_time
        
        detection_target = "project" if request.project_path else "single file"
        
        return DefectDetectionResponse(
            success=True,
            defects=defects,
            total_defects=len(defects),
            code_quality_score=quality_score,
            message=f"Found {len(defects)} defects in {detection_target} in {execution_time:.2f}s"
        )
    
    except Exception as e:
        logger.error(f"Error in defect detection: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/repair", response_model=CodeRepairResponse)
async def repair_code(request: CodeRepairRequest):
    """修复代码缺陷"""
    try:
        start_time = time.time()
        service = get_repair_service()
        
        # 调用代码修复服务
        result = await service.repair_code(
            code=request.code,
            defect_type=request.defect_type,
            repair_mode=request.repair_mode,
            max_rounds=request.max_rounds,
            use_rag=request.use_rag,
            model_name=request.model_name
        )
        
        execution_time = time.time() - start_time
        
        return CodeRepairResponse(
            success=result["success"],
            original_code=request.code,
            repaired_code=result.get("repaired_code"),
            defects_found=result.get("defects_found", []),
            repair_rounds=result.get("repair_rounds", 0),
            repair_history=result.get("repair_history", []),
            message=result.get("message", "Repair completed"),
            execution_time=execution_time
        )
    
    except Exception as e:
        logger.error(f"Error in code repair: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/verify", response_model=FunctionalityCheckResponse)
async def verify_functionality(request: FunctionalityCheckRequest):
    """验证修复后代码的功能完整性"""
    try:
        service = get_repair_service()
        
        # 调用功能验证服务
        result = await service.verify_functionality(
            original_code=request.original_code,
            repaired_code=request.repaired_code
        )
        
        return FunctionalityCheckResponse(
            success=result["success"],
            functionality_preserved=result.get("functionality_preserved", False),
            differences=result.get("differences", []),
            message=result.get("message", "Verification completed")
        )
    
    except Exception as e:
        logger.error(f"Error in functionality verification: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/repair/batch")
async def repair_batch(
    codes: list[str],
    background_tasks: BackgroundTasks,
    defect_type: str = None,
    model_name: str = "gpt-4o-mini"
):
    """批量修复代码（异步处理）"""
    try:
        service = get_repair_service()
        
        # 创建批处理任务
        task_id = f"batch_{int(time.time())}"
        
        # 添加后台任务
        background_tasks.add_task(
            service.batch_repair,
            codes=codes,
            task_id=task_id,
            defect_type=defect_type,
            model_name=model_name
        )
        
        return JSONResponse(
            content={
                "task_id": task_id,
                "status": "accepted",
                "message": f"Batch repair task created with {len(codes)} files"
            }
        )
    
    except Exception as e:
        logger.error(f"Error in batch repair: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/repair/batch/{task_id}")
async def get_batch_status(task_id: str):
    """获取批处理任务状态"""
    try:
        service = get_repair_service()
        status = await service.get_batch_status(task_id)
        
        if status is None:
            raise HTTPException(status_code=404, detail="Task not found")
        
        return status
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting batch status: {e}")
        raise HTTPException(status_code=500, detail=str(e))