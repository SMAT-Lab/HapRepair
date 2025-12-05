from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from enum import Enum

class DefectType(str, Enum):
    PERFORMANCE = "performance"
    SECURITY = "security"
    FUNCTIONALITY = "functionality"
    STYLE = "style"

class RepairMode(str, Enum):
    SINGLE = "single"
    MULTI_ROUND = "multi_round"

class CodeRepairRequest(BaseModel):
    code: str = Field(..., description="The ArkTS code to be repaired")
    defect_type: Optional[DefectType] = Field(None, description="Type of defect to detect and repair")
    repair_mode: RepairMode = Field(RepairMode.SINGLE, description="Repair mode: single or multi-round")
    max_rounds: int = Field(3, ge=1, le=10, description="Maximum repair rounds for multi-round mode")
    use_rag: bool = Field(True, description="Whether to use RAG for repair")
    model_name: str = Field("gpt-4o-mini", description="LLM model to use for repair")

class CodeRepairResponse(BaseModel):
    success: bool
    original_code: str
    repaired_code: Optional[str] = None
    defects_found: List[Dict[str, Any]] = []  
    repair_rounds: int = 0
    repair_history: List[Dict[str, Any]] = []
    message: str
    execution_time: float

class DefectDetectionRequest(BaseModel):
    code: Optional[str] = Field(None, description="The ArkTS code to analyze (for single file)")
    project_path: Optional[str] = Field(None, description="Path to the harmony project directory")
    detection_rules: Optional[List[str]] = Field(None, description="Specific rules to check")
    
    def __init__(self, **data):
        super().__init__(**data)
        # 验证至少提供code或project_path之一
        if not self.code and not self.project_path:
            raise ValueError("Either 'code' or 'project_path' must be provided")

class DefectDetectionResponse(BaseModel):
    success: bool
    defects: List[Dict[str, Any]] = []
    total_defects: int
    code_quality_score: float = Field(0.0, ge=0.0, le=100.0)
    message: str

class FunctionalityCheckRequest(BaseModel):
    original_code: str = Field(..., description="Original code before repair")
    repaired_code: str = Field(..., description="Code after repair")

class FunctionalityCheckResponse(BaseModel):
    success: bool
    functionality_preserved: bool
    differences: List[Dict[str, Any]] = []
    message: str

class HealthResponse(BaseModel):
    status: str
    timestamp: str
    version: str
    services: Dict[str, str]

class RAGSearchRequest(BaseModel):
    code: str = Field(..., description="The ArkTS code to search for similar patterns")
    query: Optional[str] = Field(None, description="Optional text query for semantic search")
    top_k: int = Field(5, ge=1, le=20, description="Number of similar examples to retrieve")
    include_metadata: bool = Field(True, description="Whether to include metadata in results")
    rule: Optional[str] = Field(None, description="Rule to filter results")

class RAGSearchResponse(BaseModel):
    success: bool
    query_embedding_created: bool
    matches: List[Dict[str, Any]] = []
    total_matches: int
    search_time: float
    message: str

class CodeSimilarityRequest(BaseModel):
    source_code: str = Field(..., description="Source code to compare")
    target_codes: List[str] = Field(..., description="List of target codes to compare against")
    similarity_threshold: float = Field(0.7, ge=0.0, le=1.0, description="Similarity threshold")

class CodeSimilarityResponse(BaseModel):
    success: bool
    similarities: List[Dict[str, Any]] = []
    most_similar: Optional[Dict[str, Any]] = None
    message: str

class ContextExtractionRequest(BaseModel):
    project_path: Optional[str] = Field(None, description="Path to the harmony project directory")
    code: Optional[str] = Field(None, description="The ArkTS code to analyze (for single file)")
    defects: Optional[List[Dict[str, Any]]] = Field(None, description="List of detected defects with their details")
    context_window: int = Field(5, ge=1, le=20, description="Number of lines to include as context around defects")
    include_similar_patterns: bool = Field(True, description="Whether to include similar code patterns from RAG")
    auto_detect: bool = Field(False, description="Whether to auto-detect defects from project")
    
    def __init__(self, **data):
        super().__init__(**data)
        # 验证至少提供project_path或code/defects之一
        if not self.project_path and (not self.code or not self.defects):
            raise ValueError("Either 'project_path' or both 'code' and 'defects' must be provided")
    
class ContextExtractionResponse(BaseModel):
    success: bool
    context_groups: List[Dict[str, Any]] = []
    total_context_groups: int
    project_defects: List[Dict[str, Any]] = []
    total_project_defects: int
    similar_patterns: List[Dict[str, Any]] = []
    message: str