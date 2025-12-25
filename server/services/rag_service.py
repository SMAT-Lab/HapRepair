import logging
import sys
import os
import json
from typing import Optional, List, Dict, Any
import numpy as np

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# 可选导入
try:
    from transformers import AutoTokenizer, AutoModel
    import torch
    HAS_TRANSFORMERS = True
except ImportError:
    AutoTokenizer = None
    AutoModel = None
    torch = None
    HAS_TRANSFORMERS = False

try:
    from pinecone import Pinecone
    HAS_PINECONE = True
except ImportError:
    Pinecone = None
    HAS_PINECONE = False

# 导入项目模块
try:
    from get_prompt import get_rag_prompt
except ImportError as e:
    logging.warning(f"Could not import project modules: {e}")
    get_rag_prompt = None

logger = logging.getLogger(__name__)

class RAGService:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.pinecone_client = None
        self.index = None
        self.model_name = "dunzhang/stella_en_1.5B_v5"
        self.index_name = "arkts-1536"
        self.embedding_dimension = 1536
        
    async def initialize(self):
        """初始化RAG服务"""
        try:
            logger.info("Initializing RAG service...")
            
            # 初始化嵌入模型
            if HAS_TRANSFORMERS and self.model is None:
                await self._load_embedding_model()
            else:
                if not HAS_TRANSFORMERS:
                    logger.warning("Transformers not available, embedding functionality disabled")
            
            # 初始化Pinecone
            if HAS_PINECONE and self.index is None:
                await self._connect_pinecone()
            else:
                if not HAS_PINECONE:
                    logger.warning("Pinecone not available, vector search functionality disabled")
            
            logger.info("RAG service initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize RAG service: {e}")
            raise
    
    async def _load_embedding_model(self):
        """加载嵌入模型"""
        try:
            if self.model is not None and self.tokenizer is not None:
                return
            cache_dir = os.getenv("MODEL_CACHE_DIR", "/home/models")
            
            logger.info(f"Loading embedding model: {self.model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name, 
                cache_dir=cache_dir
            )
            self.model = AutoModel.from_pretrained(
                self.model_name, 
                cache_dir=cache_dir
            )
            
            # 设置为评估模式
            self.model.eval()

            if torch is None:
                raise RuntimeError("Torch is required for embedding model usage")

            device_raw = os.getenv("RAG_DEVICE", "").strip().lower()
            if not device_raw:
                device_raw = "cuda" if torch.cuda.is_available() else "cpu"
            if device_raw in ("gpu",):
                device_raw = "cuda"

            try:
                device = torch.device(device_raw)
            except Exception:
                device = torch.device("cpu")

            if device.type == "cuda" and not torch.cuda.is_available():
                device = torch.device("cpu")

            dtype_raw = os.getenv("RAG_DTYPE", "").strip().lower()
            if device.type == "cuda":
                if dtype_raw in ("bf16", "bfloat16"):
                    self.model = self.model.to(dtype=torch.bfloat16)
                elif dtype_raw in ("fp16", "float16", "half"):
                    self.model = self.model.to(dtype=torch.float16)

            self.model = self.model.to(device)
            logger.info(f"Embedding model device: {device}, dtype={dtype_raw or 'fp32'}")
            
            logger.info("Embedding model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            raise
    
    async def _connect_pinecone(self):
        """连接Pinecone"""
        try:
            if self.index is not None:
                return
            api_key = os.getenv("PINECONE_API_KEY")
            if not api_key:
                logger.warning("PINECONE_API_KEY not set, vector search disabled")
                return
            
            logger.info("Connecting to Pinecone...")
            self.pinecone_client = Pinecone(api_key=api_key)
            self.index = self.pinecone_client.Index(self.index_name)
            
            # 测试连接
            stats = self.index.describe_index_stats()
            logger.info(f"Connected to Pinecone index '{self.index_name}' with {stats.total_vector_count} vectors")
            
        except Exception as e:
            logger.error(f"Failed to connect to Pinecone: {e}")
            self.pinecone_client = None
            self.index = None
    
    async def create_embedding(self, text: str) -> Dict[str, Any]:
        """创建文本嵌入向量"""
        try:
            if not HAS_TRANSFORMERS or not self.model:
                return {
                    "success": False,
                    "embedding_created": False,
                    "message": "Embedding model not available"
                }
            
            if torch is None:
                return {
                    "success": False,
                    "embedding_created": False,
                    "message": "Torch not available"
                }

            # 分词和编码
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=512,
            )
            device = next(self.model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # 生成嵌入
            with torch.inference_mode():
                outputs = self.model(**inputs)
                # 使用平均池化
                embeddings = outputs.last_hidden_state.mean(dim=1).squeeze()
                
            # 转换为numpy数组
            embedding_vector = embeddings.detach().cpu().numpy()
            
            return {
                "success": True,
                "embedding_created": True,
                "embedding": embedding_vector.tolist(),
                "embedding_dimension": len(embedding_vector),
                "message": "Embedding created successfully"
            }
            
        except Exception as e:
            logger.error(f"Error creating embedding: {e}")
            return {
                "success": False,
                "embedding_created": False,
                "message": f"Failed to create embedding: {str(e)}"
            }
    
    async def search_similar_patterns(
        self, 
        code: str, 
        query: Optional[str] = None,
        top_k: int = 5,
        include_metadata: bool = True,
        rule: Optional[str] = None
    ) -> Dict[str, Any]:
        """搜索相似的代码模式"""
        try:
            if not self.index:
                return {
                    "success": False,
                    "matches": [],
                    "message": "Vector search not available - Pinecone not connected"
                }
            
            # 创建查询向量
            search_text = f"{code}\n{query}" if query else code
            embedding_result = await self.create_embedding(search_text)
            
            if not embedding_result["success"]:
                return {
                    "success": False,
                    "embedding_created": False,
                    "matches": [],
                    "message": "Failed to create query embedding"
                }
            
            # 执行向量搜索
            query_vector = embedding_result["embedding"]
            search_results = self.index.query(
                namespace="arkts",
                vector=query_vector,
                top_k=top_k,
                include_metadata=include_metadata,
                include_values=False,
                filter={"rule": rule} if rule else None
            )

            print(search_results)
            
            # 格式化结果
            matches = []
            for match in search_results.matches:
                match_data = {
                    "id": match.id,
                    "score": float(match.score),
                    "metadata": match.metadata if include_metadata else None
                }
                
                # 提取有用的元数据信息
                if include_metadata and match.metadata:
                    match_data.update({
                        "rule": match.metadata.get("rule", ""),
                        "defect_type": match.metadata.get("defect_type", ""),
                        "description": match.metadata.get("description", ""),
                        "original_code": match.metadata.get("original_code", ""),
                        "fixed_code": match.metadata.get("fixed_code", "")
                    })
                
                matches.append(match_data)
            
            return {
                "success": True,
                "embedding_created": True,
                "matches": matches,
                "message": f"Found {len(matches)} similar patterns"
            }
            
        except Exception as e:
            logger.error(f"Error in similarity search: {e}")
            return {
                "success": False,
                "embedding_created": False,
                "matches": [],
                "message": f"Search failed: {str(e)}"
            }
    
    async def compare_code_similarity(
        self, 
        source_code: str, 
        target_codes: List[str],
        threshold: float = 0.7
    ) -> Dict[str, Any]:
        """比较代码相似度"""
        try:
            # 创建源代码嵌入
            source_embedding_result = await self.create_embedding(source_code)
            if not source_embedding_result["success"]:
                return {
                    "success": False,
                    "similarities": [],
                    "message": "Failed to create source code embedding"
                }
            
            source_vector = np.array(source_embedding_result["embedding"])
            similarities = []
            
            # 计算与每个目标代码的相似度
            for i, target_code in enumerate(target_codes):
                target_embedding_result = await self.create_embedding(target_code)
                
                if target_embedding_result["success"]:
                    target_vector = np.array(target_embedding_result["embedding"])
                    
                    # 计算余弦相似度
                    similarity = np.dot(source_vector, target_vector) / (
                        np.linalg.norm(source_vector) * np.linalg.norm(target_vector)
                    )
                    
                    similarities.append({
                        "index": i,
                        "target_code": target_code,
                        "similarity": float(similarity),
                        "above_threshold": similarity >= threshold
                    })
            
            # 找到最相似的代码
            most_similar = max(similarities, key=lambda x: x["similarity"]) if similarities else None
            
            return {
                "success": True,
                "similarities": similarities,
                "most_similar": most_similar,
                "message": f"Computed similarity for {len(similarities)} code pairs"
            }
            
        except Exception as e:
            logger.error(f"Error in code similarity comparison: {e}")
            return {
                "success": False,
                "similarities": [],
                "message": f"Similarity comparison failed: {str(e)}"
            }
    
    async def get_repair_suggestions(
        self, 
        code: str, 
        defects: List[Dict[str, Any]],
        top_k: int = 3
    ) -> Dict[str, Any]:
        """基于RAG获取修复建议"""
        try:
            # 构建搜索查询
            defect_rules = [defect.get("rule", "") for defect in defects]
            search_query = f"Code with defects: {', '.join(defect_rules)}"
            
            # 搜索相似模式
            search_result = await self.search_similar_patterns(
                code=code,
                query=search_query,
                top_k=top_k,
                include_metadata=True
            )
            
            if not search_result["success"]:
                return {
                    "success": False,
                    "suggestions": [],
                    "message": "Failed to search for similar patterns"
                }
            
            # 生成修复建议
            suggestions = []
            similar_patterns = search_result["matches"]
            
            for pattern in similar_patterns:
                suggestion = {
                    "confidence": pattern["score"],
                    "rule": pattern.get("rule", ""),
                    "description": pattern.get("description", ""),
                    "original_code": pattern.get("original_code", ""),
                    "fixed_code": pattern.get("fixed_code", ""),
                    "similarity_score": pattern["score"]
                }
                suggestions.append(suggestion)
            
            # 计算总体置信度
            avg_confidence = np.mean([s["confidence"] for s in suggestions]) if suggestions else 0.0
            
            return {
                "success": True,
                "suggestions": suggestions,
                "similar_patterns": similar_patterns,
                "confidence_score": float(avg_confidence),
                "message": f"Generated {len(suggestions)} repair suggestions"
            }
            
        except Exception as e:
            logger.error(f"Error getting repair suggestions: {e}")
            return {
                "success": False,
                "suggestions": [],
                "message": f"Failed to get repair suggestions: {str(e)}"
            }
    
    async def get_service_status(self) -> Dict[str, Any]:
        """获取RAG服务状态"""
        try:
            status = {
                "available": True,
                "embedding_model_loaded": self.model is not None,
                "vector_db_connected": self.index is not None,
                "transformers_available": HAS_TRANSFORMERS,
                "pinecone_available": HAS_PINECONE
            }
            
            # 获取索引信息
            if self.index:
                try:
                    index_stats = self.index.describe_index_stats()
                    status["index_info"] = {
                        "total_vectors": index_stats.total_vector_count,
                        "dimension": self.embedding_dimension,
                        "index_name": self.index_name
                    }
                except Exception as e:
                    logger.warning(f"Could not get index stats: {e}")
                    status["index_info"] = {"error": str(e)}
            
            status["message"] = "RAG service status retrieved successfully"
            return status
            
        except Exception as e:
            logger.error(f"Error getting service status: {e}")
            return {
                "available": False,
                "message": f"Failed to get service status: {str(e)}"
            }
    
    async def extract_context_from_defects(
        self, 
        project_path: Optional[str] = None,
        code: Optional[str] = None,
        defects: Optional[List[Dict[str, Any]]] = None,
        context_window: int = 5,
        include_similar_patterns: bool = True,
        auto_detect: bool = False
    ) -> Dict[str, Any]:
        """基于缺陷结果提取上下文，正确处理缺陷合并"""
        try:
            from services.repair_service import RepairService
            
            # 获取所有缺陷
            if project_path:
                if auto_detect or defects is None:
                    repair_service = RepairService()
                    all_defects = await repair_service.detect_defects(project_path=project_path)
                else:
                    all_defects = defects or []
            elif code and defects:
                # 单文件模式，所有缺陷对应同一个文件
                all_defects = [
                    dict(defect, file="current_file", content=code) 
                    for defect in (defects or [])
                ]
            else:
                all_defects = []
            
            if not all_defects:
                return {
                    "success": True,
                    "context_groups": [],
                    "total_context_groups": 0,
                    "project_defects": [],
                    "total_project_defects": 0,
                    "similar_patterns": [],
                    "message": "No defects found"
                }
            
            # 按文件分组缺陷
            file_defects_map = {}
            for defect in all_defects:
                file_path = defect.get('file', 'unknown')
                if file_path not in file_defects_map:
                    file_defects_map[file_path] = []
                file_defects_map[file_path].append(defect)
            
            # 处理每个文件
            all_context_groups = []
            all_similar_patterns = []
            
            for file_path, file_defects in file_defects_map.items():
                # 获取文件内容
                if file_path == "current_file" and code:
                    file_content = code
                else:
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            file_content = f.read()
                    except Exception as e:
                        logger.warning(f"Cannot read file {file_path}: {e}")
                        continue
                
                # 处理该文件的缺陷
                context_groups = await self._process_defect_contexts(
                    file_content, file_defects, context_window, include_similar_patterns, file_path
                )
                all_context_groups.extend(context_groups)
            
            # 去重相似模式
            seen_patterns = set()
            unique_patterns = []
            for pattern in all_similar_patterns:
                pattern_key = f"{pattern.get('id', '')}_{pattern.get('score', 0)}"
                if pattern_key not in seen_patterns:
                    seen_patterns.add(pattern_key)
                    unique_patterns.append(pattern)
            
            return {
                "success": True,
                "context_groups": all_context_groups,
                "total_context_groups": len(all_context_groups),
                "project_defects": all_defects,
                "total_project_defects": len(all_defects),
                "similar_patterns": unique_patterns[:10],
                "message": f"Merged {len(all_context_groups)} context groups for {len(all_defects)} defects"
            }
            
        except Exception as e:
            logger.error(f"Error extracting context from defects: {e}")
            return {
                "success": False,
                "context_groups": [],
                "total_context_groups": 0,
                "project_defects": [],
                "total_project_defects": 0,
                "similar_patterns": [],
                "message": f"Failed to extract context: {str(e)}"
            }
    
    async def _process_defect_contexts(self, code: str, defects: List[Dict[str, Any]], 
                                     context_window: int, include_similar_patterns: bool, 
                                     file_path: str = None) -> List[Dict[str, Any]]:
        """使用代码结构分析提取上下文，基于已有缺陷检测结果"""
        
        # 添加当前目录到路径，确保可以导入CodeContextExtractor
        import sys
        import os
        
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(os.path.dirname(current_dir))
        if project_root not in sys.path:
            sys.path.insert(0, project_root)
        
        try:
            from code_repair import CodeContextExtractor
        except ImportError as e:
            logger.warning(f"Cannot import CodeContextExtractor: {e}")
            return await self._simple_context_extraction(code, defects, file_path)
        
        try:
            # 使用CodeContextExtractor进行代码结构分析
            code_lines = [''] + code.split('\n')  # 1-based indexing
            extractor = CodeContextExtractor()
            
            # 按缺陷分组处理
            if not defects:
                return []
            
            # 创建缺陷到代码块的映射
            defect_blocks = []
            for defect in defects:
                line_num = defect.get('line', 1)
                rule_name = defect.get('rule', 'unknown')
                
                # 提取代码块
                start_idx, block_end, surrounding_context = extractor._extract_blocks(
                    code_lines, line_num
                )
                
                if surrounding_context is None:
                    # 如果无法提取代码块，使用缺陷行本身
                    start_idx = line_num
                    block_end = line_num
                    surrounding_context = code_lines[line_num] if line_num < len(code_lines) else ""
                
                # 默认需要更多上下文进行代码结构分析
                    # 提取ArkTS上下文
                    context = extractor.extract_arkts_context(code_lines, line_num)
                    
                    # 收集所有相关代码块
                    code_blocks = []
                    block_ranges = [(start_idx, block_end)]
                    
                    # 添加定义上下文
                    for def_start, content in context.get('definition', []):
                        block_ranges.append((def_start, def_start))
                        code_blocks.append((def_start, def_start, content))
                    
                    # 添加使用上下文
                    for use_start, content in context.get('usage', []):
                        block_ranges.append((use_start, use_start))
                        code_blocks.append((use_start, use_start, content))
                    
                    # 去重和排序
                    unique_ranges = list(set(tuple(r) for r in block_ranges))
                    unique_ranges.sort(key=lambda x: x[0])
                    
                    # 合并重叠的范围
                    merged_ranges = self._merge_ranges(unique_ranges)
                    
                    # 构建最终上下文
                    final_context_parts = []
                    for rng_start, rng_end in merged_ranges:
                        context_part = '\n'.join(code_lines[rng_start:rng_end+1]).rstrip()
                        if context_part:
                            final_context_parts.append(context_part)
                    
                    final_context = '\n\n'.join(final_context_parts)
                    
                    defect_blocks.append({
                        'defect': defect,
                        'block_ranges': merged_ranges,
                        'surrounding_context': final_context
                    })
                else:
                    # 简单上下文
                    defect_blocks.append({
                        'defect': defect,
                        'block_ranges': [(start_idx, block_end)],
                        'surrounding_context': surrounding_context
                    })
            
            # 合并重叠的缺陷上下文
            if not defect_blocks:
                return []
            
            # 使用并查集合并重叠的上下文
            from get_surrounding_context import UnionFind
            
            uf = UnionFind(len(defect_blocks))
            
            # 检查重叠
            for i in range(len(defect_blocks)):
                for j in range(i + 1, len(defect_blocks)):
                    ranges1 = defect_blocks[i]['block_ranges']
                    ranges2 = defect_blocks[j]['block_ranges']
                    
                    if self._ranges_overlap(ranges1, ranges2):
                        uf.union(i, j)
            
            # 分组处理
            groups = {}
            for i in range(len(defect_blocks)):
                parent = uf.find(i)
                if parent not in groups:
                    groups[parent] = []
                groups[parent].append(defect_blocks[i])
            
            # 构建最终的上下文组
            context_groups = []
            for group_blocks in groups.values():
                merged_defects = [block['defect'] for block in group_blocks]
                
                # 合并所有范围
                all_ranges = []
                for block in group_blocks:
                    all_ranges.extend(block['block_ranges'])
                
                # 去重和合并
                unique_ranges = list(set(tuple(r) for r in all_ranges))
                unique_ranges.sort(key=lambda x: x[0])
                merged_ranges = self._merge_ranges(unique_ranges)
                
                # 构建最终上下文
                final_context_parts = []
                for rng_start, rng_end in merged_ranges:
                    context_part = '\n'.join(code_lines[rng_start:rng_end+1]).rstrip()
                    if context_part:
                        final_context_parts.append(context_part)
                
                final_context = '\n\n'.join(final_context_parts)
                
                # 计算范围信息
                if merged_ranges:
                    start_line = merged_ranges[0][0]
                    end_line = merged_ranges[-1][1]
                else:
                    start_line = 1
                    end_line = len(code_lines) - 1
                
                context_group = {
                    "defects": merged_defects,
                    "context_code": final_context,
                    "context_range": {
                        "start_line": start_line,
                        "end_line": end_line,
                        "center_line": (start_line + end_line) // 2,
                        "defect_lines": [d.get('line', 0) for d in merged_defects]
                    },
                    "related_symbols": self._extract_related_symbols(code_lines[1:], start_line - 1),
                    "file_context": self._get_file_context(code_lines[1:]),
                    "file_path": file_path,
                    "defect_count": len(merged_defects)
                }
                context_groups.append(context_group)
            
            return context_groups
            
        except Exception as e:
            logger.error(f"Error in code structure analysis: {e}")
            return await self._simple_context_extraction(code, defects, file_path)

    async def _simple_context_extraction(self, code: str, defects: List[Dict[str, Any]], 
                                       file_path: str = None) -> List[Dict[str, Any]]:
        """简单的上下文提取作为回退方案"""
        code_lines = code.split('\n')
        
        context_groups = []
        for defect in defects:
            line_num = defect.get('line', 1)
            start_line = max(1, line_num - 5)
            end_line = min(len(code_lines), line_num + 5)
            
            context_code = '\n'.join(code_lines[start_line-1:end_line])
            
            context_group = {
                "defects": [defect],
                "context_code": context_code,
                "context_range": {
                    "start_line": start_line,
                    "end_line": end_line,
                    "center_line": line_num,
                    "defect_lines": [line_num]
                },
                "related_symbols": self._extract_related_symbols(code_lines, line_num - 1),
                "file_context": self._get_file_context(code_lines),
                "file_path": file_path,
                "defect_count": 1
            }
            context_groups.append(context_group)
        
        return context_groups
    
    async def generate_repair_prompts(self, context_groups: List[Dict[str, Any]]) -> Dict[str, Any]:
        """生成完整的缺陷修复prompt"""
        try:
            repair_prompts = []
            
            for group in context_groups:
                defects = group['defects']
                context_code = group['context_code']
                file_path = group['file_path']
                
                # 收集所有规则
                rules = list(set([d.get('rule', '') for d in defects if d.get('rule')]))

                print("rules:", rules)
                
                # 为每个规则获取相似代码
                similar_examples = []
                for rule in rules:
                    if self.index and rule:
                        pattern_result = await self.search_similar_patterns(
                            code=context_code,
                            query=f"rule: {rule} defect repair",
                            top_k=3,
                            rule="@"+rule
                        )
                        if pattern_result["success"]:
                            similar_examples.extend(pattern_result["matches"])
                
                # 构建修复prompt
                prompt = self._build_repair_prompt(
                    defects=defects,
                    context_code=context_code,
                    similar_examples=similar_examples,
                    file_path=file_path
                )
                
                repair_prompts.append({
                    'defect_group': group,
                    'rules': rules,
                    'similar_examples': similar_examples,
                    'repair_prompt': prompt,
                    'prompt_length': len(prompt)
                })
            
            return {
                'success': True,
                'repair_prompts': repair_prompts,
                'total_prompts': len(repair_prompts),
                'message': f'Generated {len(repair_prompts)} repair prompts'
            }
            
        except Exception as e:
            logger.error(f"Error generating repair prompts: {e}")
            return {
                'success': False,
                'repair_prompts': [],
                'total_prompts': 0,
                'message': f'Failed to generate repair prompts: {str(e)}'
            }
    
    def _build_repair_prompt(self, defects: List[Dict[str, Any]], context_code: str, 
                           similar_examples: List[Dict[str, Any]], file_path: str) -> str:
        """构建单个缺陷组的修复prompt"""
        
        # 收集缺陷信息
        defect_summary = []
        for defect in defects:
            defect_summary.append(
                f"- 规则: {defect.get('rule', 'unknown')}"
                f"\n  行号: {defect.get('line', 'unknown')}"
                f"\n  消息: {defect.get('message', 'No message')}"
                f"\n  文件: {defect.get('file', file_path)}"
            )
        
        defect_summary_str = '\n'.join(defect_summary)
        
        # 构建相似示例
        similar_examples_str = ""
        if similar_examples:
            similar_examples_str = "\n\n## 相似修复示例:\n"
            for i, example in enumerate(similar_examples[:3], 1):
                metadata = example.get('metadata', {})
                original = metadata.get('original_code', '')
                description = metadata.get('description', '')
                problem_code = metadata.get('problem_code', '')
                problem_explain = metadata.get('problem_explain', '')
                problem_fix = metadata.get('problem_fix', '')
                difflib_info = metadata.get('difflib', '')
                gpt_diff = metadata.get('gpt_diff', '')
                
                similar_examples_str += f"\n### 示例 {i}:\n"
                similar_examples_str += f"**问题代码:**\n```typescript\n{problem_code or original}\n```\n"
                similar_examples_str += f"**修复后代码:**\n```typescript\n{problem_fix}\n```\n"
                similar_examples_str += f"**问题说明:** {problem_explain or description}\n"
                if difflib_info:
                    similar_examples_str += f"**代码差异分析:**\n```diff\n{difflib_info}\n```\n"
                if gpt_diff:
                    similar_examples_str += f"**修复差异逻辑:**\n```diff\n{gpt_diff}\n```\n"
        
        # 构建完整prompt
        prompt = f"""# ArkTS 代码缺陷修复任务

## 缺陷信息:
{defect_summary_str}

## 当前上下文代码:
```typescript
{context_code}
```

{similar_examples_str}

## 修复要求:
1. 请根据上述缺陷信息和相似示例，修复当前代码中的问题
2. 保持代码功能不变，仅修复指定的缺陷
3. 遵循ArkTS最佳实践
4. 返回完整的修复后代码
5. 在修复位置添加简短注释说明修复内容

## 修复后代码:
```typescript
"""
        
        return prompt
    
    async def cleanup(self):
        """清理资源"""
        logger.info("Cleaning up RAG service resources")
        # 这里可以添加资源清理逻辑
    
    def _extract_related_symbols(self, code_lines: List[str], line_num: int) -> Dict[str, List[str]]:
        """提取相关变量和函数"""
        try:
            variables = []
            functions = []
            classes = []
            
            # 扫描缺陷行附近的代码
            start_scan = max(0, line_num - 10)
            end_scan = min(len(code_lines), line_num + 10)
            
            for i in range(start_scan, end_scan):
                line = code_lines[i].strip()
                
                # 提取变量定义
                if '=' in line and not line.startswith('//') and not line.startswith('/*'):
                    var_name = line.split('=')[0].strip()
                    if var_name and not var_name.startswith('//') and not var_name.startswith('/*'):
                        variables.append(var_name)
                
                # 提取函数定义
                if line.startswith('function ') or line.startswith('private ') or line.startswith('public '):
                    func_match = line.split('(')[0].split()[-1]
                    if func_match:
                        functions.append(func_match)
                
                # 提取类/组件定义
                if line.startswith('@Component') or line.startswith('class '):
                    class_match = line.split()[-1].split('{')[0].strip()
                    if class_match:
                        classes.append(class_match)
            
            return {
                "variables": list(set(variables)),
                "functions": list(set(functions)),
                "classes": list(set(classes))
            }
            
        except Exception as e:
            logger.warning(f"Error extracting related symbols: {e}")
            return {"variables": [], "functions": [], "classes": []}
    
    def _get_file_context(self, code_lines: List[str]) -> Dict[str, Any]:
        """获取文件级上下文"""
        try:
            total_lines = len(code_lines)
            imports = []
            exports = []
            
            for line in code_lines:
                stripped = line.strip()
                if stripped.startswith('import ') or stripped.startswith('from '):
                    imports.append(stripped)
                elif stripped.startswith('export '):
                    exports.append(stripped)
            
            return {
                "total_lines": total_lines,
                "imports": imports,
                "exports": exports,
                "has_main_component": any('@Entry' in line for line in code_lines)
            }
            
        except Exception as e:
            logger.warning(f"Error getting file context: {e}")
            return {"total_lines": len(code_lines), "imports": [], "exports": [], "has_main_component": False}

    def _ranges_overlap(self, ranges1: List[List[int]], ranges2: List[List[int]]) -> bool:
        """检查两个范围列表是否有重叠"""
        for start1, end1 in ranges1:
            for start2, end2 in ranges2:
                if not (end2 + 1 < start1 or start2 - 1 > end1):
                    return True
        return False

    def _merge_ranges(self, ranges: List[List[int]]) -> List[List[int]]:
        """合并重叠的范围"""
        if not ranges:
            return []
        
        # 去重并排序
        unique_ranges = [list(x) for x in set(tuple(x) for x in ranges)]
        sorted_ranges = sorted(unique_ranges, key=lambda x: x[0])
        
        merged_ranges = []
        start, end = sorted_ranges[0]
        
        for curr_start, curr_end in sorted_ranges[1:]:
            if curr_start <= end + 1:
                end = max(end, curr_end)
            else:
                merged_ranges.append([start, end])
                start, end = curr_start, curr_end
        
        merged_ranges.append([start, end])
        return merged_ranges


class UnionFind:
    """并查集类，用于合并重叠的缺陷上下文"""
    def __init__(self, size):
        self.parent = list(range(size))
        
    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]
        
    def union(self, x, y):
        px = self.find(x)
        py = self.find(y)
        if px != py:
            self.parent[py] = px

    async def cleanup(self):
        """清理资源"""
        logger.info("Cleaning up RAG service resources")
        # 这里可以添加资源清理逻辑
