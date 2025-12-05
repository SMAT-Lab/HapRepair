import logging
import sys
import os
import subprocess
import json
import tempfile
from typing import Optional, List, Dict, Any
from pathlib import Path

# 可选导入
try:
    from transformers import AutoTokenizer, AutoModel
    from pinecone import Pinecone
    HAS_ML_DEPS = True
except ImportError:
    AutoTokenizer = None
    AutoModel = None
    Pinecone = None
    HAS_ML_DEPS = False

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

try:
    # Only import functions that actually exist in llm.py to avoid ImportError
    from llm import get_openai_answer
    from get_prompt import generate_fix_prompt, get_rag_prompt
    from output_handler import extract_code_from_markdown_block, check_functionality
except ImportError as e:
    logging.warning(f"Could not import some modules: {e}")

logger = logging.getLogger(__name__)


def get_embedding(text, model, tokenizer):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1).squeeze().detach().numpy()
    return embeddings

def load_model_and_index():
    if not HAS_ML_DEPS:
        raise ImportError("ML dependencies not available. Install transformers and pinecone-client.")
    
    logging.getLogger().info("Loading model and index...")
    model_name = "dunzhang/stella_en_1.5B_v5"
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir="/home/models")
    model = AutoModel.from_pretrained(model_name, cache_dir="/home/models")
    
    pc = Pinecone(api_key="40075f49-8396-4571-924a-4b6d342cc81d")

    index_name = "arkts-1536"
    index = pc.Index(index_name)
    logging.getLogger().info("Model and index loaded")
    return model, tokenizer, index

class RepairService:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.index = None
        self.rules = None
        self.batch_tasks = {}  # 存储批处理任务状态
        
    async def initialize(self):
        """初始化服务所需的模型和资源"""
        try:
            logger.info("Initializing repair service...")
            
            # 加载模型和向量索引（可选，如果不使用RAG可以跳过）
            try:
                self.model, self.tokenizer, self.index = load_model_and_index()
                logger.info("Model and index loaded successfully")
            except Exception as e:
                logger.warning(f"Could not load model and index: {e}")
                logger.info("Service will work without RAG functionality")
            
            logger.info("Repair service initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize repair service: {e}")
            raise
    
    async def cleanup(self):
        """清理资源"""
        logger.info("Cleaning up repair service resources")
        # 这里可以添加资源清理逻辑
    
    
    async def detect_defects(self, code: str = None, detection_rules: Optional[List[str]] = None, project_path: Optional[str] = None) -> List[Dict[str, Any]]:
        """使用codelinter检测代码缺陷"""
        try:
            defects = []
            
            if project_path:
                # 如果提供了项目路径，直接在项目目录下运行codelinter
                defects = await self._run_codelinter_on_project(project_path, detection_rules)
            else:
                # 如果只有代码字符串，创建临时文件检测
                defects = await self._run_codelinter_on_code(code, detection_rules)
            
            return defects
            
        except Exception as e:
            logger.error(f"Error in defect detection: {e}")
            raise
    
    async def _run_codelinter_on_project(self, project_path: str, detection_rules: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """在harmony项目目录下运行codelinter"""
        try:
            # 检查项目路径是否存在
            if not os.path.exists(project_path):
                raise ValueError(f"Project path does not exist: {project_path}")
            
            # 构建codelinter命令
            cmd = ["codelinter"]

            # 统一使用 revision/code-linter.json5 作为配置文件（如果存在）
            config_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                "revision",
                "code-linter.json5",
            )
            if os.path.isfile(config_path):
                cmd.extend(["--config", config_path])
            
            # 添加输出格式参数，指定输出到json文件
            json_output_path = os.path.join(project_path, "codelinter_result.json")
            cmd.extend(["--output", "json"])
            
            # 在项目目录下执行命令
            logger.info(f"Running codelinter in {project_path}: {' '.join(cmd)}")
            result = subprocess.run(
                cmd,
                cwd=project_path,
                capture_output=True,
                text=True,
                timeout=300  # 5分钟超时
            )

            print("STDOUT:", result.stdout)
            print("STDERR:", result.stderr)
            
            # 检查是否生成了JSON输出文件
            json_files = [
                os.path.join(project_path, "json"),  # codelinter默认输出文件
                os.path.join(project_path, "codelinter_result.json"),
                os.path.join(project_path, "result.json")
            ]
            
            defects = []
            for json_file in json_files:
                if os.path.exists(json_file):
                    try:
                        with open(json_file, 'r', encoding='utf-8') as f:
                            json_content = f.read()
                        
                        # 如果是JSON格式，尝试解析
                        if json_content.strip().startswith('[') or json_content.strip().startswith('{'):
                            try:
                                json_data = json.loads(json_content)
                                defects = self._parse_json_defects(json_data)
                                logger.info(f"Successfully parsed {len(defects)} defects from {json_file}")
                                break
                            except json.JSONDecodeError:
                                logger.warning(f"Failed to parse JSON from {json_file}")
                        else:
                            # 如果是文本格式，使用文本解析
                            defects = self._parse_text_output(json_content)
                            logger.info(f"Successfully parsed {len(defects)} defects from text output in {json_file}")
                            break
                    except Exception as e:
                        logger.warning(f"Error reading {json_file}: {e}")
            
            # 如果没有找到JSON文件，尝试解析stdout/stderr
            if not defects:
                combined_output = result.stdout + "\n" + result.stderr
                defects = self._parse_text_output(combined_output)
            
            return defects
            
        except subprocess.TimeoutExpired:
            logger.error("Codelinter execution timed out")
            raise Exception("Code analysis timed out")
        except Exception as e:
            logger.error(f"Error running codelinter on project: {e}")
            raise
    
    async def _run_codelinter_on_code(self, code: str, detection_rules: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """对单个代码字符串运行codelinter"""
        try:
            # 创建临时目录和文件
            with tempfile.TemporaryDirectory() as temp_dir:
                # 创建临时.ets文件
                temp_file = Path(temp_dir) / "temp_code.ets"
                temp_file.write_text(code, encoding='utf-8')
                
                # 构建codelinter命令
                cmd = ["codelinter"]

                # 统一使用 revision/code-linter.json5 作为配置文件（如果存在）
                config_path = os.path.join(
                    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                    "revision",
                    "code-linter.json5",
                )
                if os.path.isfile(config_path):
                    cmd.extend(["--config", config_path])
                
                # 添加输出格式和目标文件
                cmd.extend(["--output", "json", str(temp_file)])
                
                logger.info(f"Running codelinter on temp file: {' '.join(cmd)}")
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=60  # 1分钟超时
                )
                
                # 解析codelinter输出
                defects = self._parse_codelinter_output(result.stdout, result.stderr, result.returncode)
                
                return defects
                
        except subprocess.TimeoutExpired:
            logger.error("Codelinter execution timed out")
            raise Exception("Code analysis timed out")
        except Exception as e:
            logger.error(f"Error running codelinter on code: {e}")
            raise
    
    def _parse_codelinter_output(self, stdout: str, stderr: str, return_code: int) -> List[Dict[str, Any]]:
        """解析codelinter输出结果"""
        defects = []
        
        try:
            if return_code != 0 and stderr:
                logger.warning(f"Codelinter returned non-zero exit code {return_code}: {stderr}")
            
            if stdout.strip():
                try:
                    # 尝试解析JSON输出
                    if stdout.startswith('[') or stdout.startswith('{'):
                        result_data = json.loads(stdout)
                        if isinstance(result_data, list):
                            defects = result_data
                        elif isinstance(result_data, dict) and 'defects' in result_data:
                            defects = result_data['defects']
                        else:
                            defects = [result_data]
                    else:
                        # 如果不是JSON格式，尝试解析文本输出
                        defects = self._parse_text_output(stdout)
                        
                except json.JSONDecodeError:
                    # JSON解析失败，尝试解析文本格式
                    defects = self._parse_text_output(stdout)
            
            # 标准化缺陷格式
            standardized_defects = []
            for defect in defects:
                standardized_defect = {
                    "rule": defect.get("rule", defect.get("ruleId", "unknown")),
                    "type": defect.get("type", defect.get("category", "unknown")),
                    "severity": defect.get("severity", defect.get("level", "medium")),
                    "description": defect.get("description", defect.get("message", "")),
                    "file": defect.get("file", defect.get("filePath", "")),
                    "line": defect.get("line", defect.get("lineNumber", 0)),
                    "column": defect.get("column", defect.get("columnNumber", 0))
                }
                standardized_defects.append(standardized_defect)
            
            return standardized_defects
            
        except Exception as e:
            logger.error(f"Error parsing codelinter output: {e}")
            # 如果解析失败，返回基本错误信息
            return [{
                "rule": "parse_error",
                "type": "error",
                "severity": "high",
                "description": f"Failed to parse codelinter output: {str(e)}",
                "file": "",
                "line": 0,
                "column": 0
            }]
    
    def _parse_text_output(self, text_output: str) -> List[Dict[str, Any]]:
        """解析codelinter的文本格式输出"""
        defects = []
        lines = text_output.strip().split('\n')
        
        current_file = ""
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # 跳过进度和状态信息
            if any(skip in line for skip in ['Working...', 'Finished...', 'Currently active', 'CodeLinter found', 'Defects:', 'The configuration file']):
                continue
            
            # 检查是否是文件路径行 (格式: /path/to/file.ets(数字))
            if line.endswith(')') and '(' in line and line.startswith('/'):
                # 提取文件路径
                if '(' in line:
                    current_file = line.split('(')[0]
                continue
            
            # 解析缺陷行 (格式: 行号:列号 级别 描述 @规则名)
            if ':' in line and current_file:
                try:
                    # 分割行号和其余部分
                    line_parts = line.split(':', 1)
                    if len(line_parts) >= 2:
                        line_number_str = line_parts[0].strip()
                        rest_content = line_parts[1].strip()
                        
                        # 尝试解析行号
                        try:
                            line_number = int(line_number_str)
                        except ValueError:
                            continue
                        
                        # 解析列号和其余内容 (格式可能是 "3 suggestion" 或 "11 warn")
                        content_parts = rest_content.split(None, 2)  # 最多分割成3部分
                        if len(content_parts) >= 2:
                            try:
                                column_number = int(content_parts[0])
                                severity = content_parts[1]
                                content = content_parts[2] if len(content_parts) > 2 else ""
                            except ValueError:
                                column_number = 0
                                severity = content_parts[0] if content_parts else "unknown"
                                content = ' '.join(content_parts[1:]) if len(content_parts) > 1 else ""
                        else:
                            column_number = 0
                            severity = "unknown"
                            content = rest_content
                        
                        # 解析描述和规则（content已经是描述部分）
                        if content:
                            content_parts = content.split()
                            
                            # 查找规则名（以@开头）
                            rule_name = "unknown"
                            description_parts = []
                            
                            for i, part in enumerate(content_parts):
                                if part.startswith('@'):
                                    rule_name = part[1:]  # 移除@符号
                                    description_parts = content_parts[:i]
                                    break
                                else:
                                    description_parts.append(part)
                            
                            description = ' '.join(description_parts).strip()
                        else:
                            rule_name = "unknown"
                            description = ""
                        
                        # 映射严重级别
                        severity_map = {
                            'suggestion': 'low',
                            'warn': 'medium', 
                            'error': 'high'
                        }
                        
                        # 从规则名推断类型
                        rule_type = "unknown"
                        if 'performance' in rule_name:
                            rule_type = "performance"
                        elif 'security' in rule_name:
                            rule_type = "security"
                        elif 'style' in rule_name:
                            rule_type = "style"
                        
                        defect = {
                            "rule": rule_name,
                            "type": rule_type,
                            "severity": severity_map.get(severity, severity),
                            "description": description,
                            "file": current_file,
                            "line": line_number,
                            "column": column_number,
                            "original_severity": severity
                        }
                        defects.append(defect)
                            
                except Exception as e:
                    logger.warning(f"Failed to parse defect line: {line}, error: {e}")
                    continue
        
        return defects
    
    def _parse_json_defects(self, json_data: any) -> List[Dict[str, Any]]:
        """解析JSON格式的codelinter输出"""
        defects = []
        
        try:
            # 处理不同的JSON格式
            if isinstance(json_data, list):
                # 如果是缺陷列表
                for item in json_data:
                    if isinstance(item, dict):
                        defect = self._normalize_defect_from_json(item)
                        if defect:
                            defects.append(defect)
            elif isinstance(json_data, dict):
                # 如果是包含缺陷的对象
                if 'defects' in json_data:
                    for item in json_data['defects']:
                        defect = self._normalize_defect_from_json(item)
                        if defect:
                            defects.append(defect)
                elif 'results' in json_data:
                    for item in json_data['results']:
                        defect = self._normalize_defect_from_json(item)
                        if defect:
                            defects.append(defect)
                else:
                    # 可能整个对象就是一个缺陷
                    defect = self._normalize_defect_from_json(json_data)
                    if defect:
                        defects.append(defect)
                        
        except Exception as e:
            logger.error(f"Error parsing JSON defects: {e}")
        
        return defects
    
    def _normalize_defect_from_json(self, item: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """从JSON项目规范化缺陷信息"""
        try:
            # 常见的字段映射
            rule = item.get('rule') or item.get('ruleId') or item.get('ruleName') or 'unknown'
            severity = item.get('severity') or item.get('level') or item.get('type') or 'medium'
            description = item.get('description') or item.get('message') or item.get('detail') or ''
            file_path = item.get('file') or item.get('filePath') or item.get('fileName') or ''
            line = item.get('line') or item.get('lineNumber') or 0
            column = item.get('column') or item.get('columnNumber') or 0
            
            # 映射严重级别
            severity_map = {
                'suggestion': 'low',
                'warn': 'medium',
                'warning': 'medium',
                'error': 'high',
                'critical': 'high'
            }
            
            # 推断类型
            defect_type = "unknown"
            if 'performance' in rule.lower():
                defect_type = "performance"
            elif 'security' in rule.lower():
                defect_type = "security"
            elif 'style' in rule.lower():
                defect_type = "style"
            
            return {
                "rule": rule,
                "type": defect_type,
                "severity": severity_map.get(severity.lower(), severity),
                "description": description,
                "file": file_path,
                "line": int(line) if isinstance(line, (int, str)) and str(line).isdigit() else 0,
                "column": int(column) if isinstance(column, (int, str)) and str(column).isdigit() else 0,
                "original_severity": severity
            }
            
        except Exception as e:
            logger.warning(f"Error normalizing defect from JSON: {e}")
            return None
    
    async def repair_code(
        self,
        code: str,
        repair_mode: str = "single",
        max_rounds: int = 3,
        use_rag: bool = True,
        model_name: str = "gpt-4o-mini"
    ) -> Dict[str, Any]:
        """修复代码缺陷"""
        try:
            repair_history = []
            current_code = code
            defects_found = await self.detect_defects(current_code)
            
            if not defects_found:
                return {
                    "success": True,
                    "repaired_code": code,
                    "defects_found": [],
                    "repair_rounds": 0,
                    "repair_history": [],
                    "message": "No defects found in the code"
                }
            
            rounds = 1 if repair_mode == "single" else max_rounds
            
            for round_num in range(rounds):
                try:
                    # 生成修复提示
                    if use_rag and self.model and self.tokenizer and self.index:
                        prompt = await self._generate_rag_prompt(current_code, defects_found)
                    else:
                        prompt = generate_fix_prompt(current_code, defects_found)
                    
                    # 调用LLM进行修复
                    # 目前仅通过 get_openai_answer 这一通路调用后端模型，
                    # deepseek 等模型名称也复用这一接口，避免依赖不存在的封装函数。
                    if "gpt" in model_name.lower() or "deepseek" in model_name.lower():
                        response = get_openai_answer(prompt, model_name)
                    else:
                        # 对于其他模型名，回退到默认模型
                        response = get_openai_answer(prompt, "gpt-4o-mini")
                    
                    # 提取修复后的代码
                    repaired_code = extract_code_from_markdown_block(response)
                    
                    # 记录本轮修复历史
                    repair_history.append({
                        "round": round_num + 1,
                        "defects": defects_found.copy(),
                        "prompt": prompt[:500] + "..." if len(prompt) > 500 else prompt,
                        "response": response[:500] + "..." if len(response) > 500 else response,
                        "repaired_code": repaired_code
                    })
                    
                    current_code = repaired_code
                    
                    # 如果是单轮修复，直接返回
                    if repair_mode == "single":
                        break
                    
                    # 检查是否还有缺陷
                    new_defects = await self.detect_defects(current_code)
                    if not new_defects:
                        logger.info(f"No more defects found after round {round_num + 1}")
                        break
                    
                    defects_found = new_defects
                    
                except Exception as e:
                    logger.error(f"Error in repair round {round_num + 1}: {e}")
                    if round_num == 0:  # 如果第一轮就失败，抛出异常
                        raise
                    break
            
            return {
                "success": True,
                "repaired_code": current_code,
                "defects_found": defects_found,
                "repair_rounds": len(repair_history),
                "repair_history": repair_history,
                "message": f"Code repair completed in {len(repair_history)} rounds"
            }
            
        except Exception as e:
            logger.error(f"Error in code repair: {e}")
            return {
                "success": False,
                "repaired_code": None,
                "defects_found": [],
                "repair_rounds": 0,
                "repair_history": [],
                "message": f"Code repair failed: {str(e)}"
            }
    
    async def verify_functionality(self, original_code: str, repaired_code: str) -> Dict[str, Any]:
        """验证修复后代码的功能完整性"""
        try:
            # 使用现有的功能检查逻辑
            functionality_preserved = check_functionality(original_code, repaired_code)
            
            # 简单的差异分析
            differences = []
            if original_code != repaired_code:
                differences.append({
                    "type": "code_change",
                    "description": "Code has been modified during repair",
                    "impact": "low" if functionality_preserved else "high"
                })
            
            return {
                "success": True,
                "functionality_preserved": functionality_preserved,
                "differences": differences,
                "message": "Functionality preserved" if functionality_preserved else "Functionality may be affected"
            }
            
        except Exception as e:
            logger.error(f"Error in functionality verification: {e}")
            return {
                "success": False,
                "functionality_preserved": False,
                "differences": [],
                "message": f"Verification failed: {str(e)}"
            }
    
    async def batch_repair(self, codes: List[str], task_id: str, defect_type: str = None, model_name: str = "gpt-4o-mini"):
        """批量修复代码（后台任务）"""
        try:
            self.batch_tasks[task_id] = {
                "status": "processing",
                "total": len(codes),
                "completed": 0,
                "results": [],
                "errors": []
            }
            
            for i, code in enumerate(codes):
                try:
                    result = await self.repair_code(
                        code=code,
                        defect_type=defect_type,
                        model_name=model_name
                    )
                    self.batch_tasks[task_id]["results"].append(result)
                    self.batch_tasks[task_id]["completed"] += 1
                    
                except Exception as e:
                    error_info = {
                        "index": i,
                        "error": str(e)
                    }
                    self.batch_tasks[task_id]["errors"].append(error_info)
                    logger.error(f"Error in batch repair item {i}: {e}")
            
            self.batch_tasks[task_id]["status"] = "completed"
            
        except Exception as e:
            self.batch_tasks[task_id]["status"] = "failed"
            self.batch_tasks[task_id]["error"] = str(e)
            logger.error(f"Batch repair task {task_id} failed: {e}")
    
    async def get_batch_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """获取批处理任务状态"""
        return self.batch_tasks.get(task_id)
    
    async def _generate_rag_prompt(self, code: str, defects: List[Dict[str, Any]]) -> str:
        """使用RAG生成修复提示"""
        try:
            if not self.model or not self.tokenizer or not self.index:
                return generate_fix_prompt(code, defects)
            
            # 获取代码嵌入
            embedding = get_embedding(code, self.model, self.tokenizer)
            
            # 从向量数据库检索相关示例
            query_result = self.index.query(
                vector=embedding.tolist(),
                top_k=3,
                include_metadata=True
            )
            
            # 构建RAG提示
            rag_prompt = get_rag_prompt(code, defects, query_result.matches)
            
            return rag_prompt
            
        except Exception as e:
            logger.warning(f"RAG prompt generation failed, falling back to simple prompt: {e}")
            return generate_fix_prompt(code, defects)
