from transformers import AutoTokenizer, AutoModel
import torch
from sklearn.metrics.pairwise import cosine_similarity
import json
import re
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from llm import get_openai_answer

def get_context_prompt():
    """获取上下文提取的提示模板"""
    return """
    You are an AI debugging assistant. Your task is to fix the provided code based on the error log.

    ### Input:
    Code snippets:
    {code_snippet}

    Error log:
    {error_log}

    ### Task:
    Please analyze the code and error log, then provide the fixed code directly.
    Keep the original indentation of each code snippet.

    ### Output format:
    Return the fixed code snippets with original indentation preserved.
    """

def get_repair_prompt():
    """获取代码修复的提示模板"""
    return """请帮我修改以下ArkTS代码。

原始代码:
{original_code}

提取出的有关上下文代码内容及行号：
{context}

对于上下文的修改代码:
{diff}

请返回完整的修改后代码，直接返回代码，不要返回任何其它内容。
"""

def get_repair_suggestion_prompt():
    """获取修复建议的提示模板"""
    return """
    You are an AI debugging assistant. Your task is to fix the provided code based on the error log.

    ### Input:
    Code snippets:
    {code_snippet}

    Error log:
    {error_log}

    ### Task:
    Please analyze the code and error log, then provide the fixed code directly.
    Keep the original indentation of each code snippet.

    ### Output format:
    Return the fixed code snippets with original indentation preserved.
    """

class CodeEmbedding:
    def __init__(self, model_name="microsoft/graphcodebert-base", cache_dir="/home/models"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
        self.model = AutoModel.from_pretrained(model_name, cache_dir=cache_dir)
    
    def get_embedding(self, code_snippet):
        inputs = self.tokenizer(code_snippet, return_tensors="pt", truncation=True, 
                              padding=True, max_length=512)
        outputs = self.model(**inputs)
        return outputs.last_hidden_state[:, 0, :].detach()
    
    def calculate_similarity(self, vec1, vec2):
        return cosine_similarity(vec1.reshape(1, -1), vec2.reshape(1, -1))[0][0]
    
    def get_similar_contexts(self, defect_code, context_snippets):
        defect_vec = self.get_embedding(defect_code)
        context_vectors = [self.get_embedding(snippet) for snippet in context_snippets]
        similarities = [self.calculate_similarity(defect_vec, ctx_vec) 
                       for ctx_vec in context_vectors]
        
        return sorted(zip(context_snippets, similarities), 
                     key=lambda x: x[1], reverse=True)

class CodeContextExtractor:
    def __init__(self):
        self.prompt_template = get_context_prompt()
    
    def extract_arkts_context(self, lines, line_num: int = None):
        # 从错误行提取变量名
        error_line_content = lines[line_num]
        # 匹配@State等装饰器后的变量名
        # 先匹配赋值的变量
        assign_vars = re.findall(r'@\w+\s+(\w+)|let\s+(\w+)\s*=', error_line_content)
        # 如果没有赋值的变量,再匹配使用的变量
        if not assign_vars:
            variable_names = re.findall(r'this\.(\w+)', error_line_content)
        else:
            variable_names = assign_vars
        variable_names = [name for name in variable_names if name != '']

        # 从匹配组中获取非空值
        variable_names = [name for group in variable_names for name in group if name]
        context = {
            "definition": [],
            "usage": [], 
            "blocks": []
        }
        
        for variable_name in variable_names:
            definition_pattern = rf"(@\w+.*{variable_name}|let.*{variable_name}.*=).*;"
            usage_pattern = rf".*this\.{variable_name}.*"
            
            # 提取代码块
            blocks = []
            if line_num is not None:
                begin_line, end_line, block = self._extract_blocks(lines, line_num)
                if block:
                    blocks.append((begin_line, end_line, block))
            else:
                for i, line in enumerate(lines):
                    if re.search(definition_pattern, line) or re.search(usage_pattern, line):
                        begin_line, end_line, block = self._extract_blocks(lines, i)
                        if block:
                            blocks.append((begin_line, end_line, block))
            
            context["blocks"] = self._filter_nested_blocks(blocks)
            
            # 提取定义和使用
            for i, line in enumerate(lines):
                self._process_line(i, line, definition_pattern, usage_pattern, 
                                 context, context["blocks"])
        
        return context
    
    def _extract_blocks(self, lines, line_number):
        """
        提取代码块的完整范围。
        支持普通代码块和回调函数块。
        """
        # 匹配ArkTS中代码块或组件的起始模式
        block_pattern = r"(for|if|while|ForEach|LazyForEach|List|ListItem|Grid|Flex|Column|Row|Stack|Scroll|Swiper|TabContent|Panel|Navigation|RelativeContainer|WaterFlow|\.on\w+|aboutToReuse)\s*.*\{"

        # 找到最外层的代码块
        start_idx = line_number
        outer_block = None
        while start_idx >= 0:
            # print(len(lines))
            if re.search(block_pattern, lines[start_idx]):
                block_end = self._find_block_end(lines, start_idx)
                # 确认目标行在block范围内
                if line_number < block_end:
                    # 将连续的block合并为一个字符串
                    block_content = "\n".join(lines[start_idx:block_end])
                    outer_block = (start_idx, block_end - 1, block_content)
            start_idx -= 1

        if outer_block:
            return outer_block

        # 如果没有匹配到完整代码块，则返回当前行
        return line_number, line_number, lines[line_number]

    def _find_block_end(self, lines, start_idx):
        """
        找到代码块的结束位置。
        """
        brace_count = 1  # 记录花括号的数量
        block_end = start_idx + 1
    
        while block_end < len(lines):
            current_line = lines[block_end].strip()
    
            # 忽略空行
            if not current_line:
                block_end += 1
                continue
            
            # 处理花括号
            brace_count += current_line.count('{') - current_line.count('}')
            if brace_count == 0:
                return block_end + 1
    
            block_end += 1
    
        return block_end

        
    def _find_callback_end(self, lines, start_idx):
        brace_count = 1
        current_idx = start_idx + 1
        
        while current_idx < len(lines):
            current_line = lines[current_idx]
            if not current_line.strip():
                current_idx += 1
                continue
                
            brace_count += current_line.count('{') - current_line.count('}')
            if brace_count == 0:
                return current_idx + 1
                
            current_idx += 1
            
        return current_idx
    
    def _filter_nested_blocks(self, blocks):
        filtered_blocks = []
        for start, end, block in blocks:
            if not any(start > parent_start and start < parent_start + len(parent_block) 
                      for parent_start, _, parent_block in filtered_blocks):
                filtered_blocks.append((start, end, block))
        return filtered_blocks
    
    def _process_line(self, i, line, def_pattern, usage_pattern, context, blocks):
        if re.search(def_pattern, line) or re.search(usage_pattern, line):
            if not self._is_in_block(i + 1, blocks):
                target_list = context["definition"] if re.search(def_pattern, line) \
                            else context["usage"]
                target_list.append((i, line))  # 修改这里,只保留行号和代码内容
    
    def _is_in_block(self, line_num, blocks):
        return any(start <= line_num <= end 
                  for start, end, _ in blocks)

class CodeRepair:
    def __init__(self):
        self.repair_prompt = get_repair_prompt()

    def modify_arkts_code(self, original_code, context, surrounding_context, error_log=""):
        """修改ArkTS代码"""
        # 首先获取修改建议
        diff = self._get_repair_suggestion(surrounding_context, error_log)
        
        # 然后基于建议生成完整的修复代码
        prompt = self.repair_prompt.format(
            original_code=original_code,
            context=context,
            diff=diff
        )
        
        # 获取修改建议
        response = get_openai_answer(prompt, model_name="gpt-4o-mini")
        
        # 提取代码
        modified_code = extract_code_from_response(response)
        
        return modified_code
    
    def _get_repair_suggestion(self, surrounding_context, error_log):
        """获取代码修复建议"""
        prompt_template = get_repair_suggestion_prompt()
        
        prompt = prompt_template.format(
            code_snippet=surrounding_context, 
            error_log=error_log
        )
        
        response = get_openai_answer(prompt, model_name="gpt-4o-mini")
        return extract_code_from_response(response)

def extract_code_from_response(response):
    """从回复中提取代码块内容"""
    code_blocks = re.findall(r'```(?:\w+)?\n(.*?)```', response, re.DOTALL)
    return '\n'.join(code_blocks) if code_blocks else response
