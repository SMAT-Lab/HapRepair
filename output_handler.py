import json
import re
import logging

from get_prompt import get_functionality_check_prompt
from llm import get_answer, get_deepseek_answer, get_ollama_answer, get_openai_answer

def sort_json_lines(data):
    if isinstance(data, list):
        return sorted(data, key=lambda x: x.get('line', 0))
    return data

def handle_vul_type_res(res):
    # 先尝试直接解析整个响应
    try:
        data = json.loads(res)
        data = sort_json_lines(data)
        return data
    except json.JSONDecodeError:
        # 如果直接解析失败,尝试提取JSON格式数据块
        json_match = re.search(r'```json\n(.*?)\n```', res, flags=re.DOTALL)
        if json_match:
            cleaned_res = json_match.group(1)
        else:
            logging.getLogger().error("未找到JSON数据块")
            return None
        
        try:
            data = json.loads(cleaned_res)
            data = sort_json_lines(data)
            return data
        except json.JSONDecodeError as e:
            logging.getLogger().error(f"JSON解析错误: {e}")
            return cleaned_res


def handle_result(res):
    start = res.find('```json')
    if start == -1:
        logging.getLogger().error("未找到JSON数据块")
        return None
    start += 7
    end = res.find('```', start)
    if end == -1:
        json_str = res[start:].strip()
    else:
        json_str = res[start:end].strip()
    
    try:
        result_dict = json.loads(json_str)
        return {
            'problem_code': result_dict['problem_code'],
            'problem_fix': result_dict['problem_fix']
        }
    except json.JSONDecodeError as e:
        logging.getLogger().error(f"JSON解析错误: {e}")
        return None

def extract_code_from_markdown_block(markdown_block):
    try:
        # 先尝试查找代码块
        match = re.search(r'```(?:arkts|javascript|js|ts|typescript)\n(.*)\n```', markdown_block, re.DOTALL)
        if match is not None:
            return match.group(1)
        
        # 如果没找到代码块,说明可能直接返回了代码,直接返回原文本
        logging.getLogger().info("未找到代码块,返回原始文本")
        return markdown_block
    except Exception as e:
        logging.getLogger().error(f"提取代码块时出错: {str(e)}")
        return markdown_block
    
def remove_difflib_line(code):
    lines = code.split('\n')
    result_lines = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith('-'):
            # Skip the - line and keep the + line without the + prefix
            i += 1
        elif lines[i].startswith('+'):
            result_lines.append(line[1:])
        else:
            result_lines.append(line)
        i += 1
    return '\n'.join(result_lines)

## 检查arkts的代码括号是否匹配，不匹配的话就进行匹配
def fix_brackets(code):
    # Store bracket pairs
    brackets_map = {')': '(', '}': '{', ']': '['}
    opening_brackets = set('({[')
    closing_brackets = set(')}]')
    stack = []
    lines = code.split('\n')
    fixed_lines = []
    errors = []
    
    # Process each line
    for i, line in enumerate(lines, 1):
        stripped = line.strip()
        if not stripped:
            fixed_lines.append(line)
            continue
            
        # Check for misplaced method chain
        if stripped.startswith('}') and '.tabBar' in stripped:
            # Remove the leading brace and keep the method chain
            stripped = stripped[1:].strip()
            errors.append(f"Line {i}: Removed misplaced closing brace before .tabBar()")
            
        indent = len(line) - len(line.lstrip())
        fixed_line = ' ' * indent + stripped
        fixed_lines.append(fixed_line)
        
        # Track brackets
        for char in stripped:
            if char in opening_brackets:
                stack.append((char, i))
            elif char in closing_brackets:
                if not stack:
                    errors.append(f"Line {i}: Extra closing bracket '{char}'")
                    continue
                opening, _ = stack.pop()
                if opening != brackets_map[char]:
                    errors.append(f"Line {i}: Mismatched brackets, found '{char}' for '{opening}'")
    
    # Check for unclosed brackets
    while stack:
        bracket, line_num = stack.pop()
        # Find matching closing bracket
        closing_bracket = None
        for close, open_bracket in brackets_map.items():
            if open_bracket == bracket:
                closing_bracket = close
                break
        if closing_bracket:
            errors.append(f"Line {line_num}: Unclosed bracket '{bracket}'")
    
    fixed_code = '\n'.join(fixed_lines)
    return (fixed_code, errors) if errors else (code, [])

import re
from dataclasses import dataclass
from typing import List, Optional, Set

@dataclass
class ValidationIssue:
    line_number: int
    original_line: str 
    issue_type: str
    suggested_fix: str
    struct_declaration: Optional[str] = None
    
@dataclass
class RepairResult:
    fixed_code: str
    changes: List[str]

class ArkTSDeclarationFixer:
    def __init__(self):
        # Core declaration patterns
        self.decorator_pattern = re.compile(r'@(\w+)(?:\([^)]*\))?\s+\w+\s*:')
        self.let_const_pattern = re.compile(r'(?:let|const)\s+(\w+)(?:\s*:\s*[^=]+)?\s*=\s*([^;]+)')
        self.private_pattern = re.compile(r'private\s+(\w+)(?:\s*:\s*[^=]+)?\s*=')
        self.direct_assign_pattern = re.compile(r'^(\w+)(?:\s*:\s*[^=]+)?\s*=')
        self.state_pattern = re.compile(r'@State\s+(\w+)\s*:')
        self.struct_pattern = re.compile(r'(?:@\w+\s+)?(?:export\s+)?struct\s+\w+\s*{')
        self.type_annotation_pattern = re.compile(r':\s*([^=]+?)\s*(?==|$)')
        
        # Enhanced function detection patterns
        self.function_pattern = re.compile(
            r'\s*'                     # Allow any leading whitespace
            r'(?:private\s+)?'         # Optional private modifier
            r'(?:async\s+)?'           # Optional async modifier
            r'[a-zA-Z_]\w*'            # Function name
            r'\s*\([^)]*\)'            # Parameters with any content
            r'\s*(?::\s*[^{;]+)?'      # Optional return type (up to { or ;)
            r'\s*{'                    # Function body start
        )
        
        # Separate patterns for special function types
        self.build_pattern = re.compile(r'\s*build\s*\(\s*\)\s*{')
        self.callback_pattern = re.compile(
            r'\s*'                     # Allow any leading whitespace
            r'(?:onClick|onAppear|onChange|onTouch|aboutTo(?:Appear|Disappear)|onVisibility(?:Change)|onPageShow|onBackPress|onPageHide)'  # Event handler names
            r'\s*\(\s*'                # Opening parenthesis
            r'(?:\([^)]*\))?\s*=>'     # Optional parameters and arrow
        )
        self.arrow_function_pattern = re.compile(
            r'\s*'                     # Allow any leading whitespace
            r'(?:'                     # Start non-capturing group for all arrow function forms
                r'(?:const\s+)?[a-zA-Z_]\w*\s*=\s*'  # Named arrow function 
                r'|'                   # OR
                r'func\s*:\s*'         # Object property arrow function
                r'|'                   # OR
                r':\s*'                # Simple colon and whitespace
            r')?'                      # End non-capturing group
            r'\([^)]*\)\s*=>'         # Parameters and arrow
        )
        
        # Valid decorator list with parameter patterns
        self.valid_decorators = {
            '@State', '@Prop', '@Link', '@ObjectLink', '@Provide', '@Consume', '@Watch', '@StorageLink', '@StorageProp'
        }

    def validate_and_fix(self, code: str) -> Optional[RepairResult]:
        lines = code.split('\n')
        issues: List[ValidationIssue] = []
        fixed_lines = lines.copy()
        struct_declarations: List[str] = []
        
        # Track scopes
        in_struct = False
        in_function = False
        in_build = False
        in_callback = False
        in_arrow_function = False  # Track arrow function scope
        struct_start_line = 0
        struct_indent = ""
        struct_brace_count = 0
        function_brace_count = 0
        callback_brace_count = 0
        arrow_brace_count = 0  # Track braces in arrow functions
        
        for line_num, line in enumerate(lines, 1):
            trimmed = line.strip()
            
            # Skip empty lines and comments
            if not trimmed or trimmed.startswith('//'):
                continue
                
            # Track struct scope and indentation
            if self.struct_pattern.search(trimmed):
                in_struct = True
                struct_brace_count = 1
                struct_start_line = line_num
                struct_indent = re.match(r'^\s*', line).group()
                continue
            
            # Check if entering arrow function
            if '=>' in trimmed and not in_arrow_function:
                in_arrow_function = True
                arrow_brace_count = trimmed.count('{')
                continue
                
            # Track all types of function scopes
            is_function_start = (
                self.function_pattern.match(trimmed) or 
                self.build_pattern.match(trimmed) or 
                self.callback_pattern.match(trimmed) or 
                self.arrow_function_pattern.match(trimmed)
            )
            
            if is_function_start and not in_arrow_function:
                in_function = True
                function_brace_count = 1
                if self.build_pattern.match(trimmed):
                    in_build = True
                elif self.callback_pattern.match(trimmed):
                    in_callback = True
                continue
            
            # Update arrow function brace count
            if in_arrow_function:
                arrow_brace_count += trimmed.count('{')
                arrow_brace_count -= trimmed.count('}')
                if arrow_brace_count <= 0:
                    in_arrow_function = False
                    arrow_brace_count = 0
                continue
                
            # Update callback scope tracking
            if in_callback:
                callback_brace_count += trimmed.count('{')
                callback_brace_count -= trimmed.count('}')
                if callback_brace_count <= 0:
                    in_callback = False
                    callback_brace_count = 0
            
            if in_function:
                function_brace_count += trimmed.count('{')
                function_brace_count -= trimmed.count('}')
                if function_brace_count <= 0:
                    in_function = False
                    in_build = False
                    in_callback = False
                    function_brace_count = 0
                    callback_brace_count = 0
            
            if in_struct:
                struct_brace_count += trimmed.count('{')
                struct_brace_count -= trimmed.count('}')
                if struct_brace_count <= 0:
                    in_struct = False
                    
            # Validate declarations based on scope
            if in_struct and not in_function and not in_arrow_function:
                # Check for invalid let/const in struct scope
                if self.let_const_pattern.search(trimmed):
                    issue = self._create_let_const_in_struct_issue(line, line_num)
                    if issue:
                        issues.append(issue)
                        fixed_lines[line_num - 1] = issue.suggested_fix
                        
                # Validate decorators
                if trimmed.startswith('@'):
                    decorator_match = self.decorator_pattern.search(trimmed)
                    if decorator_match:
                        decorator_name = f'@{decorator_match.group(1)}'
                        if decorator_name not in self.valid_decorators:
                            issue = ValidationIssue(
                                line_number=line_num,
                                original_line=line,
                                issue_type=f"Invalid decorator {decorator_name}. Must be one of {', '.join(sorted(self.valid_decorators))}",
                                suggested_fix=line
                            )
                            issues.append(issue)
                            
            elif in_build and not in_callback and not in_arrow_function:
                # Check for variable declarations in build scope
                if let_const_match := self.let_const_pattern.search(trimmed):
                    var_name = let_const_match.group(1)
                    init_value = let_const_match.group(2)
                    
                    # Extract type annotation if present
                    type_match = self.type_annotation_pattern.search(trimmed)
                    type_annotation = f": {type_match.group(1)}" if type_match else ""
                    
                     # Check if this is a multi-line declaration (ends with {)
                    if trimmed.rstrip().endswith('{'):
                        # Track braces to find the end of declaration
                        brace_count = 1
                        declaration_lines = [line]  # Start with the first line
                        end_line_num = line_num
                        
                        # Keep collecting lines until we find matching }
                        for next_line_num in range(line_num, len(lines)):
                            next_line = lines[next_line_num]
                            next_trimmed = next_line.strip()
                            if not next_trimmed:
                                continue
                                
                            if next_trimmed != line.strip():  # Don't count the first line twice
                                declaration_lines.append(next_line)
                                brace_count += next_trimmed.count('{')
                                brace_count -= next_trimmed.count('}')
                                
                            if brace_count == 0:
                                end_line_num = next_line_num
                                break
                            
                        # Combine all lines with proper indentation
                        full_declaration = '\n'.join(declaration_lines)
                        
                        # Create struct-level declaration
                        struct_decl = f"{struct_indent}  {var_name}{type_annotation} = {full_declaration}"
                        
                        # Create issue
                        issue = ValidationIssue(
                            line_number=line_num,
                            original_line=full_declaration,
                            issue_type="Variable declarations are not allowed in build method. Moving to struct scope.",
                            suggested_fix="",
                            struct_declaration=struct_decl
                        )
                        issues.append(issue)
                        
                        # Remove all lines of the declaration
                        for i in range(line_num - 1, end_line_num + 1):
                            fixed_lines[i] = ""
                            
                        # Add to struct declarations
                        struct_declarations.append(struct_decl)
                        
                        # Find and update all references to this variable in the rest of the code
                        for i in range(end_line_num + 1, len(fixed_lines)):
                            fixed_lines[i] = re.sub(
                                r'\b' + var_name + r'\b',  # Match whole word
                                f'this.{var_name}',        # Replace with this.var
                                fixed_lines[i]
                            )
                    else:
                        # Single line declaration
                        struct_decl = f"{struct_indent}  {var_name}{type_annotation} = {init_value}"
                        
                        issue = ValidationIssue(
                            line_number=line_num,
                            original_line=line,
                            issue_type="Variable declarations are not allowed in build method. Moving to struct scope.",
                            suggested_fix="",
                            struct_declaration=struct_decl
                        )
                        issues.append(issue)
                        
                        fixed_lines[line_num - 1] = ""
                        struct_declarations.append(struct_decl)
                        
                        for i in range(line_num, len(fixed_lines)):
                            fixed_lines[i] = re.sub(
                                r'\b' + var_name + r'\b',
                                f'this.{var_name}',
                                fixed_lines[i]
                            )
                    
        if not issues:
            return None
            
        # Insert struct declarations after struct opening
        if struct_declarations:
            fixed_lines[struct_start_line:struct_start_line] = struct_declarations

        # 清理连续空行
        cleaned_lines = []
        prev_empty = False
        for line in fixed_lines:
            if line.strip():
                cleaned_lines.append(line)
                prev_empty = False
            elif not prev_empty:
                cleaned_lines.append(line)
                prev_empty = True
                
        return RepairResult(
            fixed_code='\n'.join(cleaned_lines),
            changes=[f"Line {issue.line_number}: {issue.issue_type}\n"
                    f"  Found: {issue.original_line.strip()}\n"
                    f"  Fixed: {issue.suggested_fix.strip()}"
                    + (f"\n  Added to struct: {issue.struct_declaration}" if issue.struct_declaration else "")
                    for issue in issues]
        )
    
    def _create_let_const_in_struct_issue(self, line: str, line_num: int) -> Optional[ValidationIssue]:
        """Create issue for let/const declaration in struct scope"""
        match = self.let_const_pattern.search(line.strip())
        if not match:
            return None
            
        var_name = match.group(1)
        indentation = re.match(r'^\s*', line).group()
        
        # Preserve type annotation if exists
        trimmed = line.strip()
        colon_index = trimmed.find(':')
        if colon_index != -1:
            fixed_line = f"{indentation}{var_name}{trimmed[colon_index:]}"
        else:
            equal_index = trimmed.find('=')
            fixed_line = f"{indentation}{var_name}{trimmed[equal_index:]}"
        
        return ValidationIssue(
            line_number=line_num,
            original_line=line,
            issue_type="'let/const' declarations are not allowed in struct scope. Use @State, private, or direct declaration instead.",
            suggested_fix=fixed_line
        )
    
    def _create_private_in_function_issue(self, line: str, line_num: int) -> Optional[ValidationIssue]:
        """Create issue for private declaration in function scope"""
        match = self.private_pattern.search(line.strip())
        if not match:
            return None
            
        var_name = match.group(1)
        indentation = re.match(r'^\s*', line).group()
        
        # Preserve type annotation if exists
        trimmed = line.strip()
        colon_index = trimmed.find(':')
        if colon_index != -1:
            fixed_line = f"{indentation}let {var_name}{trimmed[colon_index:]}"
        else:
            equal_index = trimmed.find('=')
            fixed_line = f"{indentation}let {var_name}{trimmed[equal_index:]}"
        
        return ValidationIssue(
            line_number=line_num,
            original_line=line,
            issue_type="'private' declarations are not allowed in function scope. Use 'let' or 'const' instead.",
            suggested_fix=fixed_line
        )
    
def check_functionality(original_code, repaired_code):
    prompt = get_functionality_check_prompt(original_code, repaired_code)
    res = get_answer(prompt, model_name='gpt-4o-2024-08-06')
    #res = get_ollama_answer(prompt, model_name='qwen2.5-coder:32b')r
    #res = get_deepseek_answer(prompt)
    # 移除可能存在的```json前缀
    res = res.replace('```json', '').replace('```', '').strip()
    res_json = json.loads(res)
    return res_json