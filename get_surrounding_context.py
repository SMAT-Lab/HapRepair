import glob
import json
import os
import pandas as pd
import warnings
from code_repair import CodeContextExtractor

# 忽略特定的 UserWarning
warnings.filterwarnings("ignore", category=UserWarning, module="openpyxl")

def load_rules():
    """加载规则文件并添加缺陷代码"""
    rules_data = json.load(open('rules.json', 'r', encoding='utf-8'))
    
    for rule in rules_data:
        file_name = rule['rule'].split('/')[-1]
        with open(f'./pages/negative/{file_name}.ets', 'r', encoding='utf-8') as f:
            rule['defect_code'] = f.read()
            
    return {rule['rule']: rule for rule in rules_data}

def get_defects_from_file(proj_dir, file_path):
    """从文件中获取缺陷信息"""
    with open(file_path, 'r', encoding='utf-8') as f:
        code = f.read()
    code_lines = [''] + code.split('\n')
    
    result_files = glob.glob(os.path.join(proj_dir,'result*.xlsx'))
    file_df = pd.read_excel(result_files[0], header=1)
    # 获取相对路径的最后几层目录结构
    rel_path = os.path.relpath(file_path, proj_dir)
    
    # 过滤出'Source File'含有rel_path的记录
    mask = file_df['Source File'].apply(lambda x: rel_path.replace('/', '\\') in x.replace('/', '\\'))
    file_df = file_df[mask]
    
    defects = []
    for _, row in file_df.iterrows():
        line_num = int(row['Line'])
        rule_name = row['RuleName']
        message = row['Detail']

        # 提取完整的条件块
        extracted_code = []
        current_line = line_num

        while current_line < len(code_lines):
            line_content = code_lines[current_line].strip()
            extracted_code.append(line_content)
            if line_content.endswith("{") or line_content.endswith(";") or line_content.endswith(')') or line_content.endswith('"') or line_content.endswith("'") or line_content.startswith('@') or line_content.startswith('private'):  # 遇到条件结束符号 `{` 停止
                break
            current_line += 1

        # 拼接多行条件
        code_snippet = "\n".join(extracted_code)

        # 创建缺陷信息
        defect = {
            "rule": rule_name,
            "line": line_num,
            "message": message,
            "code": code_snippet
        }
        defects.append(defect)
        
    return code_lines, sorted(defects, key=lambda x: x['line'])

def process_code_blocks(defect, code_lines, rules_dict):
    """处理代码块"""
    extractor = CodeContextExtractor()
    start_idx, block_end, surrounding_context = extractor._extract_blocks(code_lines, defect['line'])
    if surrounding_context is None:
        surrounding_context = defect['code']
        
    block_ranges = [(start_idx, block_end)]
    
    if rules_dict[defect['rule']]['needsMoreContext']:
        context = extractor.extract_arkts_context(code_lines, defect['line'])
        code_blocks = []

        if surrounding_context:
            # 检查是否有重叠的代码块
            is_contained = False
            for block in code_blocks:
                block_start, block_end, _ = block
                if start_idx >= block_start and block_end >= block_end:
                    is_contained = True
                    break
                elif start_idx <= block_start and block_end <= block_end:
                    code_blocks.remove(block)
            if not is_contained:
                code_blocks.append((start_idx, block_end, surrounding_context))

        for definition in context['definition']:
            def_start, content = definition
            # 检查是否有重叠
            is_contained = False
            for block in code_blocks:
                block_start, block_end, _ = block
                if def_start >= block_start and def_start <= block_end:
                    is_contained = True
                    break
            if not is_contained:
                block_ranges.append((def_start, def_start))
                code_blocks.append((def_start, def_start, content))
            
        for usage in context['usage']:
            use_start, content = usage
            # 检查是否有重叠
            is_contained = False 
            for block in code_blocks:
                block_start, block_end, _ = block
                if use_start >= block_start and use_start <= block_end:
                    is_contained = True
                    break
            if not is_contained:
                block_ranges.append((use_start, use_start))
                code_blocks.append((use_start, use_start, content))
            
        code_blocks.sort(key=lambda x: x[0])
        block_ranges.sort(key=lambda x: x[0])
        surrounding_context = '\n'.join(content for _, _, content in code_blocks)
        
    return block_ranges, surrounding_context

class UnionFind:
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

def ranges_overlap(ranges1, ranges2):
    for start1, end1 in ranges1:
        for start2, end2 in ranges2:
            # Include adjacent ranges as overlapping
            if not (end2 + 1 < start1 or start2 - 1 > end1):
                return True
    return False

def remove_redundant_nest_container(line, code_lines):
    # 移除冗余的container
    
    # 找到匹配的右大括号
    stack = []
    start_line = line
    end_line = line
    
    # 从当前行开始往下寻找
    for i in range(line, len(code_lines)):
        current = code_lines[i]
        
        # 统计左右大括号
        left_count = current.count('{')
        right_count = current.count('}')
        
        # 更新栈
        for _ in range(left_count):
            stack.append('{')
        for _ in range(right_count):
            if stack:
                stack.pop()
                
        # 如果栈为空,说明找到匹配的右大括号
        if not stack:
            end_line = i
            break
            
    # 清空首尾行
    code_lines[start_line] = ''
    code_lines[end_line] = code_lines[end_line].replace('}', '', 1)  # 只替换第一个 '}'
    
    # 处理中间行,去除两格缩进
    for i in range(start_line + 1, end_line):
        if code_lines[i].startswith('  '):
            code_lines[i] = code_lines[i][2:]

    return code_lines

def remove_container_without_property(line, code_lines):
    # 移除没有属性的container
    stack = []
    start_line = line
    end_line = line
    
    # 从当前行开始往下寻找
    for i in range(line, len(code_lines)):
        current = code_lines[i]
        
        # 统计左右大括号
        left_count = current.count('{')
        right_count = current.count('}')
        
        # 更新栈
        for _ in range(left_count):
            stack.append('{')
        for _ in range(right_count):
            if stack:
                stack.pop()
                
        # 如果栈为空,说明找到匹配的右大括号
        if not stack:
            end_line = i
            break
            
    # 清空首尾行
    code_lines[start_line] = ''
    code_lines[end_line] = code_lines[end_line].replace('}', '', 1)  # 只替换第一个 '}'
    
    # 处理中间行,去除两格缩进
    for i in range(start_line + 1, end_line):
        if code_lines[i].startswith('  '):
            code_lines[i] = code_lines[i][2:]

    return code_lines

def use_row_column_to_replace_flex(line, code_lines):
    # 获取原始行的缩进
    indent = len(code_lines[line]) - len(code_lines[line].lstrip())
    indent_str = ' ' * indent
    
    if 'Column' in code_lines[line]:
        code_lines[line] = indent_str + "Column() {"
    elif 'Row' in code_lines[line]:
        code_lines[line] = indent_str + "Row() {"
    return code_lines

def get_single_file_surrounding_context(proj_dir, file_path, rules_dict):
    """处理单个文件的缺陷检测"""

    code_lines, defects = get_defects_from_file(proj_dir, file_path)

    if len(defects) == 0:
        return [], code_lines
    
    print('-'*100)
    print(os.path.basename(file_path))
    
    all_blocks = []
    
    for defect in defects:
        if defect['rule'] == '@performance/hp-arkui-remove-redundant-nest-container':
            # 直接进行处理
            code_lines = remove_redundant_nest_container(defect['line'], code_lines)
        elif defect['rule'] == '@performance/hp-arkui-remove-container-without-property':
            code_lines = remove_container_without_property(defect['line'], code_lines)

        elif defect['rule'] == '@performance/hp-arkui-use-row-column-to-replace-flex':
            code_lines = use_row_column_to_replace_flex(defect['line'], code_lines)
    for defect in defects:
        if defect['rule'] == '@performance/hp-arkui-remove-redundant-nest-container' or defect['rule'] == '@performance/hp-arkui-use-row-column-to-replace-flex' or defect['rule'] == '@performance/hp-arkui-remove-container-without-property':
            continue
        if defect['rule'] in rules_dict:
            block_ranges, surrounding_context = process_code_blocks(defect, code_lines, rules_dict)
            # 转换为JSON格式
            all_blocks.append({
                "defect": defect,
                "block_ranges": block_ranges,
                "surrounding_context": surrounding_context
            })

    # print(json.dumps(all_blocks, indent=2))
    # 合并重叠的blocks
    uf = UnionFind(len(all_blocks))
    
    # Step 1: Union overlapping blocks
    for i in range(len(all_blocks)):
        for j in range(i + 1, len(all_blocks)):
            if ranges_overlap(all_blocks[i]['block_ranges'], all_blocks[j]['block_ranges']):
                uf.union(i, j)
    
    # Step 2: Group blocks by their root parent
    groups = {}
    for i in range(len(all_blocks)):
        parent = uf.find(i)
        if parent not in groups:
            groups[parent] = []
        groups[parent].append(all_blocks[i])
    
    # Step 3: Merge blocks in each group
    merged_blocks = []
    for group_blocks in groups.values():
        merged_defects = []
        all_ranges = []
        max_context = ""
        for block in group_blocks:
            merged_defects.append(block['defect'])
            all_ranges.extend(block['block_ranges'])
            # Choose the longest context (or concatenate if preferred)
            if len(block['surrounding_context']) > len(max_context):
                max_context = block['surrounding_context']
        # Remove duplicate ranges
        all_ranges = [list(x) for x in set(tuple(x) for x in all_ranges)]
        # Merge overlapping ranges
        sorted_ranges = sorted(all_ranges, key=lambda x: x[0])
        merged_ranges = []
        start, end = sorted_ranges[0]
        for curr_start, curr_end in sorted_ranges[1:]:
            if curr_start <= end + 1:
                end = max(end, curr_end)
            else:
                merged_ranges.append([start, end])
                start, end = curr_start, curr_end
        merged_ranges.append([start, end])

        surrounding_context = []
        for rng in merged_ranges:
            rng_start, rng_end = rng
            # 调整索引（代码行从第1行开始）
            code_snippet = '\n'.join(code_lines[rng_start:rng_end+1]).rstrip()
            surrounding_context.append(code_snippet)
    
        merged_blocks.append({
            'defects': merged_defects,
            'block_ranges': merged_ranges,
            'surrounding_context': surrounding_context
        })

    return merged_blocks, code_lines

def main():
    rules_dict = load_rules()
    proj_dir = './mydefects/ets'
    detect_dir = './mydefects/ets/pages/defects'
    
    for file in sorted(os.listdir(detect_dir)):
        file_path = os.path.join(detect_dir, file)
        merged_blocks = get_single_file_surrounding_context(proj_dir, file_path, rules_dict)
        print(json.dumps(merged_blocks, indent=2))
        # for block in merged_blocks:
        #     print(json.dumps(block['defects'], indent=2))
        #     print(json.dumps(block['block_ranges'], indent=2))
        #     print(block['surrounding_context'])

if __name__ == "__main__":
    main()
