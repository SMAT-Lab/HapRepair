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

def get_defects_from_file(file_path):
    """从文件中获取缺陷信息"""
    with open(file_path, 'r', encoding='utf-8') as f:
        code = f.read()
    code_lines = [''] + code.split('\n')
    
    result_files = glob.glob(os.path.join('./mydefects/ets', 'result*.xlsx'))
    file_df = pd.read_excel(result_files[0], header=1)
    file_df = file_df[file_df['Source File'].str.endswith(os.path.basename(file_path))]
    
    defects = []
    for _, row in file_df.iterrows():
        line_num = int(row['Line'])
        defect = {
            "rule": row['RuleName'],
            "line": line_num,
            "message": row['Detail'],
            "code": code_lines[line_num].strip() if line_num < len(code_lines) else ""
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
        surrounding_context = '\n'.join(content for _, _, content in code_blocks)
        
    return block_ranges, surrounding_context

def get_single_file_surrounding_context(file_path, rules_dict):
    """处理单个文件的缺陷检测"""
    print('-'*100)
    print(os.path.basename(file_path))
    
    code_lines, defects = get_defects_from_file(file_path)
    all_blocks = []
    
    for defect in defects:
        if defect['rule'] in rules_dict:
            block_ranges, surrounding_context = process_code_blocks(defect, code_lines, rules_dict)
            # 转换为JSON格式
            all_blocks.append({
                "defect": defect,
                "block_ranges": block_ranges,
                "surrounding_context": surrounding_context
            })

    # 合并重叠的blocks
    merged_blocks = []
    i = 0
    while i < len(all_blocks):
        current_block = all_blocks[i]
        current_defects = [current_block["defect"]]
        max_ranges = current_block["block_ranges"]
        max_context = current_block["surrounding_context"]
        
        j = i + 1
        while j < len(all_blocks):
            next_block = all_blocks[j]
            
            # 检查是否有重叠
            has_overlap = False
            for curr_start, curr_end in current_block["block_ranges"]:
                for next_start, next_end in next_block["block_ranges"]:
                    if (next_start <= curr_end and next_end >= curr_start):
                        has_overlap = True
                        break
                if has_overlap:
                    break
            
            if has_overlap:
                # 合并ranges,取最大范围
                all_ranges = list(current_block["block_ranges"]) + list(next_block["block_ranges"])
                max_ranges = []
                sorted_ranges = sorted(all_ranges, key=lambda x: x[0])
                current_start, current_end = sorted_ranges[0]
                
                for start, end in sorted_ranges[1:]:
                    if start <= current_end:
                        current_end = max(current_end, end)
                    else:
                        max_ranges.append((current_start, current_end))
                        current_start, current_end = start, end
                max_ranges.append((current_start, current_end))
                
                current_defects.append(next_block["defect"])
                max_context = next_block["surrounding_context"] if len(next_block["surrounding_context"]) > len(max_context) else max_context
                all_blocks.pop(j)
            else:
                j += 1
                
        merged_blocks.append({
            "defects": current_defects,
            "block_ranges": max_ranges,
            "surrounding_context": max_context
        })
        i += 1
        
    return merged_blocks

def main():
    rules_dict = load_rules()
    detect_dir = './mydefects/ets/pages/defects'
    
    for file in sorted(os.listdir(detect_dir)):
        file_path = os.path.join(detect_dir, file)
        merged_blocks = get_single_file_surrounding_context(file_path, rules_dict)
        print(json.dumps(merged_blocks, indent=2))
        # for block in merged_blocks:
        #     print(json.dumps(block['defects'], indent=2))
        #     print(json.dumps(block['block_ranges'], indent=2))
        #     print(block['surrounding_context'])

if __name__ == "__main__":
    main()
