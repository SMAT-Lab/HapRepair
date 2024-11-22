import re


def analyze_defect_detection():
    # 读取RQ1.log文件
    with open('RQ1.log', 'r') as f:
        content = f.read()
    
    # 统计结果
    total_files = 0
    total_defects = 0
    false_positives = 0
    false_negatives = 0
    
    # 分割每个文件的结果
    file_results = content.split('\nFile: ')
    
    for result in file_results:
        if not result.strip():
            continue
            
        # 提取文件名
        lines = result.strip().split('\n')
        filename = lines[0].strip()
        total_files += 1
        
        try:
            if '[' in result:
                json_start = result.index('[')
                json_str = result[json_start:].strip()
                
                base_filename = re.sub(r'[-_]?\d+\.ets$', '.ets', filename)
                base_rule = base_filename[:-4]
                
                if '"rule":' in json_str:
                    rules = re.findall(r'@performance/([^"]+)', json_str)
                    found_correct = False
                    
                    # 遍历所有检测到的规则
                    for rule in rules:
                        total_defects += 1  # 每个检测到的规则都计入总缺陷数
                        if base_rule == rule:
                            found_correct = True
                        else:
                            false_positives += 1  # 不匹配的规则算误报
                    
                    # 如果没有找到正确的规则，算一个漏报
                    if not found_correct:
                        false_negatives += 1
                else:
                    false_negatives += 1
            else:
                false_negatives += 1
                
        except:
            false_negatives += 1
            continue
            
    # 打印统计结果
    print(f"\n缺陷检测分析结果:")
    print(f"检测的文件总数: {total_files}")
    print(f"检测出的缺陷总数: {total_defects}")
    print(f"误报数(False Positives): {false_positives}")
    print(f"漏报数(False Negatives): {false_negatives}")
    print(f"正确检测的缺陷数: {total_defects - false_positives}")
    
    return {
        "total_files": total_files,
        "total_defects": total_defects,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "true_positives": total_defects - false_positives
    }

analyze_defect_detection()
