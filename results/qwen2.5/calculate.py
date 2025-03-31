import pandas as pd
import os

def extract_project_name(path, round_num):
    # 转换为标准路径格式
    path = path.replace('\\', '/')
    # 尝试两种可能的路径格式
    try:
        if round_num == 0:
            if 'origin/' in path:
                return path.split('origin/')[1].split('/')[0]
        if f'round_{round_num}_reject/' in path:
            return path.split(f'round_{round_num}_reject/')[1].split('/')[0]
        elif f'round_{round_num}/' in path:
            return path.split(f'round_{round_num}/')[1].split('/')[0]
        else:
            print(f"Unexpected path format: {path}")
            return None
    except Exception as e:
        print(f"Error processing path: {path}")
        print(f"Error: {str(e)}")
        return None

def analyze_excel_files():
    # 存储所有轮次的数据
    all_rounds_project = {}  # 按项目统计
    all_rounds_rules = {}    # 按规则统计
    
    # 处理round 0-5的文件
    for round_num in range(6):
        file_name = f'round_{round_num}.xlsx'
        if not os.path.exists(file_name):
            print(f"File {file_name} not found")
            continue
            
        df = pd.read_excel(file_name, skiprows=1)
        
        # 1. 按项目统计
        project_counts = df['Source File'].apply(lambda x: extract_project_name(x, round_num)).value_counts().astype(int)
        all_rounds_project[f'Round {round_num}'] = project_counts
        
        # 2. 按规则统计
        rule_counts = df['RuleName'].value_counts().astype(int)
        all_rounds_rules[f'Round {round_num}'] = rule_counts
        
        # 打印验证总数
        print(f"Round {round_num} total records: {len(df)}")
        
    # 转换为DataFrame便于查看
    projects_df = pd.DataFrame(all_rounds_project)
    rules_df = pd.DataFrame(all_rounds_rules)
    
    # 填充NaN为0并转为整数
    projects_df = projects_df.fillna(0).astype(int)
    rules_df = rules_df.fillna(0).astype(int)
    
    # 添加总和行
    projects_df.loc['Total'] = projects_df.sum()
    rules_df.loc['Total'] = rules_df.sum()
    
    # 保存结果
    projects_df.to_excel('projects_statistics.xlsx')
    rules_df.to_excel('rules_statistics.xlsx')
    
    # 打印验证
    print("\nProjects statistics:")
    print(projects_df)
    print("\nRules statistics:")
    print(rules_df)

if __name__ == "__main__":
    analyze_excel_files()