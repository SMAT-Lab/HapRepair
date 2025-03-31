import pandas as pd
import os

def extract_code_examples():
    # 读取Excel文件
    df = pd.read_excel('./demo.xlsx')
    
    # 创建目录(如果不存在)
    os.makedirs('pairs/problem_code', exist_ok=True) 
    os.makedirs('pairs/repair_code', exist_ok=True)
    
    # 遍历每一行
    for index, row in df.iterrows():
        # 提取代码样例
        problem_code = row['问题代码样例']
        repair_code = row['修复代码样例']
        rule = row['规则'].split('/')[-1]
        
        # 生成文件名
        filename = f'{rule}_{index}.ets'
        
        # 保存问题代码
        problem_path = os.path.join('pairs/problem_code', filename)
        with open(problem_path, 'w', encoding='utf-8') as f:
            f.write(problem_code)
            
        # 保存修复代码
        repair_path = os.path.join('pairs/repair_code', filename)
        with open(repair_path, 'w', encoding='utf-8') as f:
            f.write(repair_code)

def update_excel():
    # 读取Excel文件
    df = pd.read_excel('./demo.xlsx')
    
    # 遍历每一行
    for index, row in df.iterrows():
        rule = row['规则'].split('/')[-1]
        filename = f'{rule}_{index}.ets'
        
        # 读取修改后的代码文件
        problem_path = os.path.join('pairs/problem_code', filename)
        repair_path = os.path.join('pairs/repair_code', filename)
        
        with open(problem_path, 'r', encoding='utf-8') as f:
            df.at[index, '问题代码样例'] = f.read()
            
        with open(repair_path, 'r', encoding='utf-8') as f:
            df.at[index, '修复代码样例'] = f.read()
    
    # 保存更新后的Excel
    df.to_excel('./vul_pairs.xlsx', index=False)
            
if __name__ == '__main__':
    # extract_code_examples()
    update_excel()