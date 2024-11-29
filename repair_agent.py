import logging
import queue
import sys
import requests
import json
import re
import os
from openai import OpenAI

import json
import pandas as pd

from openai import OpenAI
import re
import json
import logging
from transformers import AutoTokenizer, AutoModel
from pinecone import Pinecone
import pandas as pd
import os
import requests
import time
import glob
import concurrent.futures

from llm import get_openai_answer,get_ollama_answer

from get_prompt import generate_fix_prompt, combine_repair_results, get_rag_prompt, get_context_extraction_prompt, get_defect_extraction_prompt, judge_need_context_prompt

from get_surrounding_context import get_single_file_surrounding_context, load_rules
from code_repair import get_repair_prompt


def load_model_and_index():
    logging.getLogger().info("开始加载模型和索引...")
    model_name = "dunzhang/stella_en_1.5B_v5"
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir="/home/models")
    model = AutoModel.from_pretrained(model_name, cache_dir="/home/models")
    # 初始化 Pinecone
    pc = Pinecone(api_key="40075f49-8396-4571-924a-4b6d342cc81d")

    # 创建 Pinecone 索引
    index_name = "arkts-1536"
    # 连接到索引
    index = pc.Index(index_name)
    logging.getLogger().info("模型和索引加载完成")
    return model, tokenizer, index

def get_embedding(text, model, tokenizer):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    outputs = model(**inputs)
    # 使用最后一个隐藏层的平均池化作为句子嵌入
    embeddings = outputs.last_hidden_state.mean(dim=1).squeeze().detach().numpy()
    return embeddings


def generate(query, system_prompt='You are a helpful AI assistant', base_url='https://xiaoai.plus/v1', model='gpt-4o-2024-08-06', retries=3):
    def call_ollama(query):
        data = {
            "model": model,
            "prompt": query,
            "stream": False
        }
        response = requests.post('http://localhost:11434/api/generate', json=data)
        return response.json()['response']
        
    def get_api_key(base_url):
        if base_url == 'https://xiaoai.plus/v1':
            # return "***REMOVED***"
            return "***REMOVED***"
        return "***REMOVED***"
        
    def create_chat_completion(client, query, system_prompt=None):
        messages = query if isinstance(query, list) else (
            [{"role": "user", "content": query}] if 'o1-mini' in model
            else [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query}
            ]
        )
        return client.chat.completions.create(
            model=model,
            messages=messages
        )

    for attempt in range(retries):
        try:
            if model == 'arktsLLM':
                return call_ollama(query)
                
            client = OpenAI(
                base_url=base_url,
                api_key=get_api_key(base_url)
            )
            chat_completion = create_chat_completion(client, query, system_prompt)
            return chat_completion.choices[0].message.content
            
        except Exception as e:
            logging.getLogger().error(f"API调用第{attempt + 1}次失败: {e}")
            if attempt + 1 == retries:
                raise

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

def read_code():
    dir = "../defect"
    codes = {}
    logging.getLogger().info(f"开始读取{dir}目录下的代码文件...")
    for filename in sorted(os.listdir(dir)):
        if filename.endswith('.ets'):
            logging.getLogger().info(f"正在处理文件: {filename}")
            with open(os.path.join(dir, filename), 'r', encoding='utf-8') as file:
                code = file.read()
                codes[filename.split('.')[0]] = code
    logging.getLogger().info("代码文件读取完成")
    return codes

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
    code_block = re.search(r'```(?:arkts|javascript|js|ts|typescript)\n(.*)\n```', markdown_block, re.DOTALL).group(1)
    return code_block

## few-shot learning to fault localization
def RQ1():
    base_dir = "./defects_location_detect"

    system_prompt = get_defect_extraction_prompt()

    # 配置日志记录器
    logging.basicConfig(
        level=logging.INFO,
        format='%(message)s',
        handlers=[logging.FileHandler('RQ1.log', mode='w')]
    )
    logger = logging.getLogger()

    for file in sorted(os.listdir(base_dir)):
        with open(os.path.join(base_dir, file), 'r') as f:
            code = f.read()

        res = generate("Following is my arkts code which you should check whether there are defects: \n" + code, system_prompt=system_prompt)
        vul_type_data = handle_vul_type_res(res)
        
        # 将文件名和结果写入日志
        logger.info(f"\nFile: {file}")
        logger.info(json.dumps(vul_type_data, indent=4))

def process_file_RQ2(file, proj_dir, proj_repair_dir, logger, rules_dict, model, tokenizer, index):
    """处理单个文件的逻辑，独立为函数以便多线程调用"""
    try:

        file_logs = []  # 用于暂存当前文件的所有日志

        with open(file, 'r', encoding='utf-8') as f:
            code = f.read()

        file_logs.append(f"开始处理文件: {file}")
        
        merged_blocks = get_single_file_surrounding_context(proj_dir, file, rules_dict)

        if len(merged_blocks) == 0:
            file_logs.append(f"文件 {file} 没有缺陷!")
            os.makedirs(os.path.join(proj_repair_dir, os.path.dirname(os.path.relpath(file, proj_dir))), exist_ok=True)
            with open(os.path.join(proj_repair_dir, os.path.relpath(file, proj_dir)), 'w', encoding='utf-8') as f:
                f.write(code)
            return file_logs
        
        repair_results = []
        for block in merged_blocks:
            defects = block['defects']
            contexts = block["surrounding_context"]
            sum_context = ""
            for i, text in enumerate(contexts):
                sum_context += f"Context {i+1}:\n{text}\n"

            rag_prompt = ""
            file_logs.append(defects)
            unique_defects = []
            seen_rules = set()
            for defect in defects:
                if defect['rule'] not in seen_rules:
                    unique_defects.append(defect)
                    seen_rules.add(defect['rule'])

            defect_description = "The code snippet containing the defect is as follows:\n"
            error_location = "The location of the defect is as follows:\n"

            for defect in unique_defects:
                defect_rule = defect['rule']
                repair_example = dict()
                repair_example['rule'] = defect_rule
                repair_example['problem_code'] = sum_context
                rag_prompt += get_rag_prompt(repair_example, model, tokenizer, index, number=5)
            
            for i, defect in enumerate(defects):
                defect_description += f"Defect {i+1}:\n" + defect['message'] + '\n'
                error_location += f"Defect {i+1}:\n" + defect['code'] + '\n'

            file_logs.append(f"RAG提示: {rag_prompt}")

            fix_prompt = generate_fix_prompt(rag_prompt, code, sum_context, defect_description, error_location)
            file_logs.append(f"修复提示: {fix_prompt}")
            file_logs.append('-' * 100)

            res = get_openai_answer(fix_prompt, model_name='gpt-4o-mini')
            file_logs.append(f"当前修复块修复后的代码: {res}")
            repair_results.append((sum_context, res))
        
        final_fix_prompt = combine_repair_results(repair_results, code)
        final_res = get_openai_answer(final_fix_prompt, model_name='gpt-4o-mini')
        final_code = extract_code_from_markdown_block(final_res)
        os.makedirs(os.path.join(proj_repair_dir, os.path.dirname(os.path.relpath(file, proj_dir))), exist_ok=True)
        with open(os.path.join(proj_repair_dir, os.path.relpath(file, proj_dir)), 'w', encoding='utf-8') as f:
            f.write(final_code)

        file_logs.append(f"文件 {file} 修复完成! 修复结果:\n{final_res}")
        file_logs.append("-" * 100)
        return file_logs

    except Exception as e:
        return [f"文件 {file} 处理出错: {e}"]
    
## 从提取缺陷代码上下文
def RQ2():
    model, tokenizer, index = load_model_and_index()
    project_name = "wifi_testapp"
    round_num = 5
    proj_dir = f"./{project_name + ('_round' + str(round_num) if round_num > 0 else '')}/ets"
    proj_repair_dir = f"./{project_name + '_round' + str(round_num+1)}/ets"
    logging.basicConfig(level=logging.INFO, format='%(message)s', handlers=[logging.FileHandler(f'RQ2_{project_name}.log', mode='w')])
    logger = logging.getLogger()
    
    if not os.path.exists(proj_dir):
        logger.error(f"目录 {proj_dir} 不存在!")
        return
        
    files = glob.glob(os.path.join(proj_dir, '**', '*.ets'), recursive=True)
    files.extend(glob.glob(os.path.join(proj_dir, '**', '*.ts'), recursive=True))
    
    if len(files) == 0:
        logger.error(f"在 {proj_dir} 目录下没有找到任何 .ets 或 .ts 文件!")
        return
    
    rules_dict = load_rules()

    # 使用多线程处理文件
    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
        futures = {executor.submit(process_file_RQ2, file, proj_dir, proj_repair_dir, logger, rules_dict, model, tokenizer, index): file for file in files}
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            for log in result:
                logger.info(log)

if __name__ == "__main__":
    RQ2()
    # import argparse
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--filename", help="文件名")
    # parser.add_argument("--base_dir", default="../arkts", help="文件所在目录路径") 
    # parser.add_argument("--project_dir", required="--filename" not in sys.argv, help="项目目录路径,当未指定filename时必须提供")
    # args = parser.parse_args()
    
    # main(args.project_dir, args.filename, args.base_dir)
