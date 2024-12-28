import logging
import multiprocessing
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

from llm import get_answer, get_deepseek_answer, get_openai_answer,get_ollama_answer

from get_prompt import generate_fix_prompt, combine_repair_results, get_rag_prompt, get_context_extraction_prompt, get_functionality_check_prompt,get_defect_extraction_prompt, judge_need_context_prompt
from output_handler import ArkTSDeclarationFixer, handle_vul_type_res, remove_difflib_line, fix_brackets, extract_code_from_markdown_block, check_functionality
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

def process_file(file, proj_dir, proj_repair_dir, logger, rules_dict, model, tokenizer, index, rag_type='difflib', top_n=1, surrounding_context=True, repair_model_name='gpt-4o-2024-08-06'):
    """处理单个文件的逻辑，独立为函数以便多线程调用"""
    #try:
    file_logs = []  # 用于暂存当前文件的所有日志

    # if 'AttributesSample.ets' not in file:
    #     return file_logs

    with open(file, 'r', encoding='utf-8') as f:
        try:
            code = f.read()
        except Exception as e:
            logger.info(f"文件 {file} 读取失败: {str(e)}")
            return file_logs

    file_logs.append(f"开始处理文件: {file}")
    # print(f"开始处理文件: {file}")
    
    merged_blocks, code_lines = get_single_file_surrounding_context(proj_dir, file, rules_dict)
    code = '\n'.join(code_lines)

    if len(merged_blocks) == 0:
        file_logs.append(f"文件 {file} 没有缺陷!")
        os.makedirs(os.path.join(proj_repair_dir, os.path.dirname(os.path.relpath(file, proj_dir))), exist_ok=True)
        with open(os.path.join(proj_repair_dir, os.path.relpath(file, proj_dir)), 'w', encoding='utf-8') as f:
            f.write(code)
        return file_logs
    
    max_attempts = 3
    attempt = 0
    fixed_code = code
    
    while attempt < max_attempts:
        repair_results = []
        for block in merged_blocks:
            defects = block['defects']
            contexts = block["surrounding_context"]
            sum_context = ""
            for i, text in enumerate(contexts):
                sum_context += f"...\n{text}\n"

            if not surrounding_context:
                sum_context = code

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

            valid_defects = []
            for defect in unique_defects:
                defect_rule = defect['rule']
                repair_example = dict()
                repair_example['rule'] = defect_rule
                repair_example['problem_code'] = sum_context
                rag_result = get_rag_prompt(repair_example, model, tokenizer, index, rag_type, number=top_n)
                if rag_result != "":
                    rag_prompt += rag_result
                    valid_defects.append(defect)
            
            for i, defect in enumerate(defects):
                if any(d['rule'] == defect['rule'] for d in valid_defects):
                    defect_description += f"Defect {i+1}:\n" + defect['message'] + '\n'
                    error_location += f"Defect {i+1}:\n" + defect['code'] + '\n'

            file_logs.append(f"待修复代码上下文: {sum_context}")
            file_logs.append('-' * 100)

            fix_prompt = generate_fix_prompt(rag_prompt, code, sum_context, defect_description, error_location)

            # res = get_openai_answer(fix_prompt, model_name='gpt-4o-2024-08-06')
            # print(fix_prompt)
            # res = get_ollama_answer(fix_prompt, model_name='qwen2.5-coder:32b')
            # res = get_deepseek_answer(fix_prompt)
            res = get_answer(fix_prompt, model_name=repair_model_name)
            # print(res)
            file_logs.append(f"当前修复块修复后的代码: {res}")
            repair_results.append((sum_context, res))
         
        final_fix_prompt = combine_repair_results(repair_results, code)
        # final_res = get_openai_answer(final_fix_prompt, model_name='gpt-4o-2024-08-06')
        # final_res = get_ollama_answer(final_fix_prompt, model_name='qwen2.5-coder:32b')
        file_logs.append(f"final_fix_prompt: \n{final_fix_prompt}")
        # final_res = get_deepseek_answer(final_fix_prompt)
        final_res = get_answer(final_fix_prompt, repair_model_name)
        file_logs.append(f"修复后的结果：\n{final_res}")
        fixed_code = extract_code_from_markdown_block(final_res)
         
        fixed_code = remove_difflib_line(fixed_code)

        fixer = ArkTSDeclarationFixer()
        result = fixer.validate_and_fix(fixed_code)
        if result:
            fixed_code = result.fixed_code

        res_json = check_functionality(code, fixed_code)
        # file_logs.append(f"功能性检查结果: {res_json}")
        # print(res_json)
        if res_json["result"] == "success":
            file_logs.append(f"功能性检查通过! ")
            break
        else:
            attempt += 1
            if attempt == max_attempts:
                file_logs.append(f"文件 {file} 经过{max_attempts}次尝试后功能性检查仍然失败!")
                fixed_code = code
            else:
                file_logs.append(f"文件 {file} 第{attempt}次修复后功能性检查失败,正在重试...")

    os.makedirs(os.path.join(proj_repair_dir, os.path.dirname(os.path.relpath(file, proj_dir))), exist_ok=True)
    with open(os.path.join(proj_repair_dir, os.path.relpath(file, proj_dir)), 'w', encoding='utf-8') as f:
        f.write(fixed_code)

    file_logs.append(f"文件 {file} 修复完成!")
    file_logs.append("-" * 100)
    return file_logs

    # except Exception as e:
    #     exc_type, exc_obj, exc_tb = sys.exc_info()
    #     fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
    #     return [f"文件 {file} 处理出错: {str(e)} (在文件 {fname} 第 {exc_tb.tb_lineno} 行)"]
## 从提取缺陷代码上下文
def RQ1():
    model, tokenizer, index = load_model_and_index()
    surrounding_context = True
    round_num = 4
    model_name = "qwen2.5-72b-instruct" # "qwen2.5-72b-instruct" # llama-3.2-90b-vision-instruct "llama3.3:70b-instruct-fp16"
    proj_dir = f"./{model_name}/round_{round_num}"
    # proj_repair_dir = f"./projects/{project_name + '_round' + str(round_num+1)}/ets"

    proj_repair_dir = f"./{model_name}/round_{round_num+1}"

    os.makedirs(f'./logs/{model_name}/round_{round_num+1}', exist_ok=True)
        
        # 创建状态记录文件夹
    status_dir = f'./status/{model_name}/round_{round_num+1}/'
    os.makedirs(status_dir, exist_ok=True)
        
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', handlers=[logging.FileHandler(f'./logs/{model_name}/round_{round_num+1}/RQ1_round_{round_num+1}.log', mode='a')])
    logger = logging.getLogger()
    
    if not os.path.exists(proj_dir):  
        logger.error(f"目录 {proj_dir} 不存在!")
        return
    
    files = glob.glob(os.path.join(proj_dir, '**', '*.ets'), recursive=True)
    files.extend(glob.glob(os.path.join(proj_dir, '**', '*.ts'), recursive=True))
    
    if len(files) == 0:
        logger.error(f"在 {proj_dir} 目录下没有找到任何 .ets 或 .ts 文件!")

    rules_dict = load_rules()

    # 过滤掉已处理的文件
    unprocessed_files = []
    for file in files:
        status_file = os.path.join(status_dir, os.path.relpath(file, proj_dir) + '.status')
        if not os.path.exists(status_file):
            unprocessed_files.append(file)

    # 使用多线程处理未处理的文件
    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
        futures = {executor.submit(process_file, file, proj_dir, proj_repair_dir, logger, rules_dict, model, tokenizer, index, repair_model_name=model_name): file for file in unprocessed_files}
        for future in concurrent.futures.as_completed(futures):
            file = futures[future]
            result = future.result()
            for log in result:
                logger.info(log)
            # 处理完成后创建状态文件
            status_file = os.path.join(status_dir, os.path.relpath(file, proj_dir) + '.status')
            os.makedirs(os.path.dirname(status_file), exist_ok=True)
            with open(status_file, 'w') as f:
                f.write('completed')

    # for file in unprocessed_files:
    #     logs = process_file(file, proj_dir, proj_repair_dir, logger, rules_dict, model, tokenizer, index, repair_model_name=model_name)
    #     for log in logs:
    #         logger.info(log)

    #     status_file = os.path.join(status_dir, os.path.relpath(file, proj_dir) + '.status')
    #     os.makedirs(os.path.dirname(status_file), exist_ok=True)
    #     with open(status_file, 'w') as f:
    #         f.write('completed')

if __name__ == "__main__":
    RQ1()
    # import argparse
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--filename", help="文件名")
    # parser.add_argument("--base_dir", default="../arkts", help="文件所在目录路径") 
    # parser.add_argument("--project_dir", required="--filename" not in sys.argv, help="项目目录路径,当未指定filename时必须提供")
    # args = parser.parse_args()
    
    # main(args.project_dir, args.filename, args.base_dir)
