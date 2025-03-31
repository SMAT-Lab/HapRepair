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

from get_prompt import generate_fix_prompt, combine_repair_results, get_rag_prompt, get_context_extraction_prompt, get_functionality_check_prompt,get_defect_extraction_prompt, judge_need_context_prompt
from output_handler import ArkTSDeclarationFixer, handle_vul_type_res, remove_difflib_line, fix_brackets, extract_code_from_markdown_block, check_functionality
from get_surrounding_context import get_single_file_surrounding_context, load_rules
from code_repair import get_repair_prompt


def load_model_and_index():
    logging.getLogger().info("Loading model and index...")
    model_name = "dunzhang/stella_en_1.5B_v5"
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir="/home/models")
    model = AutoModel.from_pretrained(model_name, cache_dir="/home/models")
    
    pc = Pinecone(api_key="<PINECONE_API_KEY>")

    index_name = "arkts-1536"
    index = pc.Index(index_name)
    logging.getLogger().info("Model and index loaded")
    return model, tokenizer, index

def get_embedding(text, model, tokenizer):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1).squeeze().detach().numpy()
    return embeddings

def read_code():
    dir = "../defect"
    codes = {}
    logging.getLogger().info(f"Reading code files from {dir}...")
    for filename in sorted(os.listdir(dir)):
        if filename.endswith('.ets'):
            logging.getLogger().info(f"Processing file: {filename}")
            with open(os.path.join(dir, filename), 'r', encoding='utf-8') as file:
                code = file.read()
                codes[filename.split('.')[0]] = code
    logging.getLogger().info("Code files read complete")
    return codes

def RQ1():
    base_dir = "./defects_location_detect"

    system_prompt = get_defect_extraction_prompt()

    logging.basicConfig(
        level=logging.INFO,
        format='%(message)s',
        handlers=[logging.FileHandler('RQ1.log', mode='w')]
    )
    logger = logging.getLogger()

    for file in sorted(os.listdir(base_dir)):
        with open(os.path.join(base_dir, file), 'r') as f:
            code = f.read()

        res = get_openai_answer("Following is my arkts code which you should check whether there are defects: \n" + code, system_prompt=system_prompt, model_name='gpt-4o-mini')
        vul_type_data = handle_vul_type_res(res)
        
        logger.info(f"\nFile: {file}")
        logger.info(json.dumps(vul_type_data, indent=4))

def get_control_flow_graph(file):
    ## 调用arkanalyzer获取控制流图,目前直接存储到了projects/cfg目录下进行获取
    cfg_dir = f"./projects/cfg/{file}"
    with open(cfg_dir, 'r', encoding='utf-8') as f:
        cfg = f.read()
    return cfg

def process_file_RQ2(file, proj_dir, proj_repair_dir, logger, rules_dict, model, tokenizer, index, rag_type, top_n=5, surrounding_context=True):
    file_logs = []

    with open(file, 'r', encoding='utf-8') as f:
        try:
            code = f.read()
        except Exception as e:
            logger.info(f"Failed to read file {file}: {str(e)}")
            return file_logs

    file_logs.append(f"Processing file: {file}")
    
    merged_blocks, code_lines = get_single_file_surrounding_context(proj_dir, file, rules_dict)
    code = '\n'.join(code_lines)

    if len(merged_blocks) == 0:
        file_logs.append(f"No defects found in file {file}!")
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

        file_logs.append(f"Code context to fix: {sum_context}")
        file_logs.append('-' * 100)

        fix_prompt = generate_fix_prompt(rag_prompt, code, sum_context, defect_description, error_location)

        res = get_openai_answer(fix_prompt, system_prompt="You are a code repair expert. You will strictly follow the difflib format to return the fixed code", model_name='gpt-4o-2024-08-06')
        file_logs.append(f"Fixed code for current block: {res}")
        repair_results.append((sum_context, res))
     
    final_fix_prompt = combine_repair_results(repair_results, code)
    final_res = get_openai_answer(final_fix_prompt, model_name='gpt-4o-2024-08-06')
    fixed_code = extract_code_from_markdown_block(final_res)
    
    fixed_code = remove_difflib_line(fixed_code)

    fixer = ArkTSDeclarationFixer()
    result = fixer.validate_and_fix(fixed_code)
    if result:
        fixed_code = result.fixed_code

    original_cfg = get_control_flow_graph(code)
    repaired_cfg = get_control_flow_graph(fixed_code)

    res_json = check_functionality(code, fixed_code, original_cfg, repaired_cfg)

    if res_json["result"] == "success":
        file_logs.append(f"Functionality check passed!")
    else:
        file_logs.append(f"Functionality check failed for file {file}!")
        print(f"Functionality check failed after fixing file {file}!")
        fixed_code = code

    os.makedirs(os.path.join(proj_repair_dir, os.path.dirname(os.path.relpath(file, proj_dir))), exist_ok=True)
    with open(os.path.join(proj_repair_dir, os.path.relpath(file, proj_dir)), 'w', encoding='utf-8') as f:
        f.write(fixed_code)

    file_logs.append(f"File {file} repair completed!")
    file_logs.append("-" * 100)
    return file_logs

def RQ2():
    model, tokenizer, index = load_model_and_index()
    rag_types = ["gpt_diff", "no_diff"]
    project_name = "Photos"
    top_n = 1
    surrounding_context = True
    round_num = 0
    proj_dir = f"./projects/{project_name}/{'round_' + str(round_num) + '/difflib' if round_num > 0 else ''}/ets"
    
    for rag_type in rag_types:
        if surrounding_context == False:
            proj_repair_dir = f"./projects_full_context/{project_name}_repair/{rag_type}/ets"
        else:
            proj_repair_dir = f"./projects_{top_n}/{project_name}_repair/round_{round_num+1}/{rag_type}/ets"
        os.makedirs(f'./logs/{project_name}', exist_ok=True)
        
        if surrounding_context == False:
            status_dir = f'./status/projects_full_context/{project_name}/{rag_type}'
        else:
            status_dir = f'./status/projects_{top_n}/{project_name}/round_{round_num+1}/{rag_type}'
        os.makedirs(status_dir, exist_ok=True)
        
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', handlers=[logging.FileHandler(f'./logs/{project_name}/RQ2_{project_name}_round_{round_num}_top_{top_n}_{rag_type}_surrounding_context_{surrounding_context}.log', mode='w')])
        logger = logging.getLogger()

        print(f"Processing project: {project_name}, round: {round_num}, repair type: {rag_type}, top_n: {top_n}, use context: {surrounding_context}")
        
        if not os.path.exists(proj_dir):  
            logger.error(f"Directory {proj_dir} does not exist!")
            return
        
        files = glob.glob(os.path.join(proj_dir, '**', '*.ets'), recursive=True)
        files.extend(glob.glob(os.path.join(proj_dir, '**', '*.ts'), recursive=True))
        
        if len(files) == 0:
            logger.error(f"No .ets or .ts files found in {proj_dir}!")
            continue
    
        rules_dict = load_rules()

        unprocessed_files = []
        for file in files:
            status_file = os.path.join(status_dir, os.path.relpath(file, proj_dir) + '.status')
            if not os.path.exists(status_file):
                unprocessed_files.append(file)

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = {executor.submit(process_file_RQ2, file, proj_dir, proj_repair_dir, logger, rules_dict, model, tokenizer, index, rag_type, top_n, surrounding_context): file for file in unprocessed_files}
            for future in concurrent.futures.as_completed(futures):
                file = futures[future]
                result = future.result()
                for log in result:
                    logger.info(log)
                status_file = os.path.join(status_dir, os.path.relpath(file, proj_dir) + '.status')
                os.makedirs(os.path.dirname(status_file), exist_ok=True)
                with open(status_file, 'w') as f:
                    f.write('completed')

if __name__ == "__main__":
    RQ2()
