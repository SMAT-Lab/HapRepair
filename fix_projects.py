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
    logging.getLogger().info("Loading model and index...")
    model_name = "dunzhang/stella_en_1.5B_v5"
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir="/home/models")
    model = AutoModel.from_pretrained(model_name, cache_dir="/home/models")
    
    pc = Pinecone(api_key="40075f49-8396-4571-924a-4b6d342cc81d")

    index_name = "arkts-1536"
    index = pc.Index(index_name)
    logging.getLogger().info("Model and index loaded")
    return model, tokenizer, index

def get_embedding(text, model, tokenizer):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1).squeeze().detach().numpy()
    return embeddings

def process_file(file, proj_dir, proj_repair_dir, logger, rules_dict, model, tokenizer, index, rag_type='difflib', top_n=1, surrounding_context=True, repair_model_name='gpt-4o-2024-08-06'):
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

            file_logs.append(f"Code context to fix: {sum_context}")
            file_logs.append('-' * 100)

            fix_prompt = generate_fix_prompt(rag_prompt, code, sum_context, defect_description, error_location)

            res = get_answer(fix_prompt, model_name=repair_model_name)
            file_logs.append(f"Fixed code for current block: {res}")
            repair_results.append((sum_context, res))
         
        final_fix_prompt = combine_repair_results(repair_results, code)
        file_logs.append(f"final_fix_prompt: \n{final_fix_prompt}")
        final_res = get_answer(final_fix_prompt, repair_model_name)
        file_logs.append(f"Final result: \n{final_res}")
        fixed_code = extract_code_from_markdown_block(final_res)
         
        fixed_code = remove_difflib_line(fixed_code)

        fixer = ArkTSDeclarationFixer()
        result = fixer.validate_and_fix(fixed_code)
        if result:
            fixed_code = result.fixed_code

        res_json = check_functionality(code, fixed_code)
        if res_json["result"] == "success":
            file_logs.append(f"Functionality check passed!")
            break
        else:
            attempt += 1
            if attempt == max_attempts:
                file_logs.append(f"File {file} failed functionality check after {max_attempts} attempts!")
                fixed_code = code
            else:
                file_logs.append(f"File {file} failed functionality check on attempt {attempt}, retrying...")

    os.makedirs(os.path.join(proj_repair_dir, os.path.dirname(os.path.relpath(file, proj_dir))), exist_ok=True)
    with open(os.path.join(proj_repair_dir, os.path.relpath(file, proj_dir)), 'w', encoding='utf-8') as f:
        f.write(fixed_code)

    file_logs.append(f"File {file} fix completed!")
    file_logs.append("-" * 100)
    return file_logs

def RQ1():
    model, tokenizer, index = load_model_and_index()
    surrounding_context = True
    round_num = 4
    model_name = "qwen2.5-72b-instruct"
    proj_dir = f"./{model_name}/round_{round_num}"
    proj_repair_dir = f"./{model_name}/round_{round_num+1}"

    os.makedirs(f'./logs/{model_name}/round_{round_num+1}', exist_ok=True)
        
    status_dir = f'./status/{model_name}/round_{round_num+1}/'
    os.makedirs(status_dir, exist_ok=True)
        
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', handlers=[logging.FileHandler(f'./logs/{model_name}/round_{round_num+1}/RQ1_round_{round_num+1}.log', mode='a')])
    logger = logging.getLogger()
    
    if not os.path.exists(proj_dir):  
        logger.error(f"Directory {proj_dir} does not exist!")
        return
    
    files = glob.glob(os.path.join(proj_dir, '**', '*.ets'), recursive=True)
    files.extend(glob.glob(os.path.join(proj_dir, '**', '*.ts'), recursive=True))
    
    if len(files) == 0:
        logger.error(f"No .ets or .ts files found in directory {proj_dir}!")

    rules_dict = load_rules()

    unprocessed_files = []
    for file in files:
        status_file = os.path.join(status_dir, os.path.relpath(file, proj_dir) + '.status')
        if not os.path.exists(status_file):
            unprocessed_files.append(file)

    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
        futures = {executor.submit(process_file, file, proj_dir, proj_repair_dir, logger, rules_dict, model, tokenizer, index, repair_model_name=model_name): file for file in unprocessed_files}
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
    RQ1()
