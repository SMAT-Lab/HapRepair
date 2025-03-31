import logging
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


def load_model_and_index():
    logging.getLogger().info("Loading model and index...")
    model_name = "dunzhang/stella_en_1.5B_v5"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    pc = Pinecone(api_key="xxxxxxxxxxxx")

    index_name = "arkts-1536"
    index = pc.Index(index_name)
    logging.getLogger().info("Model and index loaded")
    return model, tokenizer, index

def get_embedding(text, model, tokenizer):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1).squeeze().detach().numpy()
    return embeddings


def generate(query, system_prompt='You are a helpful AI assistant', base_url='https://api.openai.com', model='gpt-4o-2024-08-06', retries=3):
    def call_ollama(query):
        data = {
            "model": model,
            "prompt": query,
            "stream": False
        }
        response = requests.post('http://localhost:11434/api/generate', json=data)
        return response.json()['response']
        
    def get_api_key(base_url):
        if base_url == 'https://api.openai.com':
            return "xxxxxxxxxxx"
        
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
            logging.getLogger().error(f"API call failed attempt {attempt + 1}: {e}")
            if attempt + 1 == retries:
                raise

def sort_json_lines(data):
    if isinstance(data, list):
        return sorted(data, key=lambda x: x.get('line', 0))
    return data

def handle_vul_type_res(res):
    json_match = re.search(r'```json\n(.*?)\n```', res, flags=re.DOTALL)
    if json_match:
        cleaned_res = json_match.group(1)
    else:
        logging.getLogger().error("No JSON data block found")
        return None
    
    try:
        data = json.loads(cleaned_res)
        data = sort_json_lines(data)
        return data
    except json.JSONDecodeError as e:
        logging.getLogger().error(f"JSON parse error: {e}")
        return cleaned_res

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

def parse_ets_file(file_path):
    logging.getLogger().info(f"Parsing file: {file_path}")
    description = []
    code_example = ""
    in_comment = True

    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            stripped_line = line.strip()
            if in_comment and stripped_line.startswith('//'):
                description.append(stripped_line[2:].strip())
            else:
                in_comment = False
                code_example += stripped_line + " "

    return {
        'description': ' '.join(description),
        'defect_code_example': code_example.strip()
    }

def get_positive_example(file_path):
    logging.getLogger().info(f"Getting positive example: {file_path}")
    code_example = ""
    in_comment = False
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            stripped_line = line.strip()
            code_example += stripped_line + " "
    return code_example.strip()

def build_rules_dict(negative_dir, positive_dir):
    logging.getLogger().info("Building rules dictionary...")
    rules = {}
    for filename in os.listdir(negative_dir):
        if filename.endswith('.ets'):
            logging.getLogger().info(f"Processing rule file: {filename}")
            rule_name = os.path.splitext(filename)[0]
            rule_key = f"@performance/{rule_name}"
            negative_file_path = os.path.join(negative_dir, filename)
            rule_data = parse_ets_file(negative_file_path)

            positive_file_path = os.path.join(positive_dir, filename)
            if os.path.exists(positive_file_path):
                positive_example = get_positive_example(positive_file_path)
                rule_data['positive_code_example'] = positive_example
            else:
                logging.getLogger().warning(f"No corresponding positive example file found: {filename}")
                rule_data['positive_code_example'] = None

            rules[rule_key] = rule_data
    logging.getLogger().info("Rules dictionary built")
    return rules

def get_fix_prompt(repair_example, model, tokenizer, index):
    logging.getLogger().info(f"Getting fix prompt, rule: {repair_example['rule']}")
    query_text = repair_example["problem_code"]
    query_vector = get_embedding(query_text, model, tokenizer)
    
    results = index.query(
        namespace="arkts",
        vector=query_vector.tolist(),
        top_k=5,
        include_metadata=True,
        filter={"rule": repair_example["rule"]}
    )
    fix_prompt = "I will show you similar errors below. Please help me fix my code based on these error fixes.\n"
    matches = results.matches
    for j, match in enumerate(matches):
        metadata = match['metadata']
        print(metadata.keys())
        fix_prompt += (f"Demo {j+1}: \nRule Type: \n{metadata['rule']}\n\nDescription: \n{metadata['description']}\n\n"
                   f"Problem Code: \n```arkts\n{metadata['problem_code']}\n```\n\nFix Explanation: \n{metadata['problem_explain']}\n\n"
                   f"Fixed Code: \n\n```arkts\n{metadata['problem_fix']}\n```\n\n"
                )
    
    return fix_prompt

def repair_multiple_defects(defects, model, tokenizer, index):
    logging.getLogger().info(f"Starting to repair multiple defects, total: {len(defects)}")
    repair_suggestions = []
    
    for i, defect in enumerate(defects):
        if isinstance(defect, str):
            logging.getLogger().info(f"Processing defect {i+1}, skipping invalid data")
            continue

        logging.getLogger().info(f"Processing defect {i+1}, rule: {defect['rule']}")
        problem_code = ""
        for code_line in defect["defect_block"]['code']:
            problem_code += code_line + "\n"
            
        repair_example = {
            "rule": defect["rule"],
            "description": defect["analysis"],
            "problem_code": problem_code,
            "line_of_interest": defect["defect_block"]['line of interest']
        }
        
        fix_prompt = get_fix_prompt(repair_example, model, tokenizer, index)
        repair_suggestions.append({
            "defect": defect,
            "fix_prompt": fix_prompt,
            "problem_code": problem_code,
            "line_of_interest": defect["defect_block"]['line of interest'],
            "analysis": defect["analysis"]
        })
        
    logging.getLogger().info("Defect repair suggestions generated")
    return repair_suggestions

def handle_result(res):
    start = res.find('```json')
    if start == -1:
        logging.getLogger().error("No JSON data block found")
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
        logging.getLogger().error(f"JSON parse error: {e}")
        return None

def process_single_file(file_path, project_dir, df, system_prompt, model, tokenizer, index):
    print(file_path)
    project_name = project_dir.split('/')[-1]
    file_name = os.path.basename(file_path)
    log_dir = f"../log/{project_name}"
    os.makedirs(log_dir, exist_ok=True)
    
    file_handler = logging.FileHandler(f"{log_dir}/{file_name}.log")
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    
    logger = logging.getLogger(file_name)
    logger.setLevel(logging.INFO)
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    logger.addHandler(file_handler)
    logger.propagate = False
    
    logger.info(f"Reading file: {file_path}")
    with open(file_path, 'r') as f:
        code = f.read()

    logger.info("Reading defect info from Excel")
    
    results = []
    code_lines = code.split('\n')

    file_df = df[df['Source File'].str.endswith(os.path.basename(file_path))]

    for _, row in file_df.iterrows():
        line_num = int(row['Line'])
        result = {
            "rule": row['RuleName'],
            "line": line_num,
            "message": row['Detail'],
            "code": code_lines[line_num - 1].strip() if line_num <= len(code_lines) else ""
        }
        results.append(result)

    results.sort(key=lambda x: x['line'])

    unique_results = []
    seen = set()

    for result in results:
        key = (result['rule'], result['line'])
        if key not in seen:
            unique_results.append(result)
            seen.add(key)

    codelinter_res = json.dumps(unique_results, indent=2)
    logger.info("CodeLinter analysis results:")
    logger.info(codelinter_res)

    codelinter_results = json.loads(codelinter_res)

    vul_type_data = []
    defects = []

    for i, defect in enumerate(codelinter_results):
        logger.info(f"Analyzing defect {i+1}/{len(codelinter_results)}")
        prompt = f"""Following is my arkts code which you should check defects: 
    {code}

    Following is the CodeLinter analysis result:
    {json.dumps(defect, indent=2)}
    """ + """
    Now, please output the defect contained the defect code block in JSON format:
    ```json
    {
      "rule": "@performance/rule-name",
      "defect_block": {
        "code": ["Code block containing the defect"],
        "start_line": "first_line_number",
        "end_line": "last_line_number",
        "line of interest": "The code where the defect is reported by CodeLinter. Code instead of line number!!"
      },
      "analysis": "The description of the defect"
    }
    ```
                   """
        
        retry_count = 0
        max_retries = 3
        while isinstance(vul_type_data, str) and retry_count < max_retries:
            logger.warning(f"Invalid JSON format, retry {retry_count + 1}")
            res = generate(prompt, system_prompt=system_prompt)
            vul_type_data = handle_vul_type_res(res)
            retry_count += 1
            
        if isinstance(vul_type_data, list):
            vul_type_data = vul_type_data[0]
            
        if not isinstance(vul_type_data, str):
            defects.append(vul_type_data)
            

    logger.info("Generating repair suggestions")
    repair_prompts = repair_multiple_defects(defects, model, tokenizer, index)
    repair_results = []
    for i, result in enumerate(repair_prompts):
        logger.info(f"Processing repair suggestion {i+1}/{len(repair_prompts)}")
        fix_prompt = result['fix_prompt'] + f"\n待修复代码为：\n```arkts\n{code}\n```\n出现错误的代码行：\n```arkts\n{result['line_of_interest']}\n```\n其中的problem_code(错误的代码上下文)为：\n```arkts\n{result['problem_code']}\n```\n修复建议是：\n{result['analysis']}\n请根据上面所给出的缺陷修复建议,对上述待修复代码开始修复(不需要考虑其它问题，仅考虑建议中提出的), 修复的时候需要给出有问题的代码段以及修复的代码段：" + """
    请以如下json格式输出：
    ```json
    {
      "problem_code": "需要和传入的problem_code(包含错误的代码上下文)保持一致",
      "problem_fix": "修复后的代码段(包含修复的代码以及上下文),请保留换行和缩进"
    }
    ```
    """
        res = generate(fix_prompt, system_prompt="You are an arkts code repair expert. You will receive user-provided error code, problem type, error code line and context, and corresponding problem type repair cases. Please refer to the repair cases to help users fix the code based on the error code and problem type provided by the user.")
        logger.info(f"Repair suggestion {i+1} result:")
        logger.info(res)
        repair_results.append(res)

    logger.info("Processing repair results")
    for i, result in enumerate(repair_results):
        logger.info(f"Processing repair result {i+1}")
        data_json = handle_result(result)
        if data_json:
            logger.info(f"Original code:\n{data_json['problem_code']}")
            logger.info(f"Fixed code:\n{data_json['problem_fix']}")

    logger.info("Generating final fixed code")
    res = generate(f"Code:\n```arkts\n{code}\n```\n, Repair patches:\n{json.dumps(repair_results, indent=4)}\nPlease fix the code according to the patches above, only fix the parts in patches, keep other parts unchanged, and give the complete fixed file!", system_prompt="You are an arkts code repair expert. I will give you the defects in the code and the patches needed to fix these defects. Please help me execute these patches to get the complete fixed file code. Must be complete file!")
    logger.info("Final repair result:")
    logger.info(res)
    
    logger.removeHandler(file_handler)
    file_handler.close()

def main(project_dir=None, filename=None, base_dir=None):
    model, tokenizer, index = load_model_and_index()

    negative_directory = '../pages/negative'
    positive_directory = '../pages/positive'
    rules = build_rules_dict(negative_directory, positive_directory)

    system_prompt = """
    I am an expert at analyzing CodeLinter results and extracting minimal code blocks containing defects. When you provide me with ArkTS code and CodeLinter analysis results, I will:

    1. Learn the defect patterns from the rules
    2. Take the CodeLinter results as ground truth defect locations
    3. For each defect reported by CodeLinter:
       - Find the minimal code block containing the defect and its required context
       - Include any related code that is necessary to understand the defect
       - Ensure the extracted block captures the full scope of the issue
      

    Input format:
    1. Complete ArkTS source code
    2. CodeLinter results in JSON:
    [
      {
        "rule": "@performance/rule-name",
        "line": line_number, 
        "message": "Issue description",
        "code": "Code snippet"
      },
      {
        "rule": "@performance/rule-name",
        "line": line_number, 
        "message": "Issue description",
        "code": "Code snippet"
      }
    ]

    Important notes:
    - I will treat each CodeLinter result as definitive evidence of a defect
    - The defect_block will be the complete code unit containing the issue, for example, the define and use of a variable. 
    - You should find all the code related to the defect and return the defect_block. For example, for a constant-property-referencing-check-in-loops defect, you need to extract the constant variable assignment outside of the outermost loop
    - Line numbers will be preserved for accurate block replacement
    - Output will be valid JSON only, sorted by line number
    - Each block must be independently understandable and fixable

    I must output valid JSON only, sorted by line number. Make sure the output is correct JSON format which can be parsed by json.loads() and follow the format: 
    ```json
    [
      {
        "rule": "@performance/rule-name",
        "defect_block": {
          "code": ["Code block containing the defect"],
          "start_line": "first_line_number",
          "end_line": "last_line_number",
          "line of interest": "The code where the defect is reported by CodeLinter. Code instead of line number!!"
        },
        "analysis": "The description of the defect"
      },
      ...
    ]
    ```

    Following are the defect patterns I will look for:
    """

    for rule, details in rules.items():
        system_prompt += "Rule: {}\nDescription: {}\nDefect Example:\n{}\nFixed Example:\n{}\n\n".format(
            rule,
            details['description'],
            details['defect_code_example'],
            details['positive_code_example']
        )

    os.makedirs("../log", exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.FileHandler('../log/repair.log')]
    )

    if project_dir:
        result_files = glob.glob(os.path.join(project_dir, 'ets', 'result*.xlsx'))
        if not result_files:
            logging.getLogger().error(f"No result*.xlsx file found in {project_dir}/ets/")
            return
        
        df = pd.read_excel(result_files[0], header=1)
        
        files_to_process = df['Source File'].unique()
        
        for file_path in files_to_process:
            path_parts = file_path.split('\\')
            ets_index = -1
            for i, part in enumerate(path_parts):
                if part == 'ets':
                    ets_index = i
                    break
            
            if ets_index != -1:
                relative_path = '/'.join(path_parts[ets_index:])
            else:
                logging.getLogger().error(f"No ets directory found in path: {file_path}")
                continue
                
            project_file_path = os.path.join(project_dir, relative_path)
            
            if not os.path.exists(project_file_path):
                logging.getLogger().error(f"File does not exist: {project_file_path}")
                continue
                
            process_single_file(project_file_path, project_dir, df, system_prompt, model, tokenizer, index)
            
    elif filename and base_dir:
        file_path = os.path.join(base_dir, filename)
        if not os.path.exists(file_path):
            logging.getLogger().error(f"File does not exist: {file_path}")
            return
            
        df = pd.read_excel('../codelinter/result.xlsx', header=1)
        process_single_file(file_path, df, system_prompt, model, tokenizer, index)

if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--filename", help="filename")
    parser.add_argument("--base_dir", default="../arkts", help="base directory path") 
    parser.add_argument("--project_dir", required="--filename" not in sys.argv, help="project directory path, required when filename not specified")
    args = parser.parse_args()
    
    main(args.project_dir, args.filename, args.base_dir)

