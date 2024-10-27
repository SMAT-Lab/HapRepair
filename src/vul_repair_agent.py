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

# Configure logging
logging.basicConfig(filename=f'../log/output_{time.time()}.log', level=logging.INFO, format='%(message)s')
model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)
# 初始化 Pinecone
pc = Pinecone(api_key="40075f49-8396-4571-924a-4b6d342cc81d")

# 创建 Pinecone 索引
index_name = "arkts-defects"
dimension = 768  # BERT base 的输出维度是 768
# 连接到索引
index = pc.Index(index_name)

def get_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    outputs = model(**inputs)
    # 使用最后一个隐藏层的平均池化作为句子嵌入
    embeddings = outputs.last_hidden_state.mean(dim=1).squeeze().detach().numpy()
    return embeddings

def generate(query, system_prompt='You are a helpful AI assistant', base_url='https://xiaoai.plus/v1', model='gpt-4o-2024-08-06', retries=3):

    if model == 'arktsLLM':
        
        for attempt in range(retries):
            try:
                data = {
                    "model": model,
                    "prompt": query,
                    "stream": False
                }
                response = requests.post('http://localhost:11434/api/generate', json=data)
                return response.json()['response']

            except Exception as e:
                logging.error(f"Attempt {attempt + 1} failed with error: {e}")
                if attempt + 1 == retries:
                    raise  # 如果达到最大重试次数，则抛出异常

    else: 
        client = OpenAI(
            base_url=base_url,
            api_key="***REMOVED***"
        )

        for attempt in range(retries):
            try:
                chat_completion = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": query}
                    ],
                )
                return chat_completion.choices[0].message.content
            except Exception as e:
                logging.error(f"Attempt {attempt + 1} failed with error: {e}")
                if attempt + 1 == retries:
                    raise  # 如果达到最大重试次数，则抛出异常

def parse_ets_file(file_path):
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
                code_example += stripped_line + " "  # 拼接成单行字符串

    return {
        'description': ' '.join(description),
        'code_example': code_example.strip()  # 去除末尾空白
    }

def build_rules_dict(directory):
    rules = {}
    for filename in os.listdir(directory):
        if filename.endswith('.ets'):
            rule_name = os.path.splitext(filename)[0]
            rule_key = f"@performance/{rule_name}"
            file_path = os.path.join(directory, filename)
            rule_data = parse_ets_file(file_path)
            rules[rule_key] = rule_data
    return rules


def get_system_prompt(rules):
    system_prompt = ""
    # 测试输出构造好的rules字典
    for rule, details in rules.items():
        system_prompt += f"Rule: {rule}\nDescription: {details['description']}\nCode Example: {details['code_example']}\n"

    system_prompt += """我将给出代码，请判断代码中存在上述哪种缺陷类型，给出的结果需要为数组中的一项: [@performance/constant-property-referencing-check-in-loops, @performance/foreach-args-check, @performance/high-frequency-log-check, @performance/hp-arkui-load-on-demand, @performance/hp-arkui-no-state-var-access-in-loop, @performance/hp-arkui-no-stringify-in-lazyforeach-key-generator, @performance/hp-arkui-use-reusable-component, @performance/lottie-animation-destroy-check, @performance/multiple-associations-state-var-check, @performance/no-high-loaded-frame-rate-range, @performance/number-init-check, @performance/sparse-array-check, @performance/timezone-interface-check, @performance/typed-array-check, @performance/waterflow-data-preload-check, @security/no-cycle]
请以json格式输出, 不要输出任何额外信息,若defect snippet中含有换行符，则转化为\\n输出。若存在多种缺陷，请选择最重要的缺陷进行输出，例如:
{
    "rule": "@performance/constant-property-referencing-check-in-loops",
    "description": "在循环如需频繁访问某个常量，且该属性引用常量在循环中不会改变，建议提取到循环外部，减少属性访问的次数",
    "line": 10,
    "defect snippet": "for (let i = 0; i < arr.length; i++) {\\n    console.log(arr[1]);\\n}",
}"""

    return system_prompt


def read_code():
    df_input = pd.read_excel('../data/test.xlsx', header=None)
    codes = []
    for i, row in df_input.iterrows():
        if (i == 0):
            continue
        codes.append(row[2])  # 假设代码在第三列
    return codes


def handle_vul_type_res(res):
    cleaned_res = re.sub(r'^```json\s*', '', res, flags=re.MULTILINE)
    cleaned_res = re.sub(r'\s*```$', '', cleaned_res, flags=re.MULTILINE)
    
    try:
        data = json.loads(cleaned_res)  # 解析清理后的 JSON 字符串
        return data
    except json.JSONDecodeError as e:
        print(f"解析 JSON 时出错: {e}")

def find_vul_code(data):
    rule = data['rule']
    code = data['problem_code']
    description = data['description']
    prompt_find_vul_code = f"""
    缺陷规则如下：
    {rule}
    缺陷定义如下：
    {description}
    缺陷例子如下：
    {rules[rule]['code_example']}
    请从我的缺陷代码中寻找出哪些代码存在上述缺陷类型的问题
    缺陷代码如下：
    {code}
    """

    res = generate(prompt_find_vul_code)

    return res

def get_fix_prompt(repair_example):
    query_text = json.dumps(repair_example)
    query_vector = get_embedding(query_text)
    
    results = index.query(
        namespace="arkts",
        vector=query_vector.tolist(),
        top_k=10,
        include_metadata=True,
        filter={"rule": repair_example["rule"]}
    )
    
    fix_prompt = "下面我将给出你类似的错误，请根据这些错误的修复方案，帮我修复一下我的代码。\n"
    matches = results.matches
    for j, match in enumerate(matches):
        metadata_text = match['metadata']['text']
        try:
            parsed_text = json.loads(metadata_text)
            fix_prompt += (f"Demo {j+1}: \n问题类型规则: \n{parsed_text['rule']}\n\n问题描述: \n{parsed_text['description']}\n\n"
                       f"问题代码: \n```arkts\n{parsed_text['problem_code']}\n```\n\n问题修复解释: \n{parsed_text['problem_explain']}\n\n"
                       f"修复代码: \n\n```arkts\n{parsed_text['problem_fix']}\n```\n\n")
        except json.JSONDecodeError as e:
            logging.error(f"Error decoding JSON for ID {match['id']}: {e}")
            logging.error(f"Original metadata text: {metadata_text}")
            
    # print(fix_prompt)
    
    return fix_prompt


if __name__ == '__main__':
    # 读取代码
    codes = read_code()
    # 读取寻找缺陷类型系统prompt

    # 设置要遍历的目录路径
    directory = '../pages'

    # 生成字典
    rules = build_rules_dict(directory)

    sys_prompt = get_system_prompt(rules)

    # 寻找缺陷类型
    for i, code in enumerate(codes):
        logging.info(f"开始处理demo: \n{code}")
        try:
            res = generate("下面是我的arkts代码: \n" + code, system_prompt=sys_prompt)
            logging.info("-"*50 + '\n' + res + '\n' + "-"*50)
            # 处理缺陷类型结果
            vul_type_data = handle_vul_type_res(res) 
            vul_type_data['problem_code'] = code
            # 根据缺陷类型寻找代码中的缺陷问题
            vul_summary = find_vul_code(vul_type_data)

            # RAG来获得修复代码的prompt
            fix_prompt = get_fix_prompt(vul_type_data)

            repair_suggestion = generate(fix_prompt, system_prompt="比较上述案例的问题代码以及修复代码和对应的问题修复解释，总结一下该问题的修复经验")

            # 生成修复代码
            fix_prompt += (f"下面开始错误的修复！\n\n\n待修复代码的对应的问题类型是: {vul_type_data['rule']}\n\n"
                           f"该问题类型的描述如下:{vul_type_data['description']}\n缺陷问题为: {vul_summary}\n缺陷修复建议为:\n{repair_suggestion}\n请您帮我修复一下,输出包括问题修复解释以及修复代码\n"
                           f"待修复代码为：\n```arkts\n{vul_type_data['problem_code']}\n```\n下面,请根据上面所给出的缺陷修复建议,对上述待修复代码开始修复(不需要考虑其它问题，仅考虑建议中提出的), 修复的时候需要给出有问题的代码段以及修复的代码段：")
            
            logging.info("-"*50 + '\n' + fix_prompt + '\n' + "-"*50 )
            
            res = generate(fix_prompt, system_prompt="你是arkts代码修复专家。你将获得用户给出的错误代码以及问题类型，以及对应问题类型的修复案例。请参考修复案例，根据用户给出的错误代码以及问题类型，帮助用户修复代码。", base_url='http://localhost:11434/v1', model='arktsLLM')
            logging.info("-"*50 + '\n'  + res + '\n' + "-"*50 )

        except Exception as e:
            logging.error(f"处理demo {i} 时出错: {e}")