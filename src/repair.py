import os
from transformers import AutoTokenizer, AutoModel
# 加载预训练的 BERT 模型和 tokenizer
import warnings
warnings.filterwarnings('ignore')

model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

from pinecone import Pinecone
import json
import torch
import pandas as pd

# 初始化 Pinecone
pc = Pinecone(api_key="40075f49-8396-4571-924a-4b6d342cc81d")

# 创建 Pinecone 索引
index_name = "arkts-defects"
dimension = 768  # BERT base 的输出维度是 768
# 连接到索引
index = pc.Index(index_name)

# 定义生成嵌入向量的函数
def get_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    outputs = model(**inputs)
    # 使用最后一个隐藏层的平均池化作为句子嵌入
    embeddings = outputs.last_hidden_state.mean(dim=1).squeeze().detach().numpy()
    return embeddings

from openai import OpenAI

def generate(message, model='arktsLLM') -> None:
    client = OpenAI(
        base_url='http://localhost:11434/v1/',
        api_key='ollama',
    )

    chat_completion = client.chat.completions.create(
        messages=
        [
            {
                'role': 'system',
                'content': '你是arkts代码修复专家。你将获得用户给出的错误代码以及问题类型，以及对应问题类型的修复案例。请参考修复案例，根据用户给出的错误代码以及问题类型，帮助用户修复代码。'
            },
            {
                'role': 'user',
                'content': message
            }
        ],
        model=model,
        temperature=0
    )

    return chat_completion.choices[0].message.content

# 查询示例
import pandas as pd

# 读取Excel文件，指定没有表头
df_input = pd.read_excel('../data/test.xlsx', header=None)

# 遍历每一行并格式化输出
test = []
for i, row in df_input.iterrows():
    if i == 0:
        continue
    row[0] = row[0].strip()

    item = {
        "rule": row[0],
        "description": row[1],
        "problem_code": row[2],
    }

    test.append(item)

import logging
import json
import time

# Configure logging
logging.basicConfig(filename='output.log', level=logging.INFO, format='%(message)s')

def generate_with_retry(prompt, max_retries=3):
    attempt = 0
    while attempt < max_retries:
        try:
            res = generate(prompt)
            return res
        except Exception as e:
            logging.error(f"Error during generation attempt {attempt + 1}: {e}")
            attempt += 1
            time.sleep(1)  # Optional: wait for a second before retrying
    raise Exception("Failed to generate after multiple attempts.")

for i in range(len(test)):
    query_text = json.dumps(test[i])

    query_vector = get_embedding(query_text)
    results = index.query(
        namespace="arkts",
        vector=query_vector.tolist(),
        top_k=10,
        include_metadata=True,
        filter={"rule": test[i]["rule"]}
    )

    prompt = "下面我将给出你类似的错误，请根据这些错误的修复方案，帮我修复一下我的代码。\n"
    matches = results.matches
    for j, match in enumerate(matches):
        metadata_text = match['metadata']['text']
        try:
            parsed_text = json.loads(metadata_text)
            prompt += (f"Demo {j+1}: \n问题类型规则: \n{parsed_text['rule']}\n\n问题描述: \n{parsed_text['description']}\n\n"
                       f"问题代码: \n{parsed_text['problem_code']}\n\n问题修复解释: \n{parsed_text['problem_explain']}\n\n"
                       f"修复代码: \n\n{parsed_text['problem_fix']}\n\n")
        except json.JSONDecodeError as e:
            logging.error(f"Error decoding JSON for ID {match['id']}: {e}")
            logging.error(f"Original metadata text: {metadata_text}")

    logging.info(prompt)
    prompt += (f"下面开始错误的修复！\n我有如下代码：\n{test[i]['problem_code']}\n\n对应的问题类型是: {test[i]['rule']}\n\n"
               f"该问题类型的描述如下:{test[i]['description']}\n\n请您帮我修复一下,输出包括问题修复解释以及修复代码\n")
    
    try:
        res = generate_with_retry(prompt)
        logging.info('---------')
        logging.info(test[i]['problem_code'])
        logging.info('---------')
        logging.info(res)
        logging.info('---------')
    except Exception as final_error:
        logging.error(f"Failed to generate output for the given prompt: {final_error}")
