import uuid
import pandas as pd
from transformers import AutoTokenizer, AutoModel
from pinecone import Pinecone

def load_model_and_index():
    model_name = "dunzhang/stella_en_1.5B_v5"
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir="/home/models")
    model = AutoModel.from_pretrained(model_name, cache_dir="/home/models")
    # 初始化 Pinecone
    pc = Pinecone(api_key="40075f49-8396-4571-924a-4b6d342cc81d")

    # 创建 Pinecone 索引
    index_name = "arkts-1536"
    # 连接到索引
    index = pc.Index(index_name)
    return model, tokenizer, index

def read_defects_from_file(file_path):
    ## xlsx 表头为规则	描述	问题代码样例	问题解释	修复代码样例
    defects = []
    # 直接传递文件路径给 pandas.read_excel，无需手动打开文件
    df = pd.read_excel(file_path, header=0)
    for index, row in df.iterrows():
        rule = row['规则']
        description = row['描述'] 
        problem_code_example = row['问题代码样例']
        problem_explanation = row['问题解释']
        repair_code_example = row['修复代码样例']
        diff = row['差异']

        defects.append({
            'rule': rule,
            'description': description,
            'problem_code': problem_code_example,
            'problem_explain': problem_explanation,
            'problem_fix': repair_code_example,
            'diff': diff
        })
    return defects

def get_embedding(text, model, tokenizer):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    outputs = model(**inputs)
    # 使用最后一个隐藏层的平均池化作为句子嵌入
    embeddings = outputs.last_hidden_state.mean(dim=1).squeeze().detach().numpy()
    return embeddings

if __name__ == "__main__":
    model, tokenizer, index = load_model_and_index()
    defects = read_defects_from_file('../data/demo.xlsx')
    
    for defect in defects:
        ## 随机生成id， defect作为metadata， defect["problem_code"]进行embedding
        id = str(uuid.uuid4())
        metadata = defect
        values = get_embedding(defect["problem_code"], model, tokenizer)
        index.upsert(vectors=[{"id": id, "values": values, "metadata": metadata}], namespace= "arkts")