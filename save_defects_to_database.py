import uuid
import pandas as pd
from transformers import AutoTokenizer, AutoModel
from pinecone import Pinecone

def load_model_and_index():
    model_name = "dunzhang/stella_en_1.5B_v5"
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir="/home/models")
    model = AutoModel.from_pretrained(model_name, cache_dir="/home/models")
    pc = Pinecone(api_key="<PINECONE_API_KEY>")

    index_name = "arkts-1536"
    index = pc.Index(index_name)
    return model, tokenizer, index

def read_defects_from_file(file_path):
    defects = []
    df = pd.read_excel(file_path, header=0)
    for index, row in df.iterrows():
        rule = row['Rule']
        description = row['Description']
        problem_code_example = row['Problem Code Example'] 
        problem_explanation = row['Problem Explanation']
        repair_code_example = row['Repair Code Example']
        diff = row['Diff'] if not pd.isna(row['Diff']) else None
        difflib = row['Difflib'] if not pd.isna(row['Difflib']) else None

        defects.append({
            'rule': rule,
            'description': description,
            'problem_code': problem_code_example,
            'problem_explain': problem_explanation,
            'problem_fix': repair_code_example,
            'gpt_diff': diff,
            'difflib': difflib
        })
    return defects

def get_embedding(text, model, tokenizer):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1).squeeze().detach().numpy()
    return embeddings

if __name__ == "__main__":
    model, tokenizer, index = load_model_and_index()
    defects = read_defects_from_file('./data/addition.xlsx')
    
    for defect in defects:
        id = str(uuid.uuid4())
        metadata = {k: v for k, v in defect.items() if v is not None}
        values = get_embedding(defect["problem_code"], model, tokenizer)
        try:
            index.upsert(vectors=[{"id": id, "values": values, "metadata": metadata}], namespace= "arkts")
        except Exception as e:
            print(f"error: {e}")
            continue