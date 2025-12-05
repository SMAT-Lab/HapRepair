import os
import uuid

import pandas as pd
from pinecone import Pinecone
from transformers import AutoModel, AutoTokenizer


def load_model_and_index() -> tuple[AutoModel, AutoTokenizer, object]:
    model_name = "dunzhang/stella_en_1.5B_v5"
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir="/home/models")
    model = AutoModel.from_pretrained(model_name, cache_dir="/home/models")

    api_key = os.getenv("PINECONE_API_KEY")
    if not api_key:
        raise RuntimeError("PINECONE_API_KEY is not set. Please export it before running.")

    pc = Pinecone(api_key=api_key)
    index_name = os.getenv("PINECONE_INDEX_NAME", "arkts-1536")
    index = pc.Index(index_name)
    return model, tokenizer, index

def read_defects_from_file(file_path: str) -> list[dict]:
    defects = []
    df = pd.read_excel(file_path, header=0)
    for index, row in df.iterrows():
        rule = row.get('Rule')
        description = row.get('Description')
        problem_code_example = row.get('Problem Code Example')
        problem_explanation = row.get('Problem Explanation')
        repair_code_example = row.get('Repair Code Example')
        diff = row.get('Diff')
        difflib = row.get('Difflib')

        defects.append(
            {
                'rule': rule,
                'description': description,
                'problem_code': problem_code_example,
                'problem_explain': problem_explanation,
                'problem_fix': repair_code_example,
                'gpt_diff': diff if pd.notna(diff) else None,
                'difflib': difflib if pd.notna(difflib) else None
            }
        )
    return defects


def get_embedding(text, model, tokenizer):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1).squeeze().detach().numpy()
    return embeddings

if __name__ == "__main__":
    excel_path = os.getenv("DEFECTS_XLSX", "./data/security_pairs.xlsx")
    try:
        model, tokenizer, index = load_model_and_index()
    except RuntimeError as err:
        print(err)
        raise SystemExit(1)

    defects = read_defects_from_file(excel_path)

    for defect in defects:
        vector_id = str(uuid.uuid4())
        metadata = {k: v for k, v in defect.items() if v is not None}
        values = get_embedding(defect["problem_code"], model, tokenizer)
        try:
            index.upsert(
                vectors=[{"id": vector_id, "values": values, "metadata": metadata}],
                namespace="arkts",
            )
        except Exception as e:
            print(f"error: {e}")
            continue
