from openai import OpenAI
import os
from dotenv import load_dotenv, find_dotenv


def get_openai_answer(ques, model_name="gpt-4o-mini", system_prompt=None):
    _ = load_dotenv(find_dotenv())  # read local .env file
    api_key = os.environ['OPENAI_API_KEY']
    api_base = os.environ['OPENAI_API_BASE']

    client = OpenAI(api_key=api_key, base_url=api_base)
    
    messages = [{"role": "user", "content": ques}]

    if system_prompt:
        messages.insert(0, {"role": "system", "content": system_prompt})

    response = client.chat.completions.create(
        model=model_name,
        messages=messages
    )
    return response.choices[0].message.content

def get_ollama_answer(ques, model_name, system_prompt=None):
    _ = load_dotenv(find_dotenv())
    api_key = os.environ['OLLAMA_API_KEY']
    api_base = os.environ["OLLAMA_API_BASE"]

    client = OpenAI(api_key=api_key, base_url=api_base)

    
    messages = [{"role": "user", "content": ques}]

    if system_prompt:
        messages.insert(0, {"role": "system", "content": system_prompt})

    response = client.chat.completions.create(
        model=model_name,
        messages=messages
    )
    return response.choices[0].message.content

def get_gptgod_answer(ques, model_name, system_prompt=None):
    _ = load_dotenv(find_dotenv())
    api_key = os.environ['GPTGOD_API_KEY']
    api_base = os.environ["GPTGOD_API_BASE"]

    client = OpenAI(api_key=api_key, base_url=api_base)

    
    messages = [{"role": "user", "content": ques}]

    if system_prompt:
        messages.insert(0, {"role": "system", "content": system_prompt})

    response = client.chat.completions.create(
        model=model_name,
        messages=messages
    )
    return response.choices[0].message.content
