from openai import OpenAI
import os
from dotenv import load_dotenv, find_dotenv
import time
import logging

def get_openai_answer(ques, model_name="gpt-4o-mini", system_prompt=None, max_retries=5, retry_delay=1):
    _ = load_dotenv(find_dotenv())
    api_key = os.environ['OPENAI_API_KEY']
    api_base = os.environ['OPENAI_API_BASE']

    client = OpenAI(api_key=api_key, base_url=api_base)
    
    messages = [{"role": "user", "content": ques}]

    if system_prompt:
        messages.insert(0, {"role": "system", "content": system_prompt})

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=0
            )
            return response.choices[0].message.content
        except Exception as e:
            if attempt == max_retries - 1:
                logging.error(f"Maximum retries reached")
                raise
            logging.warning(f"Call {attempt + 1} failed: retrying in {retry_delay} seconds...")
            time.sleep(retry_delay)

def get_deepseek_answer(ques, model_name="deepseek-chat", system_prompt=None, max_retries=3, retry_delay=1):
    _ = load_dotenv(find_dotenv())
    api_key = os.environ['DEEPSEEK_API_KEY']
    api_base = os.environ["DEEPSEEK_API_BASE"]

    client = OpenAI(api_key=api_key, base_url=api_base)

    if isinstance(ques, list):
        messages = ques
    else:
        messages = [{"role": "user", "content": ques}]

    if system_prompt:
        messages.insert(0, {"role": "system", "content": system_prompt})

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=messages
            )
            return response.choices[0].message.content
        except Exception as e:
            if attempt == max_retries - 1:
                logging.error(f"Maximum retries reached, final error: {str(e)}")
                raise   
            logging.warning(f"Call {attempt + 1} failed: {str(e)}, retrying in {retry_delay} seconds...")
            time.sleep(retry_delay)

    return None

def get_ollama_answer(ques, model_name, system_prompt=None, max_retries=3, retry_delay=1):
    _ = load_dotenv(find_dotenv())
    api_key = os.environ['OLLAMA_API_KEY']
    api_base = os.environ["OLLAMA_API_BASE"]

    client = OpenAI(api_key=api_key, base_url=api_base)

    messages = [{"role": "user", "content": ques}]

    if system_prompt:
        messages.insert(0, {"role": "system", "content": system_prompt})

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=messages
            )
            return response.choices[0].message.content
        except Exception as e:
            if attempt == max_retries - 1:
                logging.error(f"Maximum retries reached, final error: {str(e)}")
                raise
            logging.warning(f"Call {attempt + 1} failed: {str(e)}, retrying in {retry_delay} seconds...")
            time.sleep(retry_delay)

def get_gptgod_answer(ques, model_name, system_prompt=None, max_retries=3, retry_delay=1):
    _ = load_dotenv(find_dotenv())
    api_key = os.environ['GPTGOD_API_KEY']
    api_base = os.environ["GPTGOD_API_BASE"]

    client = OpenAI(api_key=api_key, base_url=api_base)

    messages = [{"role": "user", "content": ques}]

    if system_prompt:
        messages.insert(0, {"role": "system", "content": system_prompt})

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=messages,
                stream=True
            )
            full_content = ""
            for chunk in response:
                if chunk.choices[0].delta.content:
                    full_content += chunk.choices[0].delta.content
            return full_content
        except Exception as e:
            if attempt == max_retries - 1:
                logging.error(f"Maximum retries reached, final error: {str(e)}")
                raise
            logging.warning(f"Call {attempt + 1} failed: {str(e)}, retrying in {retry_delay} seconds...")
            time.sleep(retry_delay)
            retry_delay = retry_delay * 2

def get_qwen_answer(ques, model_name, system_prompt=None, max_retries=10, retry_delay=2):
    _ = load_dotenv(find_dotenv())
    api_key = os.environ['QWEN_API_KEY']
    api_base = os.environ["QWEN_API_BASE"]

    client = OpenAI(api_key=api_key, base_url=api_base)
    messages = [{"role": "user", "content": ques}]

    if system_prompt:
        messages.insert(0, {"role": "system", "content": system_prompt})

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=messages,
                stream=True
            )
            full_content = ""
            for chunk in response:
                if chunk.choices[0].delta.content:
                    full_content += chunk.choices[0].delta.content
            return full_content
        except Exception as e:
            if attempt == max_retries - 1:
                logging.error(f"Maximum retries reached, final error: {str(e)}")
                raise
            logging.warning(f"Call {attempt + 1} failed: {str(e)}, retrying in {retry_delay} seconds...")
            time.sleep(retry_delay)
            retry_delay = retry_delay * 2

def get_llama_api_answer(ques, model_name, system_prompt=None, max_retries=10, retry_delay=2):
    _ = load_dotenv(find_dotenv())
    api_key = os.environ['LLAMA_API_KEY']
    api_base = os.environ["LLAMA_API_BASE"]

    client = OpenAI(api_key=api_key, base_url=api_base)

    messages = [{"role": "user", "content": ques}]

    if system_prompt:
        messages.insert(0, {"role": "system", "content": system_prompt})

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=messages,
                stream=True
            )
            full_content = ""
            for chunk in response:
                if chunk.choices[0].delta.content:
                    full_content += chunk.choices[0].delta.content
            return full_content
        except Exception as e:
            if attempt == max_retries - 1:
                logging.error(f"Maximum retries reached, final error: {str(e)}")
                raise
            logging.warning(f"Call {attempt + 1} failed: {str(e)}, retrying in {retry_delay} seconds...")
            time.sleep(retry_delay)
            retry_delay = retry_delay * 2

def get_answer(ques, model_name, system_prompt=None, max_retries=10, retry_delay=2):
    if 'gpt' in model_name:
        return get_openai_answer(ques, model_name, system_prompt=None, max_retries=max_retries, retry_delay=retry_delay)
    elif 'deepseek' in model_name:
        return get_deepseek_answer(ques, model_name, system_prompt=None, max_retries=max_retries, retry_delay=retry_delay)
    elif model_name in ['qwen2.5-coder:32b-base-fp16', 'qwen2.5:72b-instruct-fp16', 'llama3.3:70b-instruct-fp16']:
        return get_ollama_answer(ques, model_name, system_prompt=None, max_retries=max_retries, retry_delay=retry_delay)
    elif model_name =='qwen2.5-72b-instruct':
        return get_qwen_answer(ques, model_name, system_prompt=None, max_retries=max_retries, retry_delay=retry_delay)
    elif model_name in ['llama3.3-70b-instruct', 'llama3.3-70b']:
        return get_llama_api_answer(ques, model_name, system_prompt=None, max_retries=max_retries, retry_delay=retry_delay)
    else:
        return get_gptgod_answer(ques, model_name, system_prompt=None, max_retries=max_retries, retry_delay=retry_delay)