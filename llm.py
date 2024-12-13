from openai import OpenAI
import os
from dotenv import load_dotenv, find_dotenv
import time
import logging

def get_openai_answer(ques, model_name="gpt-4o-mini", system_prompt=None, max_retries=5, retry_delay=1):
    _ = load_dotenv(find_dotenv())  # read local .env file
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
                logging.error(f"最大重试次数已达到")
                raise
            logging.warning(f"第{attempt + 1}次调用失败: {retry_delay}秒后重试...")
            time.sleep(retry_delay)

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
                logging.error(f"最大重试次数已达到,最终错误: {str(e)}")
                raise
            logging.warning(f"第{attempt + 1}次调用失败: {str(e)}, {retry_delay}秒后重试...")
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
                messages=messages
            )
            return response.choices[0].message.content
        except Exception as e:
            if attempt == max_retries - 1:
                logging.error(f"最大重试次数已达到,最终错误: {str(e)}")
                raise
            logging.warning(f"第{attempt + 1}次调用失败: {str(e)}, {retry_delay}秒后重试...")
            time.sleep(retry_delay)
