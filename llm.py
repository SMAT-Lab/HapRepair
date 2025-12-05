from openai import OpenAI
import os
import time
import logging


def _get_client() -> OpenAI:
    """
    Construct an OpenAI client using API_KEY / API_BASE from the environment.

    IMPORTANT: We deliberately do NOT reload .env here. The expectation is that
    the shell script (e.g. run_haprepair_round_zhizengzeng.sh) has already
    exported API_KEY / API_BASE before invoking Python.
    """
    api_key = os.getenv("API_KEY")
    api_base = os.getenv("API_BASE")
    if not api_key or not api_base:
        raise RuntimeError(
            "API_KEY or API_BASE is not set in environment. "
            "Please run via the provided shell script which exports these "
            "variables, or export them manually before running."
        )
    return OpenAI(api_key=api_key, base_url=api_base)


def get_openai_answer(
    ques,
    model_name="gpt-4o-mini",
    system_prompt=None,
    max_retries=5,
    retry_delay=1,
):
    client = _get_client()

    messages = [{"role": "user", "content": ques}]
    if system_prompt:
        messages.insert(0, {"role": "system", "content": system_prompt})

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=0,
            )
            return response.choices[0].message.content
        except Exception:
            if attempt == max_retries - 1:
                logging.error("Maximum retries reached when calling LLM")
                raise
            logging.warning(
                "Call %d failed: retrying in %s seconds...",
                attempt + 1,
                retry_delay,
            )
            time.sleep(retry_delay)


def get_answer(ques, model_name, system_prompt=None, max_retries=10, retry_delay=2):
    return get_openai_answer(
        ques,
        model_name=model_name,
        system_prompt=system_prompt,
        max_retries=max_retries,
        retry_delay=retry_delay,
    )
