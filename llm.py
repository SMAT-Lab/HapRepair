from openai import OpenAI
import os
import time
import logging
import re
from pathlib import Path
from typing import Any, Optional, Tuple


def _get_default_client() -> OpenAI:
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


def _is_packy_model(model_name: str) -> bool:
    # User-requested routing: gpt-5.1 uses Packy (Responses API).
    return model_name.startswith("gpt-5.1")


_PACKY_CLIENT_RE = re.compile(
    r'OpenAI\(\s*api_key="(?P<api_key>[^"]+)"\s*,\s*base_url="(?P<base_url>[^"]+)"\s*\)'
)


def _get_packy_credentials_from_file() -> Optional[Tuple[str, str]]:
    """
    Fallback: parse packy_api.py for api_key/base_url without importing it
    (importing would execute network requests).
    """
    packy_path = Path(__file__).resolve().parent / "packy_api.py"
    if not packy_path.is_file():
        return None
    text = packy_path.read_text(encoding="utf-8", errors="ignore")
    m = _PACKY_CLIENT_RE.search(text)
    if not m:
        return None
    return m.group("api_key"), m.group("base_url")


def _get_packy_client() -> OpenAI:
    api_key = os.getenv("PACKY_API_KEY")
    api_base = os.getenv("PACKY_API_BASE")
    if not api_key or not api_base:
        creds = _get_packy_credentials_from_file()
        if creds:
            api_key, api_base = creds
    if not api_key or not api_base:
        raise RuntimeError(
            "PACKY_API_KEY/PACKY_API_BASE not set and packy_api.py fallback not found. "
            "Set PACKY_API_KEY and PACKY_API_BASE for gpt-5.1 runs."
        )
    return OpenAI(api_key=api_key, base_url=api_base)


def get_openai_answer(
    ques,
    model_name="gpt-4o-mini",
    system_prompt=None,
    max_retries=5,
    retry_delay=1,
):
    def _extract_chat_content(resp: Any) -> str:
        choices = getattr(resp, "choices", None)
        if not choices:
            raise ValueError("LLM response has no choices")
        choice0 = choices[0]
        msg = getattr(choice0, "message", None)
        if msg is None:
            raise ValueError("LLM response choice[0] has no message")

        content = getattr(msg, "content", None)
        if content is None and isinstance(msg, dict):
            content = msg.get("content")
        if not isinstance(content, str) or not content.strip():
            finish_reason = getattr(choice0, "finish_reason", None)
            raise ValueError(f"LLM returned empty content (finish_reason={finish_reason})")
        return content

    if _is_packy_model(model_name):
        client = _get_packy_client()
        messages = [{"role": "user", "content": ques}]
        if system_prompt:
            messages.insert(0, {"role": "developer", "content": system_prompt})
        reasoning_effort = "high"
        if "low" in model_name:
            reasoning_effort = "low"
        elif "medium" in model_name:
            reasoning_effort = "medium"
        elif "high" in model_name:
            reasoning_effort = "high"
        reasoning_effort = os.getenv("PACKY_REASONING_EFFORT", reasoning_effort)
    else:
        client = _get_default_client()
        messages = [{"role": "user", "content": ques}]
        if system_prompt:
            messages.insert(0, {"role": "system", "content": system_prompt})

    for attempt in range(max_retries):
        try:
            if _is_packy_model(model_name):
                response = client.responses.create(
                    model=model_name,
                    input=messages,
                    reasoning={"effort": reasoning_effort},
                )
                text = getattr(response, "output_text", None)
                if not isinstance(text, str) or not text.strip():
                    raise ValueError("Responses API returned empty output_text")
                return text
            response = client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=0,
            )
            return _extract_chat_content(response)
        except Exception as exc:
            logging.warning(
                "Call %d failed (%s): %s",
                attempt + 1,
                type(exc).__name__,
                str(exc) or repr(exc),
            )
            if attempt == max_retries - 1:
                logging.error("Maximum retries reached when calling LLM")
                raise
            time.sleep(retry_delay)


def get_answer(ques, model_name, system_prompt=None, max_retries=10, retry_delay=2):
    return get_openai_answer(
        ques,
        model_name=model_name,
        system_prompt=system_prompt,
        max_retries=max_retries,
        retry_delay=retry_delay,
    )
