"""Classify one GP referral letter: specialty + urgency.

Talks to any OpenAI-compatible chat endpoint (Ollama, llama.cpp, vLLM and others).
Standard library only.

    LLM_BASE_URL   default http://127.0.0.1:11434/v1 (Ollama)
    LLM_MODEL      default gemma4:e4b
    LLM_API_KEY    optional bearer token
"""
import json
import os
import re
import sys
import urllib.request
from pathlib import Path

SPECIALTIES = ["cardiology", "orthopaedics", "neurology",
               "gastroenterology", "dermatology", "respiratory"]
URGENCIES = ["urgent", "semi-urgent", "routine"]

BASE_URL = os.environ.get("LLM_BASE_URL", "http://127.0.0.1:11434/v1")
MODEL = os.environ.get("LLM_MODEL", "gemma4:e4b")
API_KEY = os.environ.get("LLM_API_KEY", "")

PROMPTS = Path(__file__).parent / "prompts"


def chat(messages, thinking=False, timeout=600):
    body = {
        "model": MODEL,
        "messages": messages,
        "temperature": 0,
        "max_tokens": 4000 if thinking else 400,
        # Servers spell "reasoning off" differently: Ollama reads
        # reasoning_effort, llama.cpp-based servers read the template flag.
        "reasoning_effort": "medium" if thinking else "none",
        "chat_template_kwargs": {"enable_thinking": thinking},
    }
    req = urllib.request.Request(
        BASE_URL.rstrip("/") + "/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json",
                 **({"Authorization": f"Bearer {API_KEY}"} if API_KEY else {})},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        data = json.load(r)
    return data["choices"][0]["message"]["content"] or ""


def parse(reply):
    """Pull the JSON object out of the reply and check the labels."""
    reply = re.sub(r"<think>.*?</think>", "", reply, flags=re.S)
    m = re.search(r"\{.*\}", reply, flags=re.S)
    if not m:
        raise ValueError(f"no JSON in reply: {reply[:200]!r}")
    out = json.loads(m.group(0))
    spec = str(out.get("specialty", "")).strip().lower()
    urg = str(out.get("urgency", "")).strip().lower()
    if spec not in SPECIALTIES:
        raise ValueError(f"bad specialty {spec!r}")
    if urg not in URGENCIES:
        raise ValueError(f"bad urgency {urg!r}")
    conf = out.get("confidence")
    conf = float(conf) if isinstance(conf, (int, float)) else None
    return {"specialty": spec, "urgency": urg, "confidence": conf,
            "reason": str(out.get("reason", ""))}


def classify(letter, prompt="v1", thinking=False):
    system = (PROMPTS / f"{prompt}.txt").read_text()
    reply = chat([{"role": "system", "content": system},
                  {"role": "user", "content": letter}], thinking=thinking)
    return parse(reply)


if __name__ == "__main__":
    letter = sys.stdin.read()
    print(json.dumps(classify(letter), indent=2))
