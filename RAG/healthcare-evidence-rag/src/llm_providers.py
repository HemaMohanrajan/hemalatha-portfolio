import os
import requests
from dataclasses import dataclass
from typing import Protocol

@dataclass(frozen=True)
class LLMResponse:
    text: str
    tokens_in: int
    tokens_out: int

class LLM(Protocol):
    def generate(self, prompt: str, temperature: float = 0.2) -> LLMResponse: ...

class OpenAILLM:
    def __init__(self, api_key: str, model: str):
        self.api_key = api_key
        self.model = model

    def generate(self, prompt: str, temperature: float = 0.2) -> LLMResponse:
        # Uses Responses API style via HTTPS without extra SDK dependency
        import json
        import urllib.request

        url = "https://api.openai.com/v1/responses"
        payload = {
            "model": self.model,
            "input": prompt,
            "temperature": temperature,
        }
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(url, data=data, method="POST")
        req.add_header("Content-Type", "application/json")
        req.add_header("Authorization", f"Bearer {self.api_key}")

        with urllib.request.urlopen(req, timeout=60) as resp:
            out = json.loads(resp.read().decode("utf-8"))

        # Best-effort parsing
        text = ""
        if "output" in out and out["output"]:
            # concat text segments
            for item in out["output"]:
                for c in item.get("content", []):
                    if c.get("type") == "output_text":
                        text += c.get("text", "")
        usage = out.get("usage", {}) or {}
        return LLMResponse(
            text=text.strip(),
            tokens_in=int(usage.get("input_tokens", 0) or 0),
            tokens_out=int(usage.get("output_tokens", 0) or 0),
        )

class OllamaLLM:
    def __init__(self, base_url: str, model: str):
        self.base_url = base_url.rstrip("/")
        self.model = model

    def generate(self, prompt: str, temperature: float = 0.2) -> LLMResponse:
        url = f"{self.base_url}/api/generate"
        r = requests.post(url, json={
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": temperature}
        }, timeout=300)
        r.raise_for_status()
        j = r.json()
        text = (j.get("response") or "").strip()
        # Ollama doesn't always provide token counts
        return LLMResponse(text=text, tokens_in=0, tokens_out=0)