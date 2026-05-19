import os
import requests
from dotenv import load_dotenv


load_dotenv()

class OllamaLLM:
    def __init__(self):
        self.url = os.getenv("OLLAMA_BASE_URL")

    def generate(self, query, context):
        prompt = f"{context}\n\nQuestion: {query}"

        res = requests.post(
            f"{self.url}/api/generate",
            json={"model": "llama3", "prompt": prompt, "stream": False}
        )

        return res.json()["response"]