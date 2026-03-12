import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from src.prompting import judge_faithfulness_prompt

def semantic_similarity(embedder, a: str, b: str) -> float:
    va = embedder.embed_texts([a])[0]
    vb = embedder.embed_texts([b])[0]
    return float(cosine_similarity([va], [vb])[0][0])

def faithfulness_score(llm, question: str, answer: str, context: str) -> tuple[int, str]:
    prompt = judge_faithfulness_prompt(question, answer, context)
    resp = llm.generate(prompt, temperature=0.0).text
    # best-effort JSON parse
    try:
        j = json.loads(resp)
        return int(j.get("score", 1)), str(j.get("rationale", "")[:300])
    except Exception:
        return 1, "Could not parse judge output."