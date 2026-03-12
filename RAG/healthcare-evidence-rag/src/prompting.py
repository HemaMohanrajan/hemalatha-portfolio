def build_prompt(query: str, context_blocks: list[dict], mode: str = "grounded") -> str:
    # context_blocks: [{"citation": "...", "text": "..."}]
    ctx = "\n\n".join([f"[{b['citation']}]\n{b['text']}" for b in context_blocks])

    system = (
        "You are a medical evidence assistant. You MUST only use the provided context.\n"
        "If the context is insufficient, say exactly: 'Insufficient evidence in provided documents.'\n"
        "Do NOT invent citations. Do NOT provide dosing instructions.\n"
        "Always include citations in square brackets like [doc:XYZ p:3 c:10].\n"
    )

    if mode == "cot":
        return (
            f"{system}\n"
            f"Context:\n{ctx}\n\n"
            f"Question: {query}\n\n"
            "Think step-by-step (briefly) using only the context, then give a final answer with citations.\n"
        )

    # default grounded, no explicit chain-of-thought
    return (
        f"{system}\n"
        f"Context:\n{ctx}\n\n"
        f"Question: {query}\n\n"
        "Answer concisely with citations.\n"
    )

def judge_faithfulness_prompt(question: str, answer: str, context: str) -> str:
    return (
        "You are grading a medical QA system for faithfulness.\n"
        "Given QUESTION, ANSWER, and CONTEXT, score faithfulness 1-5.\n"
        "5 = fully supported by context, 1 = contradicts or is unsupported.\n"
        "Return JSON ONLY: {\"score\": <int>, \"rationale\": \"<short>\"}\n\n"
        f"QUESTION:\n{question}\n\n"
        f"ANSWER:\n{answer}\n\n"
        f"CONTEXT:\n{context}\n"
    )