import time
import json
from dataclasses import dataclass
from src.vectorstore import FaissStore
from src.retriever import HybridRetriever, merge_results
from src.prompting import build_prompt, judge_faithfulness_prompt
from src.safety import check_safety
from src.cache import SimpleTTLCache
from src.llm_providers import LLM
from src.embeddings import Embedder
from src.reranker import Reranker


@dataclass
class Answer:
    answer: str
    citations: list[str]
    latency_ms: float
    tokens_in: int
    tokens_out: int
    estimated_cost: float
    safety_note: str
    retrieval: list[dict]


class RAGPipeline:
    def __init__(self, store: FaissStore, embedder: Embedder, llm: LLM):
        self.store = store
        self.embedder = embedder
        self.llm = llm
        self.hybrid = HybridRetriever(store.meta)
        self.cache = SimpleTTLCache(ttl_seconds=900)
        self.reranker = Reranker()

    def _to_context_blocks(self, chunks):
        blocks = []
        for c in chunks:
            cit = f"doc:{c.doc_id} p:{c.page} c:{c.chunk_id}"
            blocks.append({"citation": cit, "text": c.text})
        return blocks

    def _estimate_cost(self, tokens_in: int, tokens_out: int) -> float:
        # Simple demo pricing estimate; tune to your chosen model if needed
        return (tokens_in * 0.000001) + (tokens_out * 0.000002)

    def _judge_answer(self, question: str, answer_text: str, judge_context: str) -> tuple[int, str]:
        """
        Returns:
            (score from 1-5, rationale)
        """
        prompt = judge_faithfulness_prompt(
            question=question,
            answer=answer_text,
            context=judge_context
        )
        resp = self.llm.generate(prompt, temperature=0.0).text

        try:
            parsed = json.loads(resp)
            score = int(parsed.get("score", 1))
            rationale = str(parsed.get("rationale", ""))[:300]
            score = max(1, min(5, score))
            return score, rationale
        except Exception:
            return 1, "Judge output was not valid JSON."

    def ask(
        self,
        question: str,
        k: int = 6,
        mode: str = "grounded",
        temperature: float = 0.2,
        use_cache: bool = True
    ) -> Answer:
        safety = check_safety(question)
        if not safety.allowed:
            return Answer(
                answer=safety.user_message,
                citations=[],
                latency_ms=0.0,
                tokens_in=0,
                tokens_out=0,
                estimated_cost=0.0,
                safety_note=safety.reason,
                retrieval=[]
            )

        cache_key = f"{mode}|{k}|{temperature}|{question.strip().lower()}"
        if use_cache:
            cached = self.cache.get(cache_key)
            if cached:
                if safety.user_message and safety.user_message not in cached.safety_note:
                    cached.safety_note = safety.user_message
                return cached

        t0 = time.time()

        # 1) Hybrid retrieval
        qv = self.embedder.embed_query(question)
        dense = self.store.search(qv, k=max(10, k * 2))
        bm25 = self.hybrid.bm25_search(question, k=max(10, k * 2))
        merged = merge_results(dense, bm25, self.store.meta, k=max(10, k * 2))

        # 2) Reranking
        reranked = self.reranker.rerank(question, merged, top_k=k)

        # 3) Build grounded context
        #context_blocks = self._to_context_blocks(reranked)
        context_blocks = self._to_context_blocks(merged[:k])
        citations = [b["citation"] for b in context_blocks]
        judge_context = "\n\n".join([f"[{b['citation']}]\n{b['text']}" for b in context_blocks])

        tokens_in_total = 0
        tokens_out_total = 0
        retrieval_debug = [{"citation": b["citation"], "preview": b["text"][:160]} for b in context_blocks]

        # 4) Generation modes
        if mode == "self_consistency":
            # Generate 3 candidates and select the most faithful using the judge
            candidates = []
            gen_temp = max(0.5, min(0.9, float(temperature) if temperature else 0.7))

            for _ in range(3):
                prompt = build_prompt(question, context_blocks, mode="cot")
                resp = self.llm.generate(prompt, temperature=gen_temp)
                tokens_in_total += resp.tokens_in
                tokens_out_total += resp.tokens_out

                score, rationale = self._judge_answer(question, resp.text, judge_context)
                candidates.append({
                    "text": resp.text,
                    "score": score,
                    "rationale": rationale,
                })

            # Highest judge score wins; tie-break by shorter answer
            candidates.sort(key=lambda x: (x["score"], -len(x["text"])), reverse=True)
            best = candidates[0]

            answer_text = best["text"]
            retrieval_debug.append({
                "self_consistency_selected_score": best["score"],
                "judge_rationale": best["rationale"]
            })

        else:
            prompt_mode = mode if mode in ("grounded", "cot") else "grounded"
            prompt = build_prompt(question, context_blocks, mode=prompt_mode)
            resp = self.llm.generate(prompt, temperature=temperature)
            tokens_in_total += resp.tokens_in
            tokens_out_total += resp.tokens_out
            answer_text = resp.text

        latency_ms = (time.time() - t0) * 1000.0
        estimated_cost = self._estimate_cost(tokens_in_total, tokens_out_total)

        ans = Answer(
            answer=answer_text,
            citations=citations,
            latency_ms=latency_ms,
            tokens_in=tokens_in_total,
            tokens_out=tokens_out_total,
            estimated_cost=estimated_cost,
            safety_note=safety.user_message,
            retrieval=retrieval_debug,
        )

        if use_cache:
            self.cache.set(cache_key, ans)

        return ans