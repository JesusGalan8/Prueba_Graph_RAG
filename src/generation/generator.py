"""
generator.py — Genera la respuesta final usando el LLM (Ollama + Llama 3.1).

Integra los retrievers, el router y Langfuse para el pipeline completo.
"""
import time
from typing import Optional

import ollama

from src.config              import OLLAMA_BASE_URL, OLLAMA_LLM_MODEL
from src.retrieval.router    import QueryRouter
from src.retrieval.text2cypher   import Text2CypherRetriever
from src.retrieval.vector_search import VectorSearchRetriever
from src.retrieval.hybrid_search import HybridSearchRetriever
from src.retrieval.prompts       import GENERATION_SYSTEM, GENERATION_USER
from src.observability.tracing   import F1Trace


class F1Generator:
    """
    Pipeline completo de GraphRAG:
    pregunta → routing → retrieval → generación → respuesta
    """

    def __init__(self):
        self._ollama = ollama.Client(host=OLLAMA_BASE_URL)
        self._router  = QueryRouter()
        self._t2c     = Text2CypherRetriever()
        self._vector  = VectorSearchRetriever()
        self._hybrid  = HybridSearchRetriever()

    def query(self, question: str,
               session_id: Optional[str] = None,
               verbose: bool = False) -> dict:
        """
        Procesa una pregunta del usuario y devuelve la respuesta.

        Args:
            question:   Pregunta en lenguaje natural
            session_id: ID de sesión para Langfuse (opcional)
            verbose:    Si True, incluye Cypher y debug info en el resultado

        Returns:
            dict con claves:
              - answer:      Respuesta en lenguaje natural
              - retriever:   Tipo de retriever usado
              - cypher:      Query Cypher (solo Text2Cypher)
              - context:     Contexto del grafo (texto)
              - latency_ms:  Latencia total
              - trace_url:   URL de Langfuse (si disponible)
        """
        t0    = time.time()
        trace = F1Trace(question=question, session_id=session_id)

        # ── 1. Routing ────────────────────────────────────
        retriever_type, reason = self._router.route(question)

        if verbose:
            print(f"  [router] → {retriever_type} ({reason})")

        trace.span_routing(retriever_type, reason)

        # ── 2. Retrieval ──────────────────────────────────
        context     = ""
        cypher_used = None
        ret_latency = 0.0

        if retriever_type == "CYPHER":
            results, cypher_used, ret_latency = self._t2c.retrieve(question)
            context = self._t2c.format_context(results)
            trace.span_retrieval("text2cypher", cypher_used, results, ret_latency)
            if verbose:
                print(f"  [cypher] {cypher_used}")

        elif retriever_type == "VECTOR":
            results, embedding, ret_latency = self._vector.retrieve(question)
            context = self._vector.format_context(results)
            trace.span_retrieval("vector", question, results, ret_latency)

        else:  # HYBRID
            results, ret_latency = self._hybrid.retrieve(question)
            context = self._hybrid.format_context(results)
            trace.span_retrieval("hybrid", question, results, ret_latency)

        if verbose:
            print(f"  [context] {len(context)} chars | latencia retrieval: {ret_latency:.0f}ms")

        # ── 3. Generación ─────────────────────────────────
        gen_t0 = time.time()
        user_prompt = GENERATION_USER.format(context=context, question=question)

        response = self._ollama.chat(
            model=OLLAMA_LLM_MODEL,
            messages=[
                {"role": "system", "content": GENERATION_SYSTEM},
                {"role": "user",   "content": user_prompt},
            ],
            options={"temperature": 0.3},
        )
        answer = response["message"]["content"].strip()
        gen_latency = (time.time() - gen_t0) * 1000

        trace.span_generation(GENERATION_SYSTEM, context, answer, OLLAMA_LLM_MODEL)

        # ── 4. Finalizar trace ────────────────────────────
        total_ms = (time.time() - t0) * 1000
        trace.finish(answer)

        return {
            "answer":      answer,
            "retriever":   retriever_type,
            "cypher":      cypher_used,
            "context":     context,
            "latency_ms":  round(total_ms, 0),
            "ret_latency": round(ret_latency, 0),
            "gen_latency": round(gen_latency, 0),
            "trace_url":   trace.trace_url,
        }

    def close(self):
        self._t2c.close()
        self._vector.close()
        self._hybrid.close()
