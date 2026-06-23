"""
router.py — Decide qué retriever usar basándose en el análisis de la pregunta.

Estrategia:
  1. Pregunta rápida con heurísticas (sin LLM) → más rápido
  2. Si la heurística no es concluyente → usa el LLM clasificador
"""
import re
import time

import ollama

from src.config import OLLAMA_BASE_URL, OLLAMA_LLM_MODEL
from src.retrieval.prompts import ROUTER_SYSTEM, ROUTER_USER


# ─────────────────────────────────────────────────────────────
#  Heurísticas rápidas (sin LLM)
# ─────────────────────────────────────────────────────────────

# Patrones que sugieren una query Cypher (datos concretos)
CYPHER_PATTERNS = [
    r"\d{4}",                                         # año concreto
    r"\b(ganó|ganaron|victoria|victorias|ganador)\b",
    r"\b(posición|posicion|parrilla|grid|pole)\b",
    r"\b(puntos|campeonato|clasificación|clasificacion)\b",
    r"\b(cuántas|cuantas|cuántos|cuantos)\b",
    r"\b(primer|primero|segundo|tercero|podio)\b",
    r"\b(GP de|Grand Prix|carrera de|circuito de)\b",
    r"\b(para qué|para que|corrió|corrio|equipo)\b",
]

# Patrones que sugieren búsqueda semántica (conceptos difusos)
VECTOR_PATTERNS = [
    r"\b(mejor|mejores|más dominante|dominación|estilo)\b",
    r"\b(similar|parecido|como|recuerda)\b",
    r"\b(cuéntame|cuentame|explícame|explicame|háblame)\b",
    r"\b(historia|legado|impacto|influencia)\b",
]


def _heuristic_route(question: str) -> str | None:
    """
    Clasificación rápida por heurísticas.
    Devuelve 'CYPHER', 'VECTOR' o None si no es concluyente.
    """
    q = question.lower()
    cypher_hits  = sum(1 for p in CYPHER_PATTERNS  if re.search(p, q, re.IGNORECASE))
    vector_hits  = sum(1 for p in VECTOR_PATTERNS  if re.search(p, q, re.IGNORECASE))

    if cypher_hits >= 2 and vector_hits == 0:
        return "CYPHER"
    if vector_hits >= 2 and cypher_hits == 0:
        return "VECTOR"
    if cypher_hits == 1 and vector_hits == 0:
        return "CYPHER"   # la mayoría de preguntas F1 son factuales
    return None           # no concluyente → usar LLM


def _llm_route(question: str, ollama_client) -> str:
    """Clasifica la pregunta usando el LLM como fallback."""
    response = ollama_client.chat(
        model=OLLAMA_LLM_MODEL,
        messages=[
            {"role": "system", "content": ROUTER_SYSTEM},
            {"role": "user",   "content": ROUTER_USER.format(question=question)},
        ],
        options={"temperature": 0.0},
    )
    answer = response["message"]["content"].strip().upper()

    if "CYPHER" in answer:
        return "CYPHER"
    elif "VECTOR" in answer:
        return "VECTOR"
    else:
        return "HYBRID"


# ─────────────────────────────────────────────────────────────
#  Router principal
# ─────────────────────────────────────────────────────────────

class QueryRouter:
    """
    Decide qué retriever usar para cada pregunta.
    Usa heurísticas primero (rápido) y el LLM como fallback.
    """

    def __init__(self):
        self._ollama = ollama.Client(host=OLLAMA_BASE_URL)
        self.last_decision = None
        self.last_reason   = None

    def route(self, question: str) -> tuple[str, str]:
        """
        Decide el retriever.

        Returns:
            (retriever_type, reason)
            retriever_type: 'CYPHER' | 'VECTOR' | 'HYBRID'
        """
        # 1. Intento rápido con heurísticas
        heuristic = _heuristic_route(question)

        if heuristic:
            reason = f"Heurística ({heuristic})"
            self.last_decision = heuristic
            self.last_reason   = reason
            return heuristic, reason

        # 2. Fallback al LLM
        decision = _llm_route(question, self._ollama)
        reason   = f"LLM clasificador ({OLLAMA_LLM_MODEL})"

        self.last_decision = decision
        self.last_reason   = reason
        return decision, reason
