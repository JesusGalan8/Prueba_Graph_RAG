"""
tracing.py — Configuración de Langfuse para observabilidad del pipeline.

Expone decoradores y funciones helper para trazar:
  - Preguntas del usuario
  - Decisión de routing
  - Retrieval (Cypher generado, resultados, latencia)
  - Generación de respuesta (prompt, respuesta, tokens)

Si Langfuse no está configurado, todo funciona igual pero sin tracing.
"""
import functools
from contextlib import contextmanager
from typing import Any, Callable, Optional

from src.config import (
    LANGFUSE_ENABLED,
    LANGFUSE_PUBLIC_KEY,
    LANGFUSE_SECRET_KEY,
    LANGFUSE_BASE_URL,
)

# ─────────────────────────────────────────────────────────────
#  Inicialización de Langfuse (condicional)
# ─────────────────────────────────────────────────────────────

_langfuse_client = None

if LANGFUSE_ENABLED:
    try:
        from langfuse import Langfuse
        _langfuse_client = Langfuse(
            public_key=LANGFUSE_PUBLIC_KEY,
            secret_key=LANGFUSE_SECRET_KEY,
            host=LANGFUSE_BASE_URL,
        )
        print(f"[tracing] ✓ Langfuse conectado en {LANGFUSE_BASE_URL}")
    except Exception as e:
        print(f"[tracing] ⚠ Langfuse no disponible: {e}. Tracing desactivado.")
        _langfuse_client = None
else:
    print("[tracing] Langfuse no configurado. Ejecuta sin observabilidad.")


def is_enabled() -> bool:
    return _langfuse_client is not None


# ─────────────────────────────────────────────────────────────
#  Clase Trace — abstracción sobre Langfuse
# ─────────────────────────────────────────────────────────────

class F1Trace:
    """
    Encapsula un trace de Langfuse para una consulta del usuario.
    Si Langfuse no está disponible, actúa como un no-op.
    """

    def __init__(self, question: str, session_id: Optional[str] = None):
        self.question = question
        self._trace  = None
        self._spans  = {}

        if _langfuse_client:
            self._trace = _langfuse_client.trace(
                name="f1_query",
                input={"question": question},
                session_id=session_id,
                metadata={"app": "F1 GraphRAG"},
            )

    @property
    def trace_id(self) -> Optional[str]:
        return self._trace.id if self._trace else None

    @property
    def trace_url(self) -> Optional[str]:
        if self._trace and LANGFUSE_BASE_URL:
            return f"{LANGFUSE_BASE_URL}/trace/{self.trace_id}"
        return None

    def span_routing(self, retriever_chosen: str, reason: str) -> None:
        """Registra la decisión del router."""
        if self._trace:
            span = self._trace.span(
                name="routing",
                input={"question": self.question},
                output={
                    "retriever": retriever_chosen,
                    "reason": reason,
                },
                metadata={"component": "router"},
            )
            span.end()

    def span_retrieval(self, retriever: str, query: Any,
                        results: Any, latency_ms: float) -> None:
        """Registra el paso de retrieval."""
        if self._trace:
            # Limitar tamaño de resultados para no saturar Langfuse
            results_preview = str(results)[:2000] if results else ""
            span = self._trace.span(
                name=f"retrieval_{retriever}",
                input={"query": str(query)[:1000]},
                output={"results_preview": results_preview},
                metadata={
                    "component": "retrieval",
                    "retriever": retriever,
                    "latency_ms": latency_ms,
                    "result_count": len(results) if isinstance(results, list) else 1,
                },
            )
            span.end()

    def span_generation(self, system_prompt: str, context: str,
                         response: str, model: str) -> None:
        """Registra el paso de generación del LLM."""
        if self._trace:
            generation = self._trace.generation(
                name="llm_generation",
                model=model,
                input=[
                    {"role": "system",    "content": system_prompt[:500]},
                    {"role": "assistant", "content": f"Context:\n{context[:1000]}"},
                    {"role": "user",      "content": self.question},
                ],
                output=response,
                metadata={"component": "generation"},
            )
            generation.end()

    def score(self, name: str, value: float, comment: str = "") -> None:
        """Añade un score al trace (ej. evaluación automática)."""
        if self._trace:
            self._trace.score(
                name=name,
                value=value,
                comment=comment,
            )

    def finish(self, output: str) -> None:
        """Cierra el trace con la respuesta final."""
        if self._trace:
            self._trace.update(output=output)
            _langfuse_client.flush()


# ─────────────────────────────────────────────────────────────
#  Función de flush (llamar al cerrar la aplicación)
# ─────────────────────────────────────────────────────────────

def flush() -> None:
    """Envía todos los traces pendientes a Langfuse."""
    if _langfuse_client:
        _langfuse_client.flush()
