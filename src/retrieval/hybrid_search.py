"""
hybrid_search.py — Retriever híbrido que combina full-text search con vector search.
Ideal para preguntas con términos concretos Y contexto semántico.
"""
import time
from typing import Optional

import ollama
from neo4j import GraphDatabase

from src.config import (
    NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD,
    OLLAMA_BASE_URL, OLLAMA_EMBED_MODEL,
    FULLTEXT_INDEX_NAME, VECTOR_INDEX_NAME,
    VECTOR_TOP_K, HYBRID_VECTOR_WEIGHT,
)


class HybridSearchRetriever:
    """
    Retriever híbrido: combina full-text Lucene (Neo4j) con vector KNN.
    Pondera los scores y devuelve los resultados más relevantes.
    """

    def __init__(self):
        self._neo4j  = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
        self._ollama = ollama.Client(host=OLLAMA_BASE_URL)

    def _embed(self, text: str) -> list[float]:
        response = self._ollama.embeddings(model=OLLAMA_EMBED_MODEL, prompt=text)
        return response["embedding"]

    def _fulltext_search(self, query: str, top_k: int) -> list[dict]:
        """Búsqueda full-text con Lucene sobre nombres y descripciones."""
        # Sanitizar query para Lucene
        safe_query = query.replace('"', '').replace("'", "")

        q = f"""
        CALL db.index.fulltext.queryNodes('{FULLTEXT_INDEX_NAME}', $query)
        YIELD node, score
        RETURN node, score, labels(node) AS labels
        ORDER BY score DESC
        LIMIT $top_k
        """
        with self._neo4j.session() as session:
            try:
                result = session.run(q, {"query": safe_query, "top_k": top_k})
                return [
                    {
                        "node":       dict(r["node"]._properties),
                        "ft_score":   r["score"],
                        "vec_score":  0.0,
                        "labels":     r["labels"],
                    }
                    for r in result
                ]
            except Exception:
                return []

    def _vector_search_all(self, embedding: list[float], top_k: int) -> list[dict]:
        """Vector KNN sobre todos los labels disponibles."""
        labels  = ["Piloto", "Escuderia", "Carrera", "Circuito"]
        results = []
        per_label = max(2, top_k // len(labels))

        for label in labels:
            index_name = f"{VECTOR_INDEX_NAME}_{label.lower()}"
            q = f"""
            CALL db.index.vector.queryNodes('{index_name}', $top_k, $embedding)
            YIELD node, score
            RETURN node, score, labels(node) AS labels
            """
            with self._neo4j.session() as session:
                try:
                    res = session.run(q, {"top_k": per_label, "embedding": embedding})
                    for r in res:
                        results.append({
                            "node":      dict(r["node"]._properties),
                            "ft_score":  0.0,
                            "vec_score": r["score"],
                            "labels":    r["labels"],
                        })
                except Exception:
                    pass

        return results

    def _merge_and_rank(self, ft_results: list[dict],
                         vec_results: list[dict],
                         vec_weight: float = HYBRID_VECTOR_WEIGHT) -> list[dict]:
        """
        Combina y re-rankea los resultados usando Reciprocal Rank Fusion (RRF)
        ponderada por los pesos de cada fuente.
        """
        ft_weight = 1 - vec_weight
        merged: dict[str, dict] = {}

        def node_key(node: dict) -> str:
            """Clave única para identificar un nodo."""
            return str(node.get("id") or node.get("fabricante") or node.get("nombre", ""))

        # Normalizar scores de full-text (escala diferente a cosine)
        max_ft = max((r["ft_score"] for r in ft_results), default=1.0)
        for r in ft_results:
            r["ft_score_norm"] = r["ft_score"] / max_ft if max_ft > 0 else 0

        # Combinar
        for r in ft_results:
            key = node_key(r["node"])
            if key not in merged:
                merged[key] = r.copy()
                merged[key]["combined_score"] = r["ft_score_norm"] * ft_weight
            else:
                merged[key]["combined_score"] += r["ft_score_norm"] * ft_weight

        for r in vec_results:
            key = node_key(r["node"])
            if key not in merged:
                merged[key] = r.copy()
                merged[key]["combined_score"] = r["vec_score"] * vec_weight
            else:
                merged[key]["combined_score"] = merged[key].get("combined_score", 0) + r["vec_score"] * vec_weight
                merged[key]["vec_score"] = r["vec_score"]

        ranked = sorted(merged.values(), key=lambda x: x["combined_score"], reverse=True)
        return ranked

    def retrieve(self, question: str,
                 top_k: int = VECTOR_TOP_K) -> tuple[list[dict], float]:
        """
        Búsqueda híbrida: full-text + vector.

        Returns:
            (results, latency_ms)
        """
        t0 = time.time()

        # Extraer términos clave para full-text (primeras N palabras)
        words = [w for w in question.split() if len(w) > 3]
        ft_query = " OR ".join(words[:5]) if words else question

        ft_results  = self._fulltext_search(ft_query, top_k=top_k)
        embedding   = self._embed(question)
        vec_results = self._vector_search_all(embedding, top_k=top_k)

        merged = self._merge_and_rank(ft_results, vec_results)

        latency = (time.time() - t0) * 1000
        return merged[:top_k], latency

    def format_context(self, results: list[dict]) -> str:
        """Convierte los resultados a texto para el LLM."""
        if not results:
            return "No se encontraron resultados relevantes."

        lines = []
        for item in results[:10]:   # limitar al top 10 para el contexto
            node   = item.get("node", {})
            score  = item.get("combined_score", 0)
            labels = item.get("labels", [])

            label  = labels[0] if labels else "Entidad"
            desc   = node.get("descripcion", "")
            nombre = node.get("nombre", node.get("apellido", node.get("fabricante", "?")))

            source = []
            if item.get("ft_score", 0) > 0:
                source.append("texto")
            if item.get("vec_score", 0) > 0:
                source.append("semántica")
            src_str = "+".join(source) if source else "hybrid"

            if desc:
                lines.append(f"[{label}/{src_str}] {nombre} (score: {score:.3f})\n  {desc}")
            else:
                props = {k: v for k, v in node.items()
                         if k not in ("embedding", "descripcion") and v is not None}
                lines.append(f"[{label}/{src_str}] {nombre} (score: {score:.3f})\n  {props}")

        return "\n\n".join(lines)

    def close(self):
        self._neo4j.close()
