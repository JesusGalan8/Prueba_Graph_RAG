"""
vector_search.py — Retriever que vectoriza la pregunta y busca nodos similares
por cosine similarity usando los índices vectoriales de Neo4j.
"""
import time
from typing import Optional

import ollama
from neo4j import GraphDatabase

from src.config import (
    NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD,
    OLLAMA_BASE_URL, OLLAMA_EMBED_MODEL,
    VECTOR_INDEX_NAME, VECTOR_TOP_K,
)


class VectorSearchRetriever:
    """
    Retriever semántico: vectoriza la pregunta y busca los K nodos
    más similares en el grafo, luego expande su vecindario para contexto.
    """

    def __init__(self):
        self._neo4j  = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
        self._ollama = ollama.Client(host=OLLAMA_BASE_URL)

    def _embed(self, text: str) -> list[float]:
        response = self._ollama.embeddings(model=OLLAMA_EMBED_MODEL, prompt=text)
        return response["embedding"]

    def _search_by_label(self, embedding: list[float], label: str,
                          top_k: int) -> list[dict]:
        """Búsqueda KNN en el índice vectorial de un label específico."""
        index_name = f"{VECTOR_INDEX_NAME}_{label.lower()}"
        query = f"""
        CALL db.index.vector.queryNodes('{index_name}', $top_k, $embedding)
        YIELD node, score
        RETURN node, score, labels(node) AS labels
        ORDER BY score DESC
        """
        with self._neo4j.session() as session:
            try:
                result = session.run(query, {"top_k": top_k, "embedding": embedding})
                return [
                    {
                        "node":   dict(r["node"]._properties),
                        "score":  round(r["score"], 4),
                        "labels": r["labels"],
                    }
                    for r in result
                ]
            except Exception:
                return []

    def _expand_neighborhood(self, node_id: int, label: str) -> list[dict]:
        """
        Expande el vecindario de un nodo para dar más contexto al LLM.
        Por ejemplo, si encontramos un Piloto, obtenemos sus últimas carreras.
        """
        expansions = {
            "Piloto": """
                MATCH (p:Piloto) WHERE id(p) = $nid
                OPTIONAL MATCH (p)-[r:PARTICIPO_EN]->(c:Carrera)
                WHERE r.posicion_final = 1
                WITH p, c ORDER BY c.temporada DESC LIMIT 5
                RETURN p.nombre + ' ' + p.apellido AS entidad,
                       collect(c.nombre + ' ' + toString(c.temporada)) AS victorias_recientes
            """,
            "Escuderia": """
                MATCH (e:Escuderia) WHERE id(e) = $nid
                OPTIONAL MATCH (e)-[:USA_MOTOR]->(m:Motor)
                OPTIONAL MATCH (e)-[pc:POSICION_CONSTRUCTORES]->(t:Temporada)
                WITH e, m, pc ORDER BY t.anio DESC LIMIT 3
                RETURN e.nombre AS entidad, m.fabricante AS motor,
                       collect(toString(t.anio) + ':P' + toString(pc.posicion)) AS posiciones
            """,
            "Carrera": """
                MATCH (c:Carrera) WHERE id(c) = $nid
                OPTIONAL MATCH (p:Piloto)-[r:PARTICIPO_EN]->(c)
                WHERE r.posicion_final <= 3
                RETURN c.nombre + ' ' + toString(c.temporada) AS entidad,
                       collect(p.apellido + '(P' + toString(r.posicion_final) + ')') AS podio
            """,
            "Comunidad": """
                MATCH (c:Comunidad) WHERE id(c) = $nid
                RETURN c.resumen AS entidad, c.nivel AS nivel
            """,
        }

        q = expansions.get(label)
        if not q:
            return []
        with self._neo4j.session() as session:
            try:
                result = session.run(q, {"nid": node_id})
                return [dict(r) for r in result]
            except Exception:
                return []

    def retrieve(self, question: str,
                 labels: Optional[list[str]] = None,
                 top_k: int = VECTOR_TOP_K) -> tuple[list[dict], list[float], float]:
        """
        Búsqueda semántica sobre los nodos del grafo.

        Args:
            question: Pregunta del usuario
            labels:   Labels a buscar. Por defecto: Piloto, Escuderia, Carrera
            top_k:    Número de resultados por label

        Returns:
            (results, embedding_used, latency_ms)
        """
        t0 = time.time()
        if labels is None:
            labels = ["Piloto", "Escuderia", "Carrera", "Comunidad"]

        embedding = self._embed(question)
        all_results = []

        for label in labels:
            nodes = self._search_by_label(embedding, label, top_k=max(3, top_k // len(labels)))
            for node_data in nodes:
                node = node_data["node"]
                # Añadir info de vecindario
                node_neo4j_id = node.get("id")   # nuestro ID interno
                node_data["context"] = []
                all_results.append(node_data)

        # Ordenar por score
        all_results.sort(key=lambda x: x["score"], reverse=True)
        latency = (time.time() - t0) * 1000

        return all_results[:top_k], embedding, latency

    def format_context(self, results: list[dict]) -> str:
        """Convierte los resultados a texto para el LLM."""
        if not results:
            return "No se encontraron entidades relevantes."

        lines = []
        for item in results:
            node   = item.get("node", {})
            score  = item.get("score", 0)
            labels = item.get("labels", [])

            label = labels[0] if labels else "Entidad"
            desc  = node.get("descripcion", node.get("resumen", ""))
            nombre = node.get("nombre", node.get("apellido", node.get("fabricante", node.get("id", ""))))

            if desc:
                lines.append(f"[{label}] {nombre} (relevancia: {score:.2f})\n  {desc}")
            else:
                props = {k: v for k, v in node.items() if k not in ("embedding", "descripcion")}
                lines.append(f"[{label}] {nombre} (relevancia: {score:.2f})\n  {props}")

        return "\n\n".join(lines)

    def close(self):
        self._neo4j.close()
