"""
text2cypher.py — Retriever que usa el LLM para generar una query Cypher
a partir de la pregunta del usuario, la ejecuta en Neo4j y devuelve los resultados.
"""
import time
from typing import Any

import ollama
from neo4j import GraphDatabase

from src.config import (
    NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD,
    OLLAMA_BASE_URL, OLLAMA_LLM_MODEL, CYPHER_MAX_RESULTS,
)
from src.retrieval.prompts import TEXT2CYPHER_SYSTEM, TEXT2CYPHER_USER


class Text2CypherRetriever:
    """
    Retriever que traduce la pregunta a Cypher con el LLM,
    ejecuta la query y devuelve los resultados.
    """

    def __init__(self):
        self._neo4j = GraphDatabase.driver(
            NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD)
        )
        self._ollama = ollama.Client(host=OLLAMA_BASE_URL)
        self.last_cypher = None   # Para mostrar en el CLI en modo verbose

    def _generate_cypher(self, question: str) -> str:
        """Invoca el LLM para generar la query Cypher."""
        response = self._ollama.chat(
            model=OLLAMA_LLM_MODEL,
            messages=[
                {"role": "system",  "content": TEXT2CYPHER_SYSTEM},
                {"role": "user",    "content": TEXT2CYPHER_USER.format(question=question)},
            ],
            options={"temperature": 0.0},   # Determinista para Cypher
        )
        return response["message"]["content"].strip()

    def _clean_cypher(self, raw: str) -> str:
        """Limpia el Cypher devuelto por el LLM (elimina markdown, etc.)."""
        # Eliminar bloques de código markdown
        raw = raw.replace("```cypher", "").replace("```", "").strip()
        # Tomar solo hasta el primer punto y coma si lo hay
        if ";" in raw:
            raw = raw.split(";")[0].strip()
        return raw

    def _execute_cypher(self, cypher: str) -> list[dict]:
        """Ejecuta el Cypher en Neo4j y devuelve los resultados como lista de dicts."""
        with self._neo4j.session() as session:
            result = session.run(cypher)
            records = []
            for record in result:
                row = {}
                for key in record.keys():
                    val = record[key]
                    # Convertir objetos Neo4j a tipos Python básicos
                    if hasattr(val, "_properties"):
                        row[key] = dict(val._properties)
                    else:
                        row[key] = val
                records.append(row)
            return records[:CYPHER_MAX_RESULTS]

    def retrieve(self, question: str) -> tuple[list[dict], str, float]:
        """
        Genera Cypher, lo ejecuta y devuelve los resultados.

        Returns:
            (results, cypher_used, latency_ms)
        """
        t0 = time.time()

        # 1. Generar Cypher
        raw_cypher = self._generate_cypher(question)

        if "INVALID_QUERY" in raw_cypher or not raw_cypher.strip().upper().startswith("MATCH"):
            return [], raw_cypher, (time.time() - t0) * 1000

        cypher = self._clean_cypher(raw_cypher)
        self.last_cypher = cypher

        # 2. Ejecutar en Neo4j
        try:
            results = self._execute_cypher(cypher)
        except Exception as e:
            # Si falla, intentar con LIMIT reducido
            try:
                cypher_limited = cypher + " LIMIT 10" if "LIMIT" not in cypher.upper() else cypher
                results = self._execute_cypher(cypher_limited)
            except Exception as e2:
                results = [{"error": str(e2), "cypher": cypher}]

        latency = (time.time() - t0) * 1000
        return results, cypher, latency

    def format_context(self, results: list[dict]) -> str:
        """Convierte los resultados a texto para el LLM."""
        if not results:
            return "No se encontraron resultados en la base de datos."
        if "error" in results[0]:
            return f"Error en la consulta: {results[0].get('error', '')}"

        lines = []
        for i, row in enumerate(results, 1):
            parts = [f"{k}: {v}" for k, v in row.items() if v is not None]
            lines.append(f"{i}. {' | '.join(parts)}")
        return "\n".join(lines)

    def close(self):
        self._neo4j.close()
