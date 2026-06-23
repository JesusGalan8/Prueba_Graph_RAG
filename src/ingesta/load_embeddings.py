"""
load_embeddings.py — Genera embeddings vectoriales con nomic-embed-text (Ollama)
y los almacena en Neo4j. Luego crea los índices vectoriales y full-text.

Proceso:
  1. Para cada nodo con propiedad 'descripcion' → genera embedding
  2. Almacena el vector en la propiedad 'embedding' del nodo
  3. Crea índice vectorial para búsqueda KNN
  4. Crea índice full-text para búsqueda híbrida
"""
import time
from typing import Optional

import ollama
from neo4j import GraphDatabase
from tqdm import tqdm

from src.config import (
    NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD,
    OLLAMA_BASE_URL, OLLAMA_EMBED_MODEL,
    VECTOR_INDEX_NAME, FULLTEXT_INDEX_NAME,
    VECTOR_DIMENSIONS,
)


BATCH_SIZE = 50   # Embeddings por lote (limitado por Ollama)


# ─────────────────────────────────────────────────────────────
#  Cliente Ollama
# ─────────────────────────────────────────────────────────────

def get_ollama_client():
    return ollama.Client(host=OLLAMA_BASE_URL)


def embed_text(client, text: str) -> list[float]:
    """Genera un embedding para un texto dado."""
    response = client.embeddings(model=OLLAMA_EMBED_MODEL, prompt=text)
    return response["embedding"]


# ─────────────────────────────────────────────────────────────
#  Índices en Neo4j
# ─────────────────────────────────────────────────────────────

def create_vector_index(driver) -> None:
    """Crea el índice vectorial para búsqueda KNN."""
    q = f"""
    CREATE VECTOR INDEX {VECTOR_INDEX_NAME} IF NOT EXISTS
    FOR (n:Piloto|Escuderia|Carrera|Circuito)
    ON n.embedding
    OPTIONS {{
        indexConfig: {{
            `vector.dimensions`: {VECTOR_DIMENSIONS},
            `vector.similarity_function`: 'cosine'
        }}
    }}
    """
    # Neo4j no soporta labels múltiples en un solo índice vectorial,
    # así que creamos uno por label
    labels = ["Piloto", "Escuderia", "Carrera", "Circuito"]
    with driver.session() as session:
        for label in labels:
            idx_name = f"{VECTOR_INDEX_NAME}_{label.lower()}"
            q = f"""
            CREATE VECTOR INDEX {idx_name} IF NOT EXISTS
            FOR (n:{label}) ON n.embedding
            OPTIONS {{
                indexConfig: {{
                    `vector.dimensions`: {VECTOR_DIMENSIONS},
                    `vector.similarity_function`: 'cosine'
                }}
            }}
            """
            try:
                session.run(q)
                print(f"[embeddings] ✓ Índice vectorial creado: {idx_name}")
            except Exception as e:
                print(f"[embeddings] ⚠ Índice {idx_name}: {e}")


def create_fulltext_index(driver) -> None:
    """Crea índice full-text para búsqueda por texto."""
    q = f"""
    CREATE FULLTEXT INDEX {FULLTEXT_INDEX_NAME} IF NOT EXISTS
    FOR (n:Piloto|Escuderia|Carrera|Circuito|Motor)
    ON EACH [n.nombre, n.apellido, n.descripcion, n.fabricante]
    """
    with driver.session() as session:
        try:
            session.run(q)
            print(f"[embeddings] ✓ Índice full-text creado: {FULLTEXT_INDEX_NAME}")
        except Exception as e:
            print(f"[embeddings] ⚠ Full-text index: {e}")


# ─────────────────────────────────────────────────────────────
#  Generación y almacenamiento de embeddings
# ─────────────────────────────────────────────────────────────

def _get_nodes_without_embedding(session, label: str) -> list[dict]:
    """Recupera nodos de un label que no tienen embedding todavía."""
    result = session.run(f"""
        MATCH (n:{label})
        WHERE n.descripcion IS NOT NULL
          AND n.descripcion <> ''
          AND n.embedding IS NULL
        RETURN elementId(n) AS eid, n.descripcion AS texto
    """)
    return [{"eid": r["eid"], "texto": r["texto"]} for r in result]


def _set_embedding(session, eid: str, embedding: list[float]) -> None:
    session.run(
        "MATCH (n) WHERE elementId(n) = $eid SET n.embedding = $embedding",
        {"eid": eid, "embedding": embedding}
    )


def embed_label(driver, ollama_client, label: str, force: bool = False) -> int:
    """
    Genera y almacena embeddings para todos los nodos de un label.

    Args:
        force: Si True, regenera embeddings aunque ya existan.

    Returns:
        Número de nodos procesados.
    """
    if force:
        with driver.session() as session:
            session.run(f"MATCH (n:{label}) REMOVE n.embedding")

    with driver.session() as session:
        nodes = _get_nodes_without_embedding(session, label)

    if not nodes:
        print(f"[embeddings] {label}: sin nodos pendientes.")
        return 0

    print(f"[embeddings] Generando embeddings para {len(nodes)} nodos de :{label}...")
    count = 0
    for node in tqdm(nodes, desc=f"Embeddings {label}"):
        try:
            embedding = embed_text(ollama_client, node["texto"])
            with driver.session() as session:
                _set_embedding(session, node["eid"], embedding)
            count += 1
        except Exception as e:
            print(f"  ⚠ Error en nodo {node['eid']}: {e}")

    return count


def load_embeddings(force: bool = False) -> None:
    """
    Proceso completo de generación de embeddings:
    1. Crea índices en Neo4j
    2. Genera embeddings para Piloto, Escuderia, Carrera, Circuito
    """
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    ollama_client = get_ollama_client()

    t0 = time.time()

    # Verificar que el modelo está disponible
    print(f"[embeddings] Verificando modelo {OLLAMA_EMBED_MODEL}...")
    try:
        test_emb = embed_text(ollama_client, "test de conexión F1")
        print(f"[embeddings] ✓ Modelo OK — {len(test_emb)} dimensiones")
        if len(test_emb) != VECTOR_DIMENSIONS:
            print(f"[embeddings] ⚠ Dimensiones inesperadas: {len(test_emb)} vs {VECTOR_DIMENSIONS} esperadas")
    except Exception as e:
        print(f"[embeddings] ✗ Error al conectar con Ollama: {e}")
        print(f"  Asegúrate de que Ollama está corriendo y tiene el modelo:")
        print(f"  ollama pull {OLLAMA_EMBED_MODEL}")
        driver.close()
        return

    try:
        # Crear índices primero
        create_vector_index(driver)
        create_fulltext_index(driver)

        # Generar embeddings por label
        total = 0
        for label in ["Piloto", "Escuderia", "Carrera", "Circuito"]:
            total += embed_label(driver, ollama_client, label, force=force)

    finally:
        driver.close()

    elapsed = time.time() - t0
    print(f"\n[embeddings] ✓ {total} embeddings generados en {elapsed:.1f}s")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true",
                        help="Regenerar embeddings aunque ya existan")
    args = parser.parse_args()
    load_embeddings(force=args.force)
