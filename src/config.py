"""
config.py — Configuración centralizada del proyecto F1 GraphRAG.
Lee variables de entorno (desde .env o docker-compose).
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Cargar .env si existe (útil para desarrollo local sin Docker)
load_dotenv()

# ── Rutas base ───────────────────────────────────────────────
ROOT_DIR = Path(__file__).parent.parent
DATA_DIR = ROOT_DIR / "data"
CSV_DIR  = DATA_DIR / "csv"
EVAL_DIR = ROOT_DIR / "eval"

# ── Neo4j ────────────────────────────────────────────────────
NEO4J_URI      = os.getenv("NEO4J_URI",      "bolt://localhost:7687")
NEO4J_USER     = os.getenv("NEO4J_USER",     "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "f1graphrag2024")

# ── Ollama ───────────────────────────────────────────────────
OLLAMA_BASE_URL    = os.getenv("OLLAMA_BASE_URL",    "http://localhost:11434")
OLLAMA_LLM_MODEL   = os.getenv("OLLAMA_LLM_MODEL",   "llama3.1:8b")
OLLAMA_EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL",  "nomic-embed-text")

# ── Langfuse ─────────────────────────────────────────────────
LANGFUSE_PUBLIC_KEY = os.getenv("LANGFUSE_PUBLIC_KEY", "")
LANGFUSE_SECRET_KEY = os.getenv("LANGFUSE_SECRET_KEY", "")
LANGFUSE_BASE_URL   = os.getenv("LANGFUSE_BASE_URL",   "http://localhost:3000")

# Si no hay keys de Langfuse configuradas, el tracing se desactiva en silencio
LANGFUSE_ENABLED = bool(
    LANGFUSE_PUBLIC_KEY and
    LANGFUSE_PUBLIC_KEY != "pk-lf-not-set" and
    not LANGFUSE_PUBLIC_KEY.startswith("pk-lf-XXXX")
)

# ── Temporadas ───────────────────────────────────────────────
SEASON_START = int(os.getenv("SEASON_START", 2014))
SEASON_END   = int(os.getenv("SEASON_END",   2024))

# ── F1DB — URL de descarga de CSVs ───────────────────────────
F1DB_BASE_URL = os.getenv(
    "F1DB_BASE_URL",
    "https://github.com/f1db/f1db/releases/latest/download"
)

# Archivos CSV que necesitamos descargar
F1DB_CSV_FILES = [
    "f1db-csv.zip",   # F1DB empaqueta todos los CSVs en un único ZIP
]

# ── Índices de Neo4j ─────────────────────────────────────────
VECTOR_INDEX_NAME   = "f1_vector_index"
VECTOR_INDEX_COMMUNITY = "f1_community_vector_index"
FULLTEXT_INDEX_NAME = "f1_fulltext_index"
VECTOR_DIMENSIONS   = 768    # nomic-embed-text produce vectores de 768 dims

# ── Parámetros de retrieval ───────────────────────────────────
VECTOR_TOP_K         = 10    # Nodos más similares a recuperar
CYPHER_MAX_RESULTS   = 50    # Límite de resultados por query Cypher
HYBRID_VECTOR_WEIGHT = 0.6   # Peso del vector en búsqueda híbrida (0-1)
