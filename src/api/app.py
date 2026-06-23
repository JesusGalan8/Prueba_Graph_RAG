"""
app.py — API REST con FastAPI para el F1 GraphRAG.

Endpoints:
  POST /query         → Realiza una consulta al GraphRAG
  GET  /health        → Health check
  GET  /stats         → Estadísticas del grafo
  GET  /docs          → Swagger UI automático
"""
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.generation.generator import F1Generator
from src.observability.tracing import flush


# ─────────────────────────────────────────────────────────────
#  Lifespan: inicializa y libera recursos
# ─────────────────────────────────────────────────────────────

generator: Optional[F1Generator] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global generator
    print("[api] Inicializando F1 GraphRAG...")
    generator = F1Generator()
    print("[api] ✓ Listo para recibir preguntas.")
    yield
    # Limpieza al cerrar
    if generator:
        generator.close()
    flush()
    print("[api] Recursos liberados.")


# ─────────────────────────────────────────────────────────────
#  App
# ─────────────────────────────────────────────────────────────

app = FastAPI(
    title="F1 GraphRAG API",
    description="Sistema de preguntas y respuestas sobre Fórmula 1 (2014-2024) "
                "basado en un grafo de conocimiento Neo4j + LLM local (Llama 3.1).",
    version="1.0.0",
    lifespan=lifespan,
)


# ─────────────────────────────────────────────────────────────
#  Modelos Pydantic
# ─────────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    question:   str
    session_id: Optional[str] = None
    verbose:    bool = False

    model_config = {
        "json_schema_extra": {
            "example": {
                "question":   "¿Quién ganó más carreras en 2023?",
                "session_id": "sesion-001",
                "verbose":    False,
            }
        }
    }


class QueryResponse(BaseModel):
    answer:      str
    retriever:   str
    cypher:      Optional[str]
    latency_ms:  float
    trace_url:   Optional[str]


class HealthResponse(BaseModel):
    status:  str
    neo4j:   bool
    ollama:  bool


class StatsResponse(BaseModel):
    pilotos:    int
    escuderias: int
    carreras:   int
    temporadas: int
    relaciones: int


# ─────────────────────────────────────────────────────────────
#  Endpoints
# ─────────────────────────────────────────────────────────────

@app.post("/query", response_model=QueryResponse, tags=["GraphRAG"])
async def query(request: QueryRequest):
    """
    Realiza una consulta sobre la base de datos de F1.

    - **question**: Pregunta en lenguaje natural (español)
    - **session_id**: ID de sesión para agrupar traces en Langfuse (opcional)
    - **verbose**: Incluye Cypher generado en la respuesta
    """
    if not generator:
        raise HTTPException(status_code=503, detail="El sistema no está inicializado.")
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="La pregunta no puede estar vacía.")

    result = generator.query(
        question=request.question.strip(),
        session_id=request.session_id,
        verbose=request.verbose,
    )

    return QueryResponse(
        answer=result["answer"],
        retriever=result["retriever"],
        cypher=result.get("cypher"),
        latency_ms=result["latency_ms"],
        trace_url=result.get("trace_url"),
    )


@app.get("/health", response_model=HealthResponse, tags=["Sistema"])
async def health():
    """Verifica el estado de los servicios dependientes."""
    neo4j_ok  = False
    ollama_ok = False

    # Verificar Neo4j
    try:
        from neo4j import GraphDatabase
        from src.config import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD
        driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
        with driver.session() as s:
            s.run("RETURN 1")
        driver.close()
        neo4j_ok = True
    except Exception:
        pass

    # Verificar Ollama
    try:
        import ollama as ol
        from src.config import OLLAMA_BASE_URL
        client = ol.Client(host=OLLAMA_BASE_URL)
        client.list()
        ollama_ok = True
    except Exception:
        pass

    status = "ok" if (neo4j_ok and ollama_ok) else "degraded"
    return HealthResponse(status=status, neo4j=neo4j_ok, ollama=ollama_ok)


@app.get("/stats", response_model=StatsResponse, tags=["Sistema"])
async def stats():
    """Devuelve estadísticas del grafo en Neo4j."""
    from neo4j import GraphDatabase
    from src.config import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD

    try:
        driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
        with driver.session() as session:
            counts = {}
            for label in ["Piloto", "Escuderia", "Carrera", "Temporada"]:
                r = session.run(f"MATCH (n:{label}) RETURN count(n) AS n").single()
                counts[label.lower()] = r["n"] if r else 0
            r_rel = session.run("MATCH ()-[r]->() RETURN count(r) AS n").single()
            counts["relaciones"] = r_rel["n"] if r_rel else 0
        driver.close()

        return StatsResponse(
            pilotos=counts.get("piloto", 0),
            escuderias=counts.get("escuderia", 0),
            carreras=counts.get("carrera", 0),
            temporadas=counts.get("temporada", 0),
            relaciones=counts.get("relaciones", 0),
        )
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Error al conectar con Neo4j: {e}")


@app.get("/", tags=["Sistema"])
async def root():
    return {
        "app":     "F1 GraphRAG",
        "version": "1.0.0",
        "docs":    "/docs",
        "query":   "POST /query",
        "health":  "GET /health",
        "stats":   "GET /stats",
    }
