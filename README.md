# 🏎️ F1 GraphRAG

Sistema de preguntas y respuestas sobre **Fórmula 1 (2014–2024)** basado en un grafo de conocimiento (Neo4j) + LLM local (Ollama + Llama 3.1). 100% gratuito, sin APIs de pago.

## Stack

| Componente | Tecnología |
|---|---|
| Graph Database | Neo4j 5.x Community |
| LLM | Ollama + `llama3.1:8b` |
| Embeddings | Ollama + `nomic-embed-text` |
| Observabilidad | Langfuse self-hosted |
| API | Python + FastAPI |
| CLI | Python + Rich |
| Infraestructura | Docker Compose |

---

## Requisitos

- Docker + Docker Compose
- Ollama instalado localmente (o en el contenedor)
- ~8 GB de RAM libres
- ~10 GB de disco (Neo4j + modelos Ollama)

---

## Arranque rápido

### 1. Configurar entorno

```bash
cp .env.example .env
# El .env está preconfigurado para entorno local
```

### 2. Levantar infraestructura

```bash
docker-compose up -d
```

Esto levanta:
- **Neo4j** → http://localhost:7474 (user: `neo4j`, pass: `f1graphrag2024`)
- **Ollama** → http://localhost:11434
- **Langfuse** → http://localhost:3000
- **API** → http://localhost:8000/docs

### 3. Descargar modelos de Ollama (primera vez, ~5GB)

```bash
docker exec -it f1_ollama ollama pull llama3.1:8b
docker exec -it f1_ollama ollama pull nomic-embed-text
```

### 4. Configurar Langfuse (primera vez)

1. Abre http://localhost:3000
2. Crea una cuenta de administrador
3. Ve a **Settings → API Keys** y genera un par de claves
4. Copia las claves en tu `.env`:

```env
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
```

5. Reinicia el contenedor de API:
```bash
docker-compose restart api
```

### 5. Ejecutar el pipeline de ingesta

```bash
# Pipeline completo (descarga CSVs + crea grafo + genera embeddings)
docker exec -it f1_api python -m src.ingesta.pipeline

# Opciones disponibles:
# --skip-download    (si ya tienes los CSVs)
# --skip-embeddings  (solo grafo, sin vectores)
# --force-embeddings (regenerar vectores)
```

⏱️ **Tiempo estimado:** 10-20 minutos (dependiendo de la CPU para los embeddings)

### 6. ¡Preguntar sobre F1!

```bash
# Lanzar el chat en terminal
docker exec -it f1_api python -m src.cli.chat

# Con modo verbose (muestra Cypher generado)
docker exec -it f1_api python -m src.cli.chat --verbose
```

---

## Comandos del chat

| Comando | Descripción |
|---|---|
| `/ayuda` | Muestra la ayuda |
| `/stats` | Estadísticas del grafo |
| `/verbose` | Activa/desactiva modo detallado |
| `/limpiar` | Limpia la pantalla |
| `/salir` | Sale del chat |

---

## Ejemplos de preguntas

```
¿Quién ganó más carreras en 2023?
¿Cuántas victorias tuvo Hamilton con Mercedes en 2019?
¿Qué pilotos ganaron saliendo desde posición 10 o peor con motor Mercedes en escudería cliente?
¿Para qué equipos corrió Alonso entre 2014 y 2024?
¿Quién ganó el GP de Mónaco en 2021?
¿Qué motor usó Red Bull en 2022?
```

---

## Evaluación

```bash
# Evaluación completa (30 preguntas)
docker exec -it f1_api python eval/evaluate.py

# Solo preguntas fáciles (verificación rápida)
docker exec -it f1_api python eval/evaluate.py --quick

# Preguntas específicas
docker exec -it f1_api python eval/evaluate.py --ids GT-001,GT-003,GT-007
```

---

## Arquitectura

```
Pregunta → Router → Text2Cypher / Vector / Hybrid
                         ↓
                    Neo4j Graph
                         ↓
                    Contexto
                         ↓
              Llama 3.1 → Respuesta
                         ↓
                    Langfuse (trace)
```

---

## Estructura del proyecto

```
f1-graphrag/
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
├── .env.example
├── data/
│   ├── csv/                    ← CSVs descargados (auto)
│   └── motores_escuderias.csv  ← Mapeo motor↔escudería (manual)
├── src/
│   ├── config.py
│   ├── ingesta/                ← ETL pipeline
│   ├── retrieval/              ← Router + 3 retrievers
│   ├── generation/             ← Generador de respuestas
│   ├── api/                    ← FastAPI
│   ├── cli/                    ← Chat terminal
│   └── observability/          ← Langfuse
└── eval/
    ├── ground_truth.json       ← 30 preguntas de evaluación
    └── evaluate.py             ← Script de evaluación
```
