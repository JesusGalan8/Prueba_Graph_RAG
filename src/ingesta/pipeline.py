"""
pipeline.py — Orquesta el pipeline ETL completo:
  download → extract → transform → load_graph → load_embeddings

Uso:
  python -m src.ingesta.pipeline             # Ejecución completa
  python -m src.ingesta.pipeline --skip-download   # Si ya tienes los CSVs
  python -m src.ingesta.pipeline --skip-embeddings # Solo grafo, sin vectores
  python -m src.ingesta.pipeline --force-embeddings # Regenerar embeddings
"""
import argparse
import time

from src.ingesta.download        import download_f1db_csvs
from src.ingesta.extract         import extract_all
from src.ingesta.transform       import transform_all
from src.ingesta.load_graph      import load_graph
from src.ingesta.load_embeddings import load_embeddings
from src.config                  import SEASON_START, SEASON_END


def run_pipeline(
    skip_download:    bool = False,
    skip_embeddings:  bool = False,
    force_download:   bool = False,
    force_embeddings: bool = False,
) -> None:
    """
    Ejecuta el pipeline ETL completo.

    Args:
        skip_download:    Si True, asume que los CSVs ya están descargados.
        skip_embeddings:  Si True, omite la generación de embeddings.
        force_download:   Si True, vuelve a descargar aunque ya existan los CSVs.
        force_embeddings: Si True, regenera embeddings aunque ya existan.
    """
    t_start = time.time()
    print("=" * 60)
    print("  🏎  F1 GraphRAG — Pipeline de Ingesta")
    print(f"  Temporadas: {SEASON_START} – {SEASON_END}")
    print("=" * 60)

    # ── Paso 1: Descarga ──────────────────────────────────
    if not skip_download:
        print("\n[PASO 1/4] Descargando datos de F1DB...")
        download_f1db_csvs(force=force_download)
    else:
        print("\n[PASO 1/4] Descarga omitida (--skip-download).")

    # ── Paso 2: Extracción ────────────────────────────────
    print(f"\n[PASO 2/4] Extrayendo datos ({SEASON_START}–{SEASON_END})...")
    dfs = extract_all(SEASON_START, SEASON_END)

    # ── Paso 3: Transformación ────────────────────────────
    print("\n[PASO 3/4] Transformando y enriqueciendo datos...")
    dfs = transform_all(dfs)

    # ── Paso 4a: Cargar Domain Graph ──────────────────────
    print("\n[PASO 4a/4] Cargando Domain Graph en Neo4j...")
    load_graph(dfs)

    # ── Paso 4b: Cargar Embeddings (Lexical Graph) ────────
    if not skip_embeddings:
        print("\n[PASO 4b/4] Generando embeddings (Lexical Graph)...")
        print("  NOTA: Este paso puede tardar varios minutos.")
        print("  Asegúrate de que Ollama está corriendo y tiene el modelo:")
        print(f"  → ollama pull {__import__('src.config', fromlist=['OLLAMA_EMBED_MODEL']).OLLAMA_EMBED_MODEL}")
        load_embeddings(force=force_embeddings)
    else:
        print("\n[PASO 4b/4] Embeddings omitidos (--skip-embeddings).")

    total = time.time() - t_start
    print("\n" + "=" * 60)
    print(f"  ✓ Pipeline completado en {total:.1f}s ({total/60:.1f} min)")
    print("  Neo4j Browser: http://localhost:7474")
    print("  Langfuse:      http://localhost:3000")
    print("  API docs:      http://localhost:8000/docs")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="F1 GraphRAG — Pipeline de Ingesta ETL"
    )
    parser.add_argument("--skip-download",    action="store_true",
                        help="Omitir descarga de CSVs (usar los existentes)")
    parser.add_argument("--skip-embeddings",  action="store_true",
                        help="Omitir generación de embeddings")
    parser.add_argument("--force-download",   action="store_true",
                        help="Forzar re-descarga aunque los CSVs existan")
    parser.add_argument("--force-embeddings", action="store_true",
                        help="Forzar regeneración de embeddings")

    args = parser.parse_args()
    run_pipeline(
        skip_download=args.skip_download,
        skip_embeddings=args.skip_embeddings,
        force_download=args.force_download,
        force_embeddings=args.force_embeddings,
    )
