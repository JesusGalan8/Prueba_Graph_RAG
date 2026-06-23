"""
download.py — Descarga automática de los CSVs de F1DB desde GitHub.

F1DB publica un zip con todos los CSVs en cada release de GitHub.
Este módulo descarga y descomprime ese zip en data/csv/.
"""
import zipfile
import io
from pathlib import Path

import requests
from tqdm import tqdm

from src.config import F1DB_BASE_URL, CSV_DIR


# URL del ZIP con todos los CSVs de F1DB
F1DB_ZIP_URL = f"{F1DB_BASE_URL}/f1db-csv.zip"


def download_f1db_csvs(force: bool = False) -> Path:
    """
    Descarga y descomprime el archivo f1db-csv.zip de F1DB.

    Args:
        force: Si True, vuelve a descargar aunque los CSVs ya existan.

    Returns:
        Path al directorio donde se guardaron los CSVs.
    """
    CSV_DIR.mkdir(parents=True, exist_ok=True)

    # Comprobar si ya existe algún CSV para no volver a descargar
    existing_csvs = list(CSV_DIR.glob("*.csv"))
    if existing_csvs and not force:
        print(f"[download] Ya existen {len(existing_csvs)} CSVs en {CSV_DIR}. "
              f"Usa force=True para forzar la descarga.")
        return CSV_DIR

    print(f"[download] Descargando CSVs de F1DB desde:\n  {F1DB_ZIP_URL}")

    try:
        response = requests.get(F1DB_ZIP_URL, stream=True, timeout=120)
        response.raise_for_status()
    except requests.RequestException as e:
        # Si falla la descarga oficial, intentar con el mirror de Kaggle
        print(f"[download] Error con F1DB: {e}")
        print("[download] Intentando mirror alternativo (Kaggle)...")
        return _download_kaggle_fallback()

    # Calcular tamaño total para la barra de progreso
    total_size = int(response.headers.get("content-length", 0))
    print(f"[download] Tamaño: {total_size / 1_048_576:.1f} MB")

    # Descargar en memoria
    buffer = io.BytesIO()
    downloaded = 0
    with tqdm(total=total_size, unit="B", unit_scale=True, desc="Descargando") as pbar:
        for chunk in response.iter_content(chunk_size=8192):
            buffer.write(chunk)
            downloaded += len(chunk)
            pbar.update(len(chunk))

    # Descomprimir
    print(f"[download] Descomprimiendo en {CSV_DIR}...")
    buffer.seek(0)
    with zipfile.ZipFile(buffer) as zf:
        zf.extractall(CSV_DIR)

    csvs = list(CSV_DIR.glob("*.csv"))
    print(f"[download] ✓ {len(csvs)} archivos CSV extraídos en {CSV_DIR}")
    return CSV_DIR


def _download_kaggle_fallback() -> Path:
    """
    Mirror alternativo: intenta descargar desde el repositorio de CSVs
    compatible con Ergast mantenido en GitHub por la comunidad.
    """
    # URL alternativa: CSV individuales del repo f1db en GitHub
    BASE = "https://raw.githubusercontent.com/f1db/f1db/main/src/data/csv"
    tables = [
        "drivers", "constructors", "circuits", "races",
        "results", "qualifying", "constructor_standings",
        "driver_standings", "lap_times", "pit_stops",
        "status", "seasons", "constructor_results",
    ]

    CSV_DIR.mkdir(parents=True, exist_ok=True)
    for table in tqdm(tables, desc="Descargando CSVs individuales"):
        url = f"{BASE}/{table}.csv"
        try:
            r = requests.get(url, timeout=30)
            r.raise_for_status()
            (CSV_DIR / f"{table}.csv").write_bytes(r.content)
        except requests.RequestException as e:
            print(f"[download] ⚠ No se pudo descargar {table}.csv: {e}")

    csvs = list(CSV_DIR.glob("*.csv"))
    print(f"[download] ✓ {len(csvs)} CSVs descargados (fallback)")
    return CSV_DIR


if __name__ == "__main__":
    download_f1db_csvs()
