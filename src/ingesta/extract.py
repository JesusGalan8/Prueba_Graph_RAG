"""
extract.py — Lee los CSVs de F1DB (f1db-*) y devuelve DataFrames filtrados
             por el rango de temporadas configurado (2014-2024).
"""
import pandas as pd
from pathlib import Path

from src.config import CSV_DIR, DATA_DIR, SEASON_START, SEASON_END

def _read_f1db_csv(filename: str, required: bool = True) -> pd.DataFrame | None:
    path = CSV_DIR / filename
    if path.exists():
        return pd.read_csv(path, low_memory=False, na_values=["\\N", "N", ""])
    if required:
        raise FileNotFoundError(
            f"No se encontró {filename} en {CSV_DIR}. "
            "Ejecuta primero: python -m src.ingesta.download"
        )
    return None

def extract_all(season_start: int = SEASON_START,
                season_end:   int = SEASON_END) -> dict[str, pd.DataFrame]:
    print(f"[extract] Leyendo CSVs de F1DB para temporadas {season_start}–{season_end}...")

    # 1. Carreras (filtradas por temporada)
    races = _read_f1db_csv("f1db-races.csv")
    races = races[
        (races["year"] >= season_start) &
        (races["year"] <= season_end)
    ].copy()
    valid_race_ids = set(races["id"].unique())
    valid_years = set(races["year"].unique())
    print(f"[extract]   Carreras: {len(races)}")

    # 2. Entidades principales (sin filtro, o filtradas por ids usados si quisiéramos)
    drivers = _read_f1db_csv("f1db-drivers.csv")
    constructors = _read_f1db_csv("f1db-constructors.csv")
    circuits = _read_f1db_csv("f1db-circuits.csv")

    # 3. Resultados
    results = _read_f1db_csv("f1db-races-race-results.csv")
    results = results[results["raceId"].isin(valid_race_ids)].copy()
    print(f"[extract]   Resultados: {len(results)}")

    # 4. Clasificación
    qualifying = _read_f1db_csv("f1db-races-qualifying-results.csv", required=False)
    if qualifying is not None:
        qualifying = qualifying[qualifying["raceId"].isin(valid_race_ids)].copy()
        print(f"[extract]   Clasificaciones: {len(qualifying)}")

    # 5. Pit Stops
    pit_stops = _read_f1db_csv("f1db-races-pit-stops.csv", required=False)
    if pit_stops is not None:
        pit_stops = pit_stops[pit_stops["raceId"].isin(valid_race_ids)].copy()
        print(f"[extract]   Pit stops: {len(pit_stops)}")

    # 6. Standings anuales (f1db las tiene por año, no por carrera)
    driver_standings = _read_f1db_csv("f1db-seasons-driver-standings.csv", required=False)
    if driver_standings is not None:
        driver_standings = driver_standings[driver_standings["year"].isin(valid_years)].copy()
    
    constructor_standings = _read_f1db_csv("f1db-seasons-constructor-standings.csv", required=False)
    if constructor_standings is not None:
        constructor_standings = constructor_standings[constructor_standings["year"].isin(valid_years)].copy()

    # 7. Motores (mapeo manual)
    motores_path = DATA_DIR / "motores_escuderias.csv"
    motores = pd.read_csv(motores_path)
    motores = motores[
        (motores["temporada_inicio"] <= season_end) &
        (motores["temporada_fin"]    >= season_start)
    ].copy()

    print("[extract] ✓ Extracción completada.")

    return {
        "drivers":               drivers,
        "constructors":          constructors,
        "circuits":              circuits,
        "races":                 races,
        "results":               results,
        "qualifying":            qualifying,
        "driver_standings":      driver_standings,
        "constructor_standings": constructor_standings,
        "pit_stops":             pit_stops,
        "motores":               motores,
    }

if __name__ == "__main__":
    dfs = extract_all()
    for name, df in dfs.items():
        if df is not None:
            print(f"  {name:30s}: {len(df):>7,} filas")
