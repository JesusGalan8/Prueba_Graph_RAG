"""
transform.py — Limpia, normaliza y enriquece los DataFrames extraídos de F1DB.
"""
import pandas as pd
import numpy as np

from src.config import SEASON_START, SEASON_END

def _safe_int(val, default=None):
    try:
        if pd.isna(val):
            return default
        return int(float(val))
    except (ValueError, TypeError):
        return default

def _safe_float(val, default=None):
    try:
        if pd.isna(val):
            return default
        return float(val)
    except (ValueError, TypeError):
        return default

def clean_drivers(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Mapeo a nombres internos
    df["driver_id"] = df["id"].astype(str)
    df["numero_permanente"] = df["permanentNumber"].apply(lambda x: _safe_int(x))
    df["fecha_nacimiento"]  = pd.to_datetime(df["dateOfBirth"], errors="coerce")
    df["nombre"]    = df["firstName"].fillna("").str.strip()
    df["apellido"]  = df["lastName"].fillna("").str.strip()
    df["codigo"]    = df.get("abbreviation", pd.Series(dtype=str)).fillna("").str.strip().str.upper()
    df["nacionalidad"] = df["nationalityCountryId"].fillna("").str.strip()
    return df

def clean_constructors(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["constructor_id"] = df["id"].astype(str)
    df["nombre"]       = df["name"].fillna("").str.strip()
    df["nacionalidad"] = df["countryId"].fillna("").str.strip()
    return df

def clean_circuits(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["circuit_id"] = df["id"].astype(str)
    df["nombre"]   = df["name"].fillna("").str.strip()
    df["ubicacion"] = df["placeName"].fillna("").str.strip()
    df["pais"]      = df["countryId"].fillna("").str.strip()
    df["latitud"]   = df["latitude"].apply(_safe_float)
    df["longitud"]  = df["longitude"].apply(_safe_float)
    df["altitud"]   = np.nan # F1DB doesn't have altitude in basic fields
    return df

def clean_races(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["race_id"]    = df["id"].apply(lambda x: _safe_int(x, 0))
    df["temporada"]  = df["year"].apply(lambda x: _safe_int(x, 0))
    df["ronda"]      = df["round"].apply(lambda x: _safe_int(x, 0))
    df["nombre_carrera"] = df["officialName"].fillna(df["grandPrixId"]).fillna("").str.strip()
    df["fecha"] = pd.to_datetime(df["date"], errors="coerce")
    df["hora"]  = df["time"].fillna("").str.strip()
    df["circuit_id"] = df["circuitId"].astype(str)
    return df

def clean_results(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["race_id"]        = df["raceId"].apply(lambda x: _safe_int(x, 0))
    df["driver_id"]      = df["driverId"].astype(str)
    df["constructor_id"] = df["constructorId"].astype(str)
    
    df["posicion_final"]    = df["positionNumber"].apply(_safe_int)
    df["posicion_parrilla"] = df.get("gridPositionNumber", pd.Series(dtype=float)).apply(_safe_int)
    df["puntos"]            = df["points"].apply(_safe_float, default=0.0)
    df["vueltas"]           = df["laps"].apply(_safe_int)
    df["posicion_texto"]    = df["positionText"].replace({"\\N": None, "N": None})
    
    df["ranking_vuelta_rapida"] = None # No direct mapping in F1DB basic results
    df["status_id"]      = None # F1DB has reasonRetired instead of statusId
    return df

def clean_qualifying(df: pd.DataFrame) -> pd.DataFrame:
    if df is None:
        return None
    df = df.copy()
    df["race_id"]        = df["raceId"].apply(lambda x: _safe_int(x, 0))
    df["driver_id"]      = df["driverId"].astype(str)
    df["constructor_id"] = df["constructorId"].astype(str)
    
    df["posicion"] = df["positionNumber"].apply(_safe_int)
    df["q1"] = df.get("q1", pd.Series(dtype=str)).fillna("").str.strip()
    df["q2"] = df.get("q2", pd.Series(dtype=str)).fillna("").str.strip()
    df["q3"] = df.get("q3", pd.Series(dtype=str)).fillna("").str.strip()
    return df

def clean_standings(df: pd.DataFrame, is_driver: bool) -> pd.DataFrame:
    if df is None:
        return None
    df = df.copy()
    df["temporada"] = df["year"].apply(_safe_int)
    df["posicion"] = df["positionNumber"].apply(_safe_int)
    df["puntos"] = df["points"].apply(_safe_float, default=0.0)
    
    if is_driver:
        df["driver_id"] = df["driverId"].astype(str)
        # Aproximación: una victoria si "championshipWon" es True (en realidad championshipWon es solo si ganó el mundial)
        # F1DB no tiene la cuenta de victorias por año en esta tabla, omitimos o dejamos 0
        df["victorias"] = df["championshipWon"].apply(lambda x: 1 if str(x).lower() == 'true' else 0)
    else:
        df["constructor_id"] = df["constructorId"].astype(str)
        df["victorias"] = df["championshipWon"].apply(lambda x: 1 if str(x).lower() == 'true' else 0)
    return df

def clean_pit_stops(df: pd.DataFrame) -> pd.DataFrame:
    if df is None:
        return None
    df = df.copy()
    df["race_id"] = df["raceId"].apply(_safe_int)
    df["driver_id"] = df["driverId"].astype(str)
    df["numero_parada"] = df["stop"].apply(_safe_int)
    df["vuelta"] = df["lap"].apply(_safe_int)
    df["duracion"] = df["time"].astype(str)
    df["milisegundos"] = df["timeMillis"].apply(_safe_int)
    return df

# ─────────────────────────────────────────────────────────────
#  Enriquecimiento con motores
# ─────────────────────────────────────────────────────────────

def enrich_with_motors(results: pd.DataFrame, constructors: pd.DataFrame, motores: pd.DataFrame, races: pd.DataFrame) -> pd.DataFrame:
    results_with_season = results.merge(races[["race_id", "temporada"]], on="race_id", how="left")
    results_with_season = results_with_season.merge(
        constructors[["constructor_id", "nombre"]].rename(columns={"nombre": "nombre_escuderia"}),
        on="constructor_id", how="left"
    )

    def get_motor(row):
        mask = (
            (motores["escuderia"] == row.get("nombre_escuderia", "")) &
            (motores["temporada_inicio"] <= row.get("temporada", 0)) &
            (motores["temporada_fin"]    >= row.get("temporada", 0))
        )
        matches = motores[mask]
        if not matches.empty:
            return matches.iloc[0]["motor"]
        return "Desconocido"

    results_with_season["motor"] = results_with_season.apply(get_motor, axis=1)
    return results_with_season

# ─────────────────────────────────────────────────────────────
#  Generación de descripciones textuales para embeddings
# ─────────────────────────────────────────────────────────────

def generate_driver_description(row: pd.Series) -> str:
    nombre_completo = f"{row.get('nombre', '')} {row.get('apellido', '')}".strip()
    nac  = row.get("nacionalidad", "")
    num  = row.get("numero_permanente", None)
    cod  = row.get("codigo", "")

    desc = f"{nombre_completo} es un piloto de Fórmula 1 de nacionalidad {nac}."
    if num: desc += f" Porta el número permanente {num}."
    if cod: desc += f" Código: {cod}."
    return desc

def generate_constructor_description(row: pd.Series, motores_df: pd.DataFrame) -> str:
    nombre = row.get("nombre", "")
    nac    = row.get("nacionalidad", "")
    mask = motores_df["escuderia"] == nombre
    motores_usados = motores_df[mask]["motor"].unique().tolist() if mask.any() else []

    desc = f"{nombre} es una escudería de Fórmula 1 de nacionalidad {nac}."
    if motores_usados: desc += f" Usó motores de: {', '.join(motores_usados)}."
    return desc

def generate_race_description(race: pd.Series, circuit_name: str = "") -> str:
    nombre = race.get("nombre_carrera", f"GP Ronda {race.get('ronda', '?')}")
    temp   = race.get("temporada", "")
    ronda  = race.get("ronda", "")
    fecha  = race.get("fecha", None)

    desc = f"{nombre} fue la ronda {ronda} de la temporada {temp} de Fórmula 1."
    if circuit_name: desc += f" Se disputó en el circuito {circuit_name}."
    if fecha and not pd.isna(fecha):
        desc += f" Fecha: {fecha.strftime('%d de %B de %Y') if hasattr(fecha, 'strftime') else fecha}."
    return desc

# ─────────────────────────────────────────────────────────────
#  Función principal
# ─────────────────────────────────────────────────────────────

def transform_all(dfs: dict) -> dict:
    dfs["drivers"]      = clean_drivers(dfs["drivers"])
    dfs["constructors"] = clean_constructors(dfs["constructors"])
    dfs["circuits"]     = clean_circuits(dfs["circuits"])
    dfs["races"]        = clean_races(dfs["races"])
    
    # --- FILTRADO POR AÑOS (V2) ---
    races_mask = (dfs["races"]["temporada"] >= SEASON_START) & (dfs["races"]["temporada"] <= SEASON_END)
    dfs["races"] = dfs["races"][races_mask].copy()
    valid_race_ids = set(dfs["races"]["race_id"])

    dfs["results"]      = clean_results(dfs["results"])
    dfs["results"] = dfs["results"][dfs["results"]["race_id"].isin(valid_race_ids)].copy()

    dfs["qualifying"] = clean_qualifying(dfs.get("qualifying"))
    if dfs["qualifying"] is not None:
        dfs["qualifying"] = dfs["qualifying"][dfs["qualifying"]["race_id"].isin(valid_race_ids)].copy()

    dfs["driver_standings"] = clean_standings(dfs.get("driver_standings"), True)
    if dfs["driver_standings"] is not None:
        ds_mask = (dfs["driver_standings"]["temporada"] >= SEASON_START) & (dfs["driver_standings"]["temporada"] <= SEASON_END)
        dfs["driver_standings"] = dfs["driver_standings"][ds_mask].copy()

    dfs["constructor_standings"] = clean_standings(dfs.get("constructor_standings"), False)
    if dfs["constructor_standings"] is not None:
        cs_mask = (dfs["constructor_standings"]["temporada"] >= SEASON_START) & (dfs["constructor_standings"]["temporada"] <= SEASON_END)
        dfs["constructor_standings"] = dfs["constructor_standings"][cs_mask].copy()

    dfs["pit_stops"] = clean_pit_stops(dfs.get("pit_stops"))
    if dfs["pit_stops"] is not None:
        dfs["pit_stops"] = dfs["pit_stops"][dfs["pit_stops"]["race_id"].isin(valid_race_ids)].copy()

    dfs["results_enriched"] = enrich_with_motors(
        results=dfs["results"],
        constructors=dfs["constructors"],
        motores=dfs["motores"],
        races=dfs["races"],
    )

    dfs["drivers"]["descripcion"] = dfs["drivers"].apply(generate_driver_description, axis=1)
    dfs["constructors"]["descripcion"] = dfs["constructors"].apply(
        lambda r: generate_constructor_description(r, dfs["motores"]), axis=1
    )

    circ_lookup = dfs["circuits"].set_index("circuit_id")["nombre"].to_dict()
    dfs["races"]["descripcion"] = dfs["races"].apply(
        lambda r: generate_race_description(r, circ_lookup.get(r.get("circuit_id", ""), "")),
        axis=1
    )

    return dfs
