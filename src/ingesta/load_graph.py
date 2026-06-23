"""
load_graph.py — Crea nodos y relaciones en Neo4j a partir de los DataFrames transformados.

Usa transacciones por lotes (batches) para maximizar el rendimiento.
Define constraints e índices antes de insertar datos.
"""
import time
from typing import Any

import pandas as pd
from neo4j import GraphDatabase, Driver
from tqdm import tqdm

from src.config import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD


BATCH_SIZE = 500   # Nodos/relaciones por transacción


# ─────────────────────────────────────────────────────────────
#  Conexión
# ─────────────────────────────────────────────────────────────

def get_driver() -> Driver:
    return GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))


# ─────────────────────────────────────────────────────────────
#  Esquema: constraints e índices
# ─────────────────────────────────────────────────────────────

SCHEMA_QUERIES = [
    # Constraints de unicidad
    "CREATE CONSTRAINT piloto_id IF NOT EXISTS FOR (p:Piloto) REQUIRE p.id IS UNIQUE",
    "CREATE CONSTRAINT escuderia_id IF NOT EXISTS FOR (e:Escuderia) REQUIRE e.id IS UNIQUE",
    "CREATE CONSTRAINT circuito_id IF NOT EXISTS FOR (c:Circuito) REQUIRE c.id IS UNIQUE",
    "CREATE CONSTRAINT carrera_id IF NOT EXISTS FOR (r:Carrera) REQUIRE r.id IS UNIQUE",
    "CREATE CONSTRAINT temporada_id IF NOT EXISTS FOR (t:Temporada) REQUIRE t.anio IS UNIQUE",
    "CREATE CONSTRAINT motor_id IF NOT EXISTS FOR (m:Motor) REQUIRE m.fabricante IS UNIQUE",
    "CREATE CONSTRAINT estado_id IF NOT EXISTS FOR (s:Estado) REQUIRE s.id IS UNIQUE",
    # Índices de búsqueda
    "CREATE INDEX piloto_apellido IF NOT EXISTS FOR (p:Piloto) ON (p.apellido)",
    "CREATE INDEX escuderia_nombre IF NOT EXISTS FOR (e:Escuderia) ON (e.nombre)",
    "CREATE INDEX carrera_temporada IF NOT EXISTS FOR (r:Carrera) ON (r.temporada)",
    "CREATE INDEX carrera_nombre IF NOT EXISTS FOR (r:Carrera) ON (r.nombre)",
]


def setup_schema(driver: Driver) -> None:
    print("[load_graph] Configurando schema (constraints + índices)...")
    with driver.session() as session:
        for q in SCHEMA_QUERIES:
            try:
                session.run(q)
            except Exception as e:
                print(f"  ⚠ Schema: {e}")
    print("[load_graph] ✓ Schema configurado.")


# ─────────────────────────────────────────────────────────────
#  Utilidad de batch
# ─────────────────────────────────────────────────────────────

def _run_batch(session, query: str, records: list[dict]) -> None:
    """Ejecuta una query con UNWIND sobre una lista de registros."""
    session.run(query, {"rows": records})


def _batched(data: list, size: int = BATCH_SIZE):
    for i in range(0, len(data), size):
        yield data[i:i + size]


# ─────────────────────────────────────────────────────────────
#  Nodos
# ─────────────────────────────────────────────────────────────

def load_temporadas(driver: Driver, dfs: dict) -> None:
    races = dfs["races"]
    years = sorted(races["temporada"].unique().tolist())
    records = [{"anio": int(y)} for y in years]

    q = """
    UNWIND $rows AS row
    MERGE (t:Temporada {anio: row.anio})
    """
    print(f"[load_graph] Cargando {len(records)} temporadas...")
    with driver.session() as session:
        _run_batch(session, q, records)


def load_pilotos(driver: Driver, dfs: dict) -> None:
    df = dfs["drivers"]
    records = []
    for _, row in df.iterrows():
        records.append({
            "id":                str(row["driver_id"]),
            "nombre":            str(row.get("nombre", "")),
            "apellido":          str(row.get("apellido", "")),
            "codigo":            str(row.get("codigo", "")),
            "numero_permanente": row.get("numero_permanente"),
            "nacionalidad":      str(row.get("nacionalidad", "")),
            "fecha_nacimiento":  str(row["fecha_nacimiento"].date()) if pd.notna(row.get("fecha_nacimiento")) else None,
            "descripcion":       str(row.get("descripcion", "")),
        })

    q = """
    UNWIND $rows AS row
    MERGE (p:Piloto {id: row.id})
    SET p.nombre = row.nombre,
        p.apellido = row.apellido,
        p.codigo = row.codigo,
        p.numero_permanente = row.numero_permanente,
        p.nacionalidad = row.nacionalidad,
        p.fecha_nacimiento = row.fecha_nacimiento,
        p.descripcion = row.descripcion
    """
    print(f"[load_graph] Cargando {len(records)} pilotos...")
    with driver.session() as session:
        for batch in _batched(records):
            _run_batch(session, q, batch)


def load_escuderias(driver: Driver, dfs: dict) -> None:
    df = dfs["constructors"]
    records = []
    for _, row in df.iterrows():
        records.append({
            "id":          str(row["constructor_id"]),
            "nombre":      str(row.get("nombre", "")),
            "nacionalidad": str(row.get("nacionalidad", "")),
            "descripcion": str(row.get("descripcion", "")),
        })

    q = """
    UNWIND $rows AS row
    MERGE (e:Escuderia {id: row.id})
    SET e.nombre = row.nombre,
        e.nacionalidad = row.nacionalidad,
        e.descripcion = row.descripcion
    """
    print(f"[load_graph] Cargando {len(records)} escuderías...")
    with driver.session() as session:
        for batch in _batched(records):
            _run_batch(session, q, batch)


def load_motores(driver: Driver, dfs: dict) -> None:
    df = dfs["motores"]
    fabricantes = df["motor"].unique().tolist()
    records = [{"fabricante": str(f)} for f in fabricantes]

    q = """
    UNWIND $rows AS row
    MERGE (m:Motor {fabricante: row.fabricante})
    """
    print(f"[load_graph] Cargando {len(records)} motores...")
    with driver.session() as session:
        _run_batch(session, q, records)


def load_circuitos(driver: Driver, dfs: dict) -> None:
    df = dfs["circuits"]
    records = []
    for _, row in df.iterrows():
        records.append({
            "id":       str(row["circuit_id"]),
            "nombre":   str(row.get("nombre", "")),
            "ubicacion": str(row.get("ubicacion", "")),
            "pais":     str(row.get("pais", "")),
            "latitud":  row.get("latitud"),
            "longitud": row.get("longitud"),
            "altitud":  row.get("altitud"),
        })

    q = """
    UNWIND $rows AS row
    MERGE (c:Circuito {id: row.id})
    SET c.nombre = row.nombre,
        c.ubicacion = row.ubicacion,
        c.pais = row.pais,
        c.latitud = row.latitud,
        c.longitud = row.longitud,
        c.altitud = row.altitud
    """
    print(f"[load_graph] Cargando {len(records)} circuitos...")
    with driver.session() as session:
        for batch in _batched(records):
            _run_batch(session, q, batch)


def load_estados(driver: Driver, dfs: dict) -> None:
    if dfs.get("status") is None:
        return
    df = dfs["status"]
    records = []
    for _, row in df.iterrows():
        sid = row.get("status_id") or row.get("statusId")
        desc = row.get("status") or row.get("descripcion") or row.get("description", "")
        if sid:
            records.append({"id": int(sid), "descripcion": str(desc)})

    q = """
    UNWIND $rows AS row
    MERGE (s:Estado {id: row.id})
    SET s.descripcion = row.descripcion
    """
    print(f"[load_graph] Cargando {len(records)} estados...")
    with driver.session() as session:
        for batch in _batched(records):
            _run_batch(session, q, batch)


def load_carreras(driver: Driver, dfs: dict) -> None:
    df = dfs["races"]
    records = []
    for _, row in df.iterrows():
        records.append({
            "id":         int(row["race_id"]),
            "temporada":  int(row["temporada"]),
            "ronda":      int(row["ronda"]) if pd.notna(row.get("ronda")) else None,
            "nombre":     str(row.get("nombre_carrera", "")),
            "fecha":      str(row["fecha"].date()) if pd.notna(row.get("fecha")) else None,
            "hora":       str(row.get("hora", "")),
            "circuit_id": str(row["circuit_id"]) if pd.notna(row.get("circuit_id")) else None,
            "descripcion": str(row.get("descripcion", "")),
        })

    q = """
    UNWIND $rows AS row
    MERGE (r:Carrera {id: row.id})
    SET r.temporada = row.temporada,
        r.ronda = row.ronda,
        r.nombre = row.nombre,
        r.fecha = row.fecha,
        r.hora = row.hora,
        r.descripcion = row.descripcion
    WITH r, row
    MATCH (t:Temporada {anio: row.temporada})
    MERGE (r)-[:PERTENECE_A]->(t)
    WITH r, row
    MATCH (c:Circuito {id: row.circuit_id})
    MERGE (r)-[:SE_CORRIO_EN]->(c)
    """
    print(f"[load_graph] Cargando {len(records)} carreras + relaciones Temporada/Circuito...")
    with driver.session() as session:
        for batch in tqdm(_batched(records), desc="Carreras"):
            _run_batch(session, q, batch)


# ─────────────────────────────────────────────────────────────
#  Relaciones
# ─────────────────────────────────────────────────────────────

def load_participaciones(driver: Driver, dfs: dict) -> None:
    """Relación PARTICIPO_EN: Piloto → Carrera (con propiedades de resultado)."""
    df = dfs["results_enriched"]
    records = []
    for _, row in df.iterrows():
        records.append({
            "driver_id":          str(row["driver_id"]),
            "race_id":            int(row["race_id"]),
            "constructor_id":     str(row["constructor_id"]),
            "posicion_final":     row.get("posicion_final"),
            "posicion_parrilla":  row.get("posicion_parrilla"),
            "puntos":             float(row.get("puntos", 0) or 0),
            "vueltas":            row.get("vueltas"),
            "posicion_texto":     str(row.get("posicion_texto", "")) if row.get("posicion_texto") else None,
            "ranking_vr":         row.get("ranking_vuelta_rapida"),
            "status_id":          row.get("status_id"),
            "motor":              str(row.get("motor", "Desconocido")),
            "temporada":          int(row.get("temporada", 0)),
        })

    q = """
    UNWIND $rows AS row
    MATCH (p:Piloto {id: row.driver_id})
    MATCH (c:Carrera {id: row.race_id})
    MATCH (e:Escuderia {id: row.constructor_id})
    MERGE (p)-[r:PARTICIPO_EN {race_id: row.race_id}]->(c)
    SET r.posicion_final    = row.posicion_final,
        r.posicion_parrilla = row.posicion_parrilla,
        r.puntos            = row.puntos,
        r.vueltas           = row.vueltas,
        r.posicion_texto    = row.posicion_texto,
        r.ranking_vr        = row.ranking_vr,
        r.motor             = row.motor
    WITH p, e, row
    MERGE (p)-[:CORRIO_PARA {temporada: row.temporada}]->(e)
    """
    print(f"[load_graph] Cargando {len(records)} participaciones (PARTICIPO_EN + CORRIO_PARA)...")
    with driver.session() as session:
        for batch in tqdm(_batched(records), desc="Participaciones"):
            _run_batch(session, q, batch)


def load_clasificaciones(driver: Driver, dfs: dict) -> None:
    """Relación CLASIFICO_EN: Piloto → Carrera."""
    if dfs.get("qualifying") is None:
        return
    df = dfs["qualifying"]
    records = []
    for _, row in df.iterrows():
        records.append({
            "driver_id":  str(row["driver_id"]),
            "race_id":    int(row["race_id"]),
            "posicion":   row.get("posicion"),
            "q1":         str(row.get("q1", "")) or None,
            "q2":         str(row.get("q2", "")) or None,
            "q3":         str(row.get("q3", "")) or None,
        })

    q = """
    UNWIND $rows AS row
    MATCH (p:Piloto {id: row.driver_id})
    MATCH (c:Carrera {id: row.race_id})
    MERGE (p)-[r:CLASIFICO_EN {race_id: row.race_id}]->(c)
    SET r.posicion = row.posicion,
        r.q1 = row.q1,
        r.q2 = row.q2,
        r.q3 = row.q3
    """
    print(f"[load_graph] Cargando {len(records)} clasificaciones...")
    with driver.session() as session:
        for batch in tqdm(_batched(records), desc="Clasificaciones"):
            _run_batch(session, q, batch)


def load_motores_escuderias(driver: Driver, dfs: dict) -> None:
    """Relación USA_MOTOR: Escudería → Motor."""
    df = dfs["motores"]
    # Necesitamos mapear nombre de escudería a constructor_id
    constructors = dfs["constructors"]
    nombre_to_id = dict(zip(constructors["nombre"], constructors["constructor_id"]))

    records = []
    for _, row in df.iterrows():
        cid = nombre_to_id.get(str(row["escuderia"]))
        if cid:
            records.append({
                "constructor_id":    str(cid),
                "motor":             str(row["motor"]),
                "temporada_inicio":  int(row["temporada_inicio"]),
                "temporada_fin":     int(row["temporada_fin"]),
            })

    q = """
    UNWIND $rows AS row
    MATCH (e:Escuderia {id: row.constructor_id})
    MATCH (m:Motor {fabricante: row.motor})
    MERGE (e)-[r:USA_MOTOR {temporada_inicio: row.temporada_inicio, temporada_fin: row.temporada_fin}]->(m)
    """
    print(f"[load_graph] Cargando {len(records)} relaciones Motor↔Escudería...")
    with driver.session() as session:
        for batch in _batched(records):
            _run_batch(session, q, batch)


def load_estados_resultado(driver: Driver, dfs: dict) -> None:
    """Relación TERMINO_CON: Piloto → Estado (por carrera)."""
    if dfs.get("status") is None:
        return
    df = dfs["results_enriched"]
    df_status = df[df["status_id"].notna()][["driver_id", "race_id", "status_id"]].copy()
    df_status = df_status.dropna(subset=["status_id"])

    records = [
        {
            "driver_id": str(r["driver_id"]),
            "race_id":   int(r["race_id"]),
            "status_id": int(r["status_id"]),
        }
        for _, r in df_status.iterrows()
    ]

    q = """
    UNWIND $rows AS row
    MATCH (p:Piloto {id: row.driver_id})
    MATCH (s:Estado {id: row.status_id})
    MATCH (c:Carrera {id: row.race_id})
    MERGE (p)-[:TERMINO_CON {race_id: row.race_id}]->(s)
    """
    print(f"[load_graph] Cargando {len(records)} estados de resultado...")
    with driver.session() as session:
        for batch in tqdm(_batched(records), desc="Estados"):
            _run_batch(session, q, batch)


def load_pit_stops(driver: Driver, dfs: dict) -> None:
    """Relación HIZO_PARADA: Piloto → Carrera."""
    if dfs.get("pit_stops") is None:
        return
    df = dfs["pit_stops"]
    records = []
    for _, row in df.iterrows():
        records.append({
            "driver_id":     str(row["driver_id"]),
            "race_id":       int(row["race_id"]),
            "numero_parada": int(row.get("numero_parada", 0) or 0),
            "vuelta":        int(row.get("vuelta", 0) or 0),
            "duracion":      str(row.get("duracion", "")) or None,
            "milisegundos":  int(row.get("milisegundos", 0) or 0) if pd.notna(row.get("milisegundos")) else None,
        })

    q = """
    UNWIND $rows AS row
    MATCH (p:Piloto {id: row.driver_id})
    MATCH (c:Carrera {id: row.race_id})
    MERGE (p)-[r:HIZO_PARADA {race_id: row.race_id, numero_parada: row.numero_parada}]->(c)
    SET r.vuelta = row.vuelta,
        r.duracion = row.duracion,
        r.milisegundos = row.milisegundos
    """
    print(f"[load_graph] Cargando {len(records)} pit stops...")
    with driver.session() as session:
        for batch in tqdm(_batched(records), desc="Pit stops"):
            _run_batch(session, q, batch)


def load_posiciones_campeonato(driver: Driver, dfs: dict) -> None:
    """Relaciones POSICION_CAMPEONATO y POSICION_CONSTRUCTORES."""
    # Por simplicidad, F1DB ya nos da el resultado anual por temporada
    if dfs.get("driver_standings") is not None:
        ds = dfs["driver_standings"]
        records = []
        for _, row in ds.iterrows():
            pos = row.get("posicion") or row.get("position")
            records.append({
                "driver_id":  str(row["driver_id"]),
                "temporada":  int(row["temporada"]),
                "posicion":   int(pos) if pd.notna(pos) else None,
                "puntos":     float(row.get("puntos", 0) or 0),
                "victorias":  int(row.get("victorias", 0) or 0),
            })

        q = """
        UNWIND $rows AS row
        MATCH (p:Piloto {id: row.driver_id})
        MATCH (t:Temporada {anio: row.temporada})
        MERGE (p)-[r:POSICION_CAMPEONATO {temporada: row.temporada}]->(t)
        SET r.posicion = row.posicion,
            r.puntos = row.puntos,
            r.victorias = row.victorias
        """
        print(f"[load_graph] Cargando {len(records)} posiciones de campeonato de pilotos...")
        with driver.session() as session:
            for batch in tqdm(_batched(records), desc="Campeonato pilotos"):
                _run_batch(session, q, batch)

    if dfs.get("constructor_standings") is not None:
        cs = dfs["constructor_standings"]
        records_cs = []
        for _, row in cs.iterrows():
            pos = row.get("posicion") or row.get("position")
            records_cs.append({
                "constructor_id": str(row["constructor_id"]),
                "temporada":      int(row["temporada"]),
                "posicion":       int(pos) if pd.notna(pos) else None,
                "puntos":         float(row.get("puntos", 0) or 0),
                "victorias":      int(row.get("victorias", 0) or 0),
            })

        q_cs = """
        UNWIND $rows AS row
        MATCH (e:Escuderia {id: row.constructor_id})
        MATCH (t:Temporada {anio: row.temporada})
        MERGE (e)-[r:POSICION_CONSTRUCTORES {temporada: row.temporada}]->(t)
        SET r.posicion = row.posicion,
            r.puntos = row.puntos,
            r.victorias = row.victorias
        """
        print(f"[load_graph] Cargando {len(records_cs)} posiciones de campeonato de constructores...")
        with driver.session() as session:
            for batch in tqdm(_batched(records_cs), desc="Campeonato constructores"):
                _run_batch(session, q_cs, batch)


# ─────────────────────────────────────────────────────────────
#  Función principal
# ─────────────────────────────────────────────────────────────

def load_graph(dfs: dict) -> None:
    """
    Carga el grafo completo en Neo4j en el siguiente orden:
    Schema → Nodos → Relaciones
    """
    driver = get_driver()
    t0 = time.time()

    try:
        setup_schema(driver)

        # ── Nodos ─────────────────────────────────────────
        load_temporadas(driver, dfs)
        load_pilotos(driver, dfs)
        load_escuderias(driver, dfs)
        load_motores(driver, dfs)
        load_circuitos(driver, dfs)
        load_estados(driver, dfs)
        load_carreras(driver, dfs)

        # ── Relaciones ────────────────────────────────────
        load_participaciones(driver, dfs)
        load_clasificaciones(driver, dfs)
        load_motores_escuderias(driver, dfs)
        load_estados_resultado(driver, dfs)
        load_pit_stops(driver, dfs)
        load_posiciones_campeonato(driver, dfs)

    finally:
        driver.close()

    elapsed = time.time() - t0
    print(f"\n[load_graph] ✓ Grafo cargado en {elapsed:.1f}s")

    # Verificación rápida
    _verify_graph()


def _verify_graph() -> None:
    driver = get_driver()
    print("\n[load_graph] Verificación del grafo:")
    counts_query = """
    CALL apoc.meta.stats() YIELD labels
    RETURN labels
    """
    simple_counts = {
        "Piloto":    "MATCH (n:Piloto) RETURN count(n) AS n",
        "Escuderia": "MATCH (n:Escuderia) RETURN count(n) AS n",
        "Carrera":   "MATCH (n:Carrera) RETURN count(n) AS n",
        "Motor":     "MATCH (n:Motor) RETURN count(n) AS n",
        "Circuito":  "MATCH (n:Circuito) RETURN count(n) AS n",
        "PARTICIPO_EN":  "MATCH ()-[r:PARTICIPO_EN]->() RETURN count(r) AS n",
        "CORRIO_PARA":   "MATCH ()-[r:CORRIO_PARA]->() RETURN count(r) AS n",
        "USA_MOTOR":     "MATCH ()-[r:USA_MOTOR]->() RETURN count(r) AS n",
    }
    with driver.session() as session:
        for label, q in simple_counts.items():
            result = session.run(q).single()
            count = result["n"] if result else 0
            print(f"   {label:25s}: {count:>8,}")
    driver.close()
