"""
community_detection.py — Detects communities using Leiden algorithm, 
generates summaries via Ollama, and stores them in Neo4j.
"""
import time
import networkx as nx
from neo4j import GraphDatabase
import ollama
from graspologic.partition import hierarchical_leiden
from tqdm import tqdm

from src.config import (
    NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD,
    OLLAMA_BASE_URL, OLLAMA_LLM_MODEL, OLLAMA_EMBED_MODEL,
    VECTOR_INDEX_COMMUNITY, VECTOR_DIMENSIONS
)

def get_neo4j_driver():
    return GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

def get_ollama_client():
    return ollama.Client(host=OLLAMA_BASE_URL)

def extract_graph_to_networkx(driver) -> tuple[nx.Graph, dict]:
    """Extrae el Domain Graph a NetworkX."""
    print("[community] Extrayendo grafo a NetworkX...")
    G = nx.Graph()
    node_data = {}
    
    with driver.session() as session:
        # Extraer nodos: Piloto, Escuderia, Carrera (solo los que tienen conexiones)
        q_nodes = """
        MATCH (n) 
        WHERE (n:Carrera) 
           OR (n:Piloto AND EXISTS { (n)-[:PARTICIPO_EN]->(:Carrera) })
           OR (n:Escuderia AND EXISTS { (n)<-[:CORRIO_PARA]-(:Piloto)-[:PARTICIPO_EN]->(:Carrera) })
        RETURN elementId(n) AS eid, labels(n)[0] AS label, n.nombre AS nombre, 
               n.apellido AS apellido, n.descripcion AS desc
        """
        for r in session.run(q_nodes):
            eid = r["eid"]
            name = r["nombre"] or r["apellido"] or ""
            G.add_node(eid)
            node_data[eid] = {
                "label": r["label"],
                "name": name,
                "desc": r["desc"]
            }
            
        # Extraer relaciones
        q_rels = """
        MATCH (a)-[r]->(b)
        WHERE (a:Piloto OR a:Escuderia OR a:Carrera) AND (b:Piloto OR b:Escuderia OR b:Carrera)
        RETURN elementId(a) AS source, elementId(b) AS target, type(r) AS type
        """
        for r in session.run(q_rels):
            # Ignorar múltiples aristas entre mismos nodos para Leiden simple
            if not G.has_edge(r["source"], r["target"]):
                G.add_edge(r["source"], r["target"], weight=1.0)
            else:
                G[r["source"]][r["target"]]['weight'] += 1.0

    print(f"[community] Extraídos {G.number_of_nodes()} nodos y {G.number_of_edges()} relaciones.")
    return G, node_data

def generate_community_summary(ollama_client, community_nodes: list, node_data: dict) -> str:
    """Genera un resumen para la comunidad usando el LLM."""
    # Limitamos la cantidad de texto para no saturar el LLM
    context_lines = []
    for eid in community_nodes[:50]: # Muestra los primeros 50 nodos máx
        data = node_data[eid]
        context_lines.append(f"- [{data['label']}] {data['name']}: {data['desc']}")
        
    context = "\n".join(context_lines)
    prompt = f"""
Eres un experto en Fórmula 1. Analiza los siguientes elementos (pilotos, escuderías, carreras) que han sido agrupados por un algoritmo de detección de comunidades porque tienen fuertes conexiones entre sí.
Escribe un resumen conciso (máximo 4 frases) explicando qué tienen en común o por qué forman un grupo (ej. época de dominio, equipo y sus pilotos, batallas clave).

Nodos de la comunidad:
{context}

Resumen:"""

    response = ollama_client.chat(
        model=OLLAMA_LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": 0.2}
    )
    return response["message"]["content"].strip()

def embed_text(client, text: str) -> list[float]:
    response = client.embeddings(model=OLLAMA_EMBED_MODEL, prompt=text)
    return response["embedding"]

def setup_community_index(driver):
    q = f"""
    CREATE VECTOR INDEX {VECTOR_INDEX_COMMUNITY} IF NOT EXISTS
    FOR (c:Comunidad) ON c.embedding
    OPTIONS {{
        indexConfig: {{
            `vector.dimensions`: {VECTOR_DIMENSIONS},
            `vector.similarity_function`: 'cosine'
        }}
    }}
    """
    with driver.session() as session:
        try:
            session.run(q)
            print(f"[community] ✓ Índice vectorial creado: {VECTOR_INDEX_COMMUNITY}")
        except Exception as e:
            print(f"[community] ⚠ Índice {VECTOR_INDEX_COMMUNITY}: {e}")

def detect_and_store_communities(force: bool = False):
    driver = get_neo4j_driver()
    ollama_client = get_ollama_client()
    
    setup_community_index(driver)
    
    if force:
        with driver.session() as session:
            session.run("MATCH (c:Comunidad) DETACH DELETE c")
            print("[community] Comunidades anteriores eliminadas.")
            
    G, node_data = extract_graph_to_networkx(driver)
    if G.number_of_nodes() == 0:
        print("[community] No hay nodos para analizar.")
        return
        
    print("[community] Ejecutando Hierarchical Leiden...")
    t0 = time.time()
    # Ejecuta Leiden
    hierarchical_partition = hierarchical_leiden(G)
    print(f"[community] Leiden terminado en {time.time()-t0:.1f}s.")
    
    # Agrupar nodos por comunidad
    communities = {}
    for partition in hierarchical_partition:
        level = partition.level
        cluster = partition.cluster
        node = partition.node
        
        comm_id = f"L{level}_C{cluster}"
        if comm_id not in communities:
            communities[comm_id] = {"level": level, "cluster": cluster, "nodes": []}
        communities[comm_id]["nodes"].append(node)
        
    print(f"[community] Encontradas {len(communities)} comunidades en total.")
    
    # Filtrar comunidades demasiado pequeñas
    valid_communities = {k: v for k, v in communities.items() if len(v["nodes"]) >= 5}
    print(f"[community] {len(valid_communities)} comunidades válidas (más de 5 nodos).")
    
    for comm_id, data in tqdm(valid_communities.items(), desc="Procesando Comunidades"):
        # 1. Generar resumen
        summary = generate_community_summary(ollama_client, data["nodes"], node_data)
        # 2. Generar embedding
        embedding = embed_text(ollama_client, summary)
        
        # 3. Guardar en Neo4j
        with driver.session() as session:
            # Crear nodo comunidad
            q_create = """
            MERGE (c:Comunidad {id: $comm_id})
            SET c.nivel = $level, c.resumen = $summary, c.embedding = $embedding
            """
            session.run(q_create, {
                "comm_id": comm_id, "level": data["level"], 
                "summary": summary, "embedding": embedding
            })
            
            # Enlazar nodos
            q_link = """
            MATCH (c:Comunidad {id: $comm_id})
            MATCH (n) WHERE elementId(n) IN $nodes
            MERGE (c)-[:CONTIENE]->(n)
            """
            session.run(q_link, {"comm_id": comm_id, "nodes": data["nodes"]})
            
    driver.close()
    print("[community] ✓ Proceso de comunidades finalizado.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    detect_and_store_communities(force=args.force)
