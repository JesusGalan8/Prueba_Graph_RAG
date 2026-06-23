from src.retrieval.text2cypher import Text2CypherRetriever
r = Text2CypherRetriever()
res = r._execute_cypher("MATCH (c:Carrera)-[:SE_CORRIO_EN]->(ci:Circuito) WHERE c.nombre CONTAINS 'Bahrain' RETURN c.nombre, ci.nombre")
print(res)
