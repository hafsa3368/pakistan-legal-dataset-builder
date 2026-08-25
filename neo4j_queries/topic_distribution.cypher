MATCH (c:Case)-[:HAS_TOPIC]->(t:Topic)
RETURN t.name, count(c) AS total ORDER BY total DESC;
