MATCH (c:Case)-[:DECIDED_BY]->(j:Judge)
RETURN j.name, count(c) AS cases ORDER BY cases DESC LIMIT 20;
