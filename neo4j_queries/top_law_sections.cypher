MATCH (c:Case)-[:APPLIES]->(s:LawSection)
RETURN s.name, count(c) AS times_used ORDER BY times_used DESC LIMIT 20;
