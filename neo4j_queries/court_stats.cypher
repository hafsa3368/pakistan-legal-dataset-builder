MATCH (c:Case)-[:HEARD_IN]->(co:Court)
RETURN co.name, count(c) AS cases ORDER BY cases DESC;
