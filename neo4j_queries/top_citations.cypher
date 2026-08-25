MATCH (c:Case)-[:CITES]->(cit:Citation)
RETURN cit.citation_id, count(c) AS times_cited ORDER BY times_cited DESC LIMIT 20;
