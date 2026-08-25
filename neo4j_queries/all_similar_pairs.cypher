MATCH (a:Case)-[r:SIMILAR_TO]->(b:Case)
RETURN a.case_number AS from_case, b.case_number AS to_case, r.score AS score
ORDER BY score DESC;
