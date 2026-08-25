// case_id replace karo
MATCH (a:Case {case_id: "REPLACE_WITH_CASE_ID"})-[r:SIMILAR_TO]->(b:Case)
OPTIONAL MATCH (a)-[other_rel]-(x)
WHERE type(other_rel) <> "SIMILAR_TO"
RETURN a, r, b, other_rel, x;
