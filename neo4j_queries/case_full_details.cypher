// case_id replace karo
MATCH (c:Case {case_id: "REPLACE_WITH_CASE_ID"})
OPTIONAL MATCH (c)-[r]->(x)
RETURN c, r, x;
