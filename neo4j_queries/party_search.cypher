// "State" ki jagah apna search term likho
MATCH (c:Case)-[:INVOLVES]->(p:Party)
WHERE p.name CONTAINS "State"
RETURN c.case_label, p.name LIMIT 25;
