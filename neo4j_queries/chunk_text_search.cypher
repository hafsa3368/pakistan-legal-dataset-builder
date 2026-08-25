// "bail" ki jagah apna keyword likho
MATCH (c:Case)-[:HAS_CHUNK]->(ch:Chunk)
WHERE ch.text CONTAINS "bail"
RETURN c.case_label, ch.chunk_index, ch.text LIMIT 10;
