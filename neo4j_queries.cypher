// Total Nodes
MATCH (n)
RETURN count(n);

// Cases
MATCH (c:Case)
RETURN c
LIMIT 50;

// Judges
MATCH (c:Case)-[:DECIDED_BY]->(j:Judge)
RETURN c,j
LIMIT 50;

// Complete Graph
MATCH (n)-[r]->(m)
RETURN n,r,m
LIMIT 200;