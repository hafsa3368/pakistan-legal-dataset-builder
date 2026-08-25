// Court "Unknown" wale kitne cases
MATCH (c:Case {court: "Unknown"}) RETURN count(c);

// Chunks-less cases
MATCH (c:Case) WHERE NOT (c)-[:HAS_CHUNK]->() RETURN count(c);

// Total chunks
MATCH (ch:Chunk) RETURN count(ch);
