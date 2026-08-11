corpus (bail, criminal, civil cases) ke against test karne ke liye theek rahenge:

The accused is involved in a criminal case and wants pre-arrest bail.
A tenant is being illegally evicted by the landlord without proper notice.
Dispute over inheritance and division of property among siblings.
Application for anticipatory bail in a case under Section 497 CrPC.
Custody dispute between divorced parents over minor children.



Neo4j Browser mein ye query paste karo:

MATCH (c:Case)-[r:SIMILAR_TO]->(s:Case)
RETURN c, r, s
ORDER BY r.score DESC
LIMIT 50;

Agar SIMILAR_TO relationships create hui hain to graph mein Case → SIMILAR_TO → Case nodes nazar aayenge.

Agar specifically ek case ka graph dekhna hai, tumhare test wale case ke liye:

MATCH (c:Case {case_id: "civil_appeal_lhc_2020_2022LHC8835.pdf"})
OPTIONAL MATCH (c)-[r:SIMILAR_TO]->(s:Case)
RETURN c, r, s;
Aur ek useful query

Ye dekho Neo4j mein kitne Case nodes hain aur kitni SIMILAR_TO relationships hain:

MATCH (c:Case)
OPTIONAL MATCH ()-[r:SIMILAR_TO]->()
RETURN count(DISTINCT c) AS total_cases,
       count(r) AS total_similar_relationships;

MATCH ()-[r:SIMILAR_TO]->()
DELETE r;

Sirf SIMILAR_TO relationships delete hongi.



// Ye query chalao (saari SIMILAR_TO relationships dekhne ke liye):
MATCH (a:Case)-[r:SIMILAR_TO]->(b:Case)
WHERE a.case_id <> b.case_id
RETURN a, r, b
ORDER BY r.score DESC

// Agar sirf ek specific case ka subgraph dekhna ho (jaisa supervisor demo ke liye chahta hai), to:
cypher
MATCH (a:Case {case_id: "civil_appeal_lhc_2020_2021LHC2480.pdf"})-[r:SIMILAR_TO]->(b:Case)
RETURN a, r, b
ORDER BY r.score DESC

(case_id apna wala daal dena)

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