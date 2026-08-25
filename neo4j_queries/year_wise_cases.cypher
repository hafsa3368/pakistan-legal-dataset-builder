MATCH (c:Case)-[:DECIDED_IN]->(y:Year)
RETURN y.value, count(c) AS total ORDER BY y.value;
