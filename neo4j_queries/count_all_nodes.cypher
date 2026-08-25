MATCH (n) RETURN labels(n)[0] AS label, count(*) AS total ORDER BY total DESC;
