// case_id replace karo — ye legal_answer.py ki get_existing_similar_cases() wali query hai
MATCH (a:Case {case_id: "REPLACE_WITH_CASE_ID"})-[r:SIMILAR_TO]->(b:Case)
RETURN b.case_id AS case_id, b.case_number AS case_number, r.score AS score
ORDER BY r.score DESC
LIMIT 10;
