"""
get_pipeline_stats.py

Pulls REAL, current statistics from Qdrant and Neo4j for Chapter 4
(Dataset and Processing Results, Knowledge Graph Results). Reuses
your existing qdrant_to_neo4j_similarity.py connection helpers --
does not modify any data, read-only.

Place this file in the same folder as qdrant_to_neo4j_similarity.py
and legal_answer.py.

USAGE:
    python get_pipeline_stats.py

Copy the printed output and paste it back into the thesis conversation --
those are the real numbers Chapter 4 will report.
"""
import json
import qdrant_to_neo4j_similarity as qn

COLLECTION_NAME = "legal_chunks"

NODE_LABELS = ["Case", "Court", "Judge", "Topic", "Year", "LawSection", "Party", "Citation", "Chunk"]
REL_TYPES = ["HEARD_IN", "DECIDED_BY", "HAS_TOPIC", "DECIDED_IN", "APPLIES", "INVOLVES", "CITES", "HAS_CHUNK", "SIMILAR_TO"]


def qdrant_stats():
    client = qn.get_qdrant_client()
    if client is None:
        print("Could not connect to Qdrant.")
        return None
    info = client.get_collection(COLLECTION_NAME)
    return {
        "collection": COLLECTION_NAME,
        "points_count": info.points_count,
        "vectors_count": getattr(info, "vectors_count", None),
        "status": str(info.status),
    }


def neo4j_stats():
    driver = qn.get_neo4j_driver()
    if driver is None:
        print("Could not connect to Neo4j.")
        return None

    node_counts = {}
    rel_counts = {}
    try:
        with driver.session() as session:
            for label in NODE_LABELS:
                result = session.run(f"MATCH (n:{label}) RETURN count(n) AS c")
                node_counts[label] = result.single()["c"]

            for rel in REL_TYPES:
                result = session.run(f"MATCH ()-[r:{rel}]->() RETURN count(r) AS c")
                rel_counts[rel] = result.single()["c"]

            total_cases = session.run("MATCH (c:Case) RETURN count(c) AS c").single()["c"]
            total_rels = session.run("MATCH ()-[r]->() RETURN count(r) AS c").single()["c"]
    finally:
        driver.close()

    return {
        "node_counts": node_counts,
        "relationship_counts": rel_counts,
        "total_case_nodes": total_cases,
        "total_relationships_all_types": total_rels,
    }


def main():
    print("=" * 60)
    print("QDRANT STATS")
    print("=" * 60)
    q = qdrant_stats()
    if q:
        print(json.dumps(q, indent=2))

    print("\n" + "=" * 60)
    print("NEO4J STATS")
    print("=" * 60)
    n = neo4j_stats()
    if n:
        print(json.dumps(n, indent=2))

    print("\n" + "=" * 60)
    print("Copy everything above and paste it back into the thesis chat.")
    print("=" * 60)


if __name__ == "__main__":
    main()