"""
qdrant_to_neo4j_similarity.py

Purpose
-------
Similarity-linking script for Hafsa's Pakistani Legal Research Assistant.

This script does NOT:
  - regenerate embeddings
  - process JSON files
  - recreate the Qdrant collection
  - delete Qdrant points
  - delete/modify Neo4j nodes or existing relationships
  - modify existing case metadata (case_number, case_type, year, etc.)

It ONLY:
  1. Takes a user's legal problem as free text.
  2. Embeds it using Ollama (nomic-embed-text, 768-dim).
  3. Searches the EXISTING Qdrant collection "legal_chunks" (cosine similarity).
  4. Groups chunk-level results by case_id (payload["case_id"]).
  5. Matches each case_id to an EXISTING Neo4j (:Case {case_id: ...}) node,
     and always displays that node's own case_number (never a guess).
  6. Determines the Central Case:
       - If the user explicitly names a specific case (by case_id or
         case_number), THAT case is always used as the Central Case
         (manual override always wins).
       - Otherwise (a plain free-text legal query), the script AUTOMATICALLY
         picks the highest-similarity case that actually exists in Neo4j
         and uses it as the Central Case. This is a deliberate design
         choice: the pipeline flow is
             User Query -> Qdrant similarity search -> Top relevant case
             -> auto-selected as Central Case -> Neo4j -> SIMILAR_TO edges
             -> Graph
         Trade-off to be aware of: since no human confirms the anchor case,
         a high-scoring-but-off-topic chunk (e.g. a law-review article
         rather than an actual judgment) could occasionally become the
         Central Case. The script still only creates SIMILAR_TO edges
         between cases that genuinely exist in Neo4j, and never invents
         nodes or metadata — it just no longer requires a human to name
         the anchor.
  7. Creates/merges SIMILAR_TO relationships from the Central Case to the
     other retrieved cases that clear SIMILARITY_THRESHOLD.
  8. Prints a full report + suggests Cypher verification queries.

Run:
    python qdrant_to_neo4j_similarity.py
"""

import os
import sys
import json
import logging
from datetime import datetime, timezone
from dataclasses import dataclass, field
from typing import Optional

import requests
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse
from neo4j import GraphDatabase
from neo4j.exceptions import ServiceUnavailable, AuthError, Neo4jError

# ==================================================================
# CONFIGURATION
# ==================================================================
load_dotenv()  # loads variables from a .env file if present

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")

QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "legal_chunks")

TOP_K = int(os.getenv("TOP_K", "20"))
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.75"))

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")  # MUST come from env / .env, never hardcoded

# Store a short preview of matched chunk text (keep Neo4j properties small)
MATCHED_CHUNK_TEXT_MAX_CHARS = 200

# How much of the user's legal query to store on SIMILAR_TO relationships,
# so each edge is traceable back to the query that produced it.
QUERY_TEXT_MAX_CHARS = 300

# ==================================================================
# LOGGING
# ==================================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("qdrant_to_neo4j_similarity")


# ==================================================================
# DATA STRUCTURES
# ==================================================================
@dataclass
class CaseHit:
    """Represents one unique case_id retrieved from Qdrant, represented by
    its single highest-scoring matched chunk."""
    case_id: str
    score: float                       # representative score (max among its chunks)
    matched_chunk_id: str
    matched_chunk_score: float
    matched_chunk_text: str = ""
    case_number: Optional[str] = None
    court: Optional[str] = None
    judge: Optional[str] = None
    year: Optional[int] = None
    case_type: Optional[str] = None
    all_scores: list = field(default_factory=list)  # every chunk score for this case


# ==================================================================
# STEP 1 — CONNECTIVITY / SETUP CHECKS
# ==================================================================
def check_ollama() -> bool:
    try:
        r = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
        r.raise_for_status()
        return True
    except requests.exceptions.RequestException as e:
        log.error(f"Ollama is not reachable at {OLLAMA_URL}. Is it running? ({e})")
        return False


def get_query_embedding(text: str) -> Optional[list]:
    """Send the user's legal problem to Ollama and get a 768-dim embedding
    using ONLY nomic-embed-text. Does not touch the existing embedding
    generation pipeline used for the corpus."""
    try:
        resp = requests.post(
            f"{OLLAMA_URL}/api/embeddings",
            json={"model": EMBEDDING_MODEL, "prompt": text},
            timeout=60,
        )
        resp.raise_for_status()
        data = resp.json()
        embedding = data.get("embedding")
        if not embedding:
            log.error("Ollama returned no 'embedding' field in response.")
            return None
        if len(embedding) != 768:
            log.warning(
                f"Expected 768-dim embedding, got {len(embedding)}-dim. "
                f"Continuing anyway, but check EMBEDDING_MODEL config."
            )
        return embedding
    except requests.exceptions.RequestException as e:
        log.error(f"Failed to get embedding from Ollama ({EMBEDDING_MODEL}): {e}")
        return None


def get_qdrant_client() -> Optional[QdrantClient]:
    try:
        client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, timeout=15)
        # Confirm the collection exists (does NOT create/recreate it)
        collections = [c.name for c in client.get_collections().collections]
        if COLLECTION_NAME not in collections:
            log.error(
                f"Collection '{COLLECTION_NAME}' does not exist in Qdrant. "
                f"This script will NOT create it. Available: {collections}"
            )
            return None
        return client
    except Exception as e:
        log.error(f"Could not connect to Qdrant at {QDRANT_HOST}:{QDRANT_PORT} ({e})")
        return None


def get_neo4j_driver():
    if not NEO4J_PASSWORD:
        log.error(
            "NEO4J_PASSWORD is not set. Put it in a .env file or environment "
            "variable — it will not be hardcoded in this script."
        )
        return None
    try:
        driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
        driver.verify_connectivity()
        return driver
    except AuthError as e:
        log.error(f"Neo4j authentication failed: {e}")
        return None
    except ServiceUnavailable as e:
        log.error(f"Neo4j is not reachable at {NEO4J_URI}. Is it running? ({e})")
        return None
    except Exception as e:
        log.error(f"Unexpected error connecting to Neo4j: {e}")
        return None


# ==================================================================
# STEP 2 — QDRANT SEARCH + GROUPING
# ==================================================================
def search_qdrant(client: QdrantClient, query_vector: list) -> list:
    """Search Qdrant, supporting both old and new qdrant-client APIs."""
    try:
        if hasattr(client, "query_points"):
            response = client.query_points(
                collection_name=COLLECTION_NAME,
                query=query_vector,
                limit=TOP_K,
                with_payload=True,
            )
            return response.points
        else:
            return client.search(
                collection_name=COLLECTION_NAME,
                query_vector=query_vector,
                limit=TOP_K,
                with_payload=True,
            )
    except UnexpectedResponse as e:
        log.error(f"Qdrant search failed (bad request/response): {e}")
        return []
    except Exception as e:
        log.error(f"Qdrant search error: {e}")
        return []


def group_by_case_id(qdrant_results: list) -> dict:
    """Group raw Qdrant chunk hits by payload['case_id'].
    For each case_id, keep the highest-scoring chunk as representative,
    while tracking all scores seen for that case."""
    cases: dict[str, CaseHit] = {}

    for point in qdrant_results:
        payload = point.payload or {}
        case_id = payload.get("case_id")
        score = point.score

        if not case_id:
            log.warning(
                f"Skipping chunk '{payload.get('chunk_id', '?')}': "
                f"payload has no case_id."
            )
            continue

        chunk_text = payload.get("chunk_text", "") or ""
        chunk_text_preview = chunk_text[:MATCHED_CHUNK_TEXT_MAX_CHARS]

        if case_id not in cases:
            cases[case_id] = CaseHit(
                case_id=case_id,
                score=score,
                matched_chunk_id=payload.get("chunk_id", ""),
                matched_chunk_score=score,
                matched_chunk_text=chunk_text_preview,
                case_number=payload.get("case_number"),
                court=payload.get("court"),
                judge=payload.get("judge"),
                year=payload.get("year"),
                case_type=payload.get("case_type"),
                all_scores=[score],
            )
        else:
            existing = cases[case_id]
            existing.all_scores.append(score)
            if score > existing.score:
                # A stronger chunk for this same case was found — update
                # the representative score/chunk.
                existing.score = score
                existing.matched_chunk_id = payload.get("chunk_id", "")
                existing.matched_chunk_score = score
                existing.matched_chunk_text = chunk_text_preview

    return cases


# ==================================================================
# STEP 3 — NEO4J MATCHING + SIMILAR_TO CREATION
# ==================================================================
def resolve_central_case(driver, identifier: str) -> Optional[dict]:
    """Looks up a case the USER explicitly named (by case_id or case_number)
    in the existing Neo4j graph. Returns {"case_id", "case_number"} or None
    if nothing matches. Used only for the manual-override path."""
    query = """
    MATCH (c:Case)
    WHERE c.case_id = $identifier OR c.case_number = $identifier
    RETURN c.case_id AS case_id, c.case_number AS case_number
    LIMIT 1
    """
    try:
        with driver.session() as session:
            result = session.run(query, identifier=identifier)
            record = result.single()
            if record:
                return {"case_id": record["case_id"], "case_number": record["case_number"]}
    except Neo4jError as e:
        log.error(f"Neo4j error resolving central case '{identifier}': {e}")
    return None


def find_cases_in_neo4j(driver, case_ids: list) -> dict:
    """Return {case_id: {"exists": bool, "case_number": str|None}} for each
    case_id. The case_number returned here comes straight from the Neo4j
    Case node — this is the SINGLE authoritative source for case_number
    display. Does NOT create anything."""
    found = {}
    query = """
    UNWIND $case_ids AS cid
    OPTIONAL MATCH (c:Case {case_id: cid})
    RETURN cid AS case_id, c IS NOT NULL AS exists, c.case_number AS case_number
    """
    try:
        with driver.session() as session:
            result = session.run(query, case_ids=case_ids)
            for record in result:
                found[record["case_id"]] = {
                    "exists": record["exists"],
                    "case_number": record["case_number"],
                }
    except Neo4jError as e:
        log.error(f"Neo4j error while checking case existence: {e}")
    return found


def update_case_importance(driver, case_id: str, score: float, timestamp: str):
    """Updates node-importance properties for graph visualization sizing.
    Keeps the BEST score the case has ever achieved across all queries,
    plus how many times it has been matched. Never touches any other
    existing metadata field."""
    query = """
    MATCH (c:Case {case_id: $case_id})
    SET c.current_similarity_score = CASE
            WHEN c.current_similarity_score IS NULL OR $score > c.current_similarity_score
            THEN $score ELSE c.current_similarity_score END,
        c.times_matched = coalesce(c.times_matched, 0) + 1,
        c.last_queried_at = $timestamp
    """
    try:
        with driver.session() as session:
            session.run(query, case_id=case_id, score=score, timestamp=timestamp)
    except Neo4jError as e:
        log.warning(f"Could not update importance properties for {case_id}: {e}")


def create_similar_to(driver, central_case_id: str, other: CaseHit,
                       query_text: str, timestamp: str) -> bool:
    """MERGE a SIMILAR_TO relationship from the central case to a similar
    case, so re-running the script does not create duplicates.

    Every relationship records the legal query that (re)confirmed it, plus
    how many distinct queries have matched this pair. If the relationship
    already exists, the stored score/matched_chunk_id/matched_chunk_score
    are only overwritten when this query's evidence is STRONGER than what's
    already there. A later, weaker query can never downgrade a previously
    stronger match; it still updates times_matched/last_query_text so the
    edge shows it keeps getting reconfirmed.
    """
    query = """
    MATCH (a:Case {case_id: $central_id})
    MATCH (b:Case {case_id: $other_id})
    MERGE (a)-[r:SIMILAR_TO]->(b)
    ON CREATE SET
        r.score = $score,
        r.matched_chunk_id = $matched_chunk_id,
        r.matched_chunk_score = $matched_chunk_score,
        r.matched_chunk_text = $matched_chunk_text,
        r.source = "Qdrant",
        r.times_matched = 1,
        r.first_query_text = $query_text,
        r.last_query_text = $query_text,
        r.last_queried_at = $timestamp
    ON MATCH SET
        r.score = CASE WHEN $score > r.score THEN $score ELSE r.score END,
        r.matched_chunk_id = CASE WHEN $score > r.score THEN $matched_chunk_id ELSE r.matched_chunk_id END,
        r.matched_chunk_score = CASE WHEN $score > r.score THEN $matched_chunk_score ELSE r.matched_chunk_score END,
        r.matched_chunk_text = CASE WHEN $score > r.score THEN $matched_chunk_text ELSE r.matched_chunk_text END,
        r.times_matched = coalesce(r.times_matched, 0) + 1,
        r.last_query_text = $query_text,
        r.last_queried_at = $timestamp
    RETURN r
    """
    try:
        with driver.session() as session:
            result = session.run(
                query,
                central_id=central_case_id,
                other_id=other.case_id,
                score=other.score,
                matched_chunk_id=other.matched_chunk_id,
                matched_chunk_score=other.matched_chunk_score,
                matched_chunk_text=other.matched_chunk_text,
                query_text=query_text[:QUERY_TEXT_MAX_CHARS],
                timestamp=timestamp,
            )
            return result.single() is not None
    except Neo4jError as e:
        log.error(
            f"Neo4j error creating SIMILAR_TO {central_case_id} -> "
            f"{other.case_id}: {e}"
        )
        return False


# ==================================================================
# STEP 4 — REPORTING
# ==================================================================
def print_case_report(rank: int, case: CaseHit):
    print(f"\n{rank}.")
    print(f"Case ID: {case.case_id}")
    print(f"Case Number: {display_label(case)}")
    print(f"Similarity: {case.score:.4f}")
    print(f"Matched Chunk: {case.matched_chunk_id}")
    print(f"Court: {case.court or '(not available)'}")
    print(f"Judge: {case.judge or '(not available)'}")


def is_valid_case_number(value: Optional[str]) -> bool:
    """Reject chunk-text fragments that have leaked into the case_number
    field. This never modifies the stored value — it only decides what to
    DISPLAY."""
    if not value or not isinstance(value, str):
        return False
    v = value.strip()
    if len(v) < 3 or len(v) > 80:
        return False
    if not any(ch.isdigit() for ch in v):
        return False
    if v[0].islower():
        # Real case numbers start with a capital letter/abbreviation/digit;
        # a lowercase first letter strongly suggests a mid-sentence fragment.
        return False
    if len(v.split()) > 10:
        # Real case numbers are short; long word counts suggest prose.
        return False
    return True


def display_label(case: CaseHit) -> str:
    """Display value for a case_number.

    case.case_number must already be the AUTHORITATIVE value pulled from
    the Neo4j Case node (set in main() after the Neo4j lookup) — never the
    raw Qdrant payload value, and never synthesized from case_type/court/
    year/filename/chunk_text. If it's missing or fails validation, we show
    '(not available)' rather than inventing something.
    """
    if is_valid_case_number(case.case_number):
        return case.case_number.strip()
    return "(not available)"


# ==================================================================
# STEP 5 — VERIFICATION QUERIES TO PRINT
# ==================================================================
VERIFICATION_QUERIES = """
==================================================
NEO4J VERIFICATION QUERIES
==================================================

1. All SIMILAR_TO relationships:
----------------------------------------------------
MATCH (a:Case)-[r:SIMILAR_TO]->(b:Case)
RETURN a.case_id, a.case_number, b.case_id, b.case_number, r.score
ORDER BY r.score DESC;

2. Highest similarity relationships (top 10):
----------------------------------------------------
MATCH (a:Case)-[r:SIMILAR_TO]->(b:Case)
RETURN a.case_number AS central_case, b.case_number AS similar_case,
       r.score AS similarity
ORDER BY r.score DESC
LIMIT 10;

3. Central case and its related cases (replace the case_id):
----------------------------------------------------
MATCH (a:Case {case_id: "REPLACE_WITH_CASE_ID"})-[r:SIMILAR_TO]->(b:Case)
RETURN a, r, b
ORDER BY r.score DESC;

4. Case number + similarity score view:
----------------------------------------------------
MATCH (a:Case)-[r:SIMILAR_TO]->(b:Case)
RETURN a.case_number AS from_case, b.case_number AS to_case,
       r.score AS score, r.matched_chunk_id AS matched_chunk
ORDER BY score DESC;

5. Full graph visualization (SIMILAR_TO + existing legal relationships)
   for a given central case (replace the case_id):
----------------------------------------------------
MATCH (a:Case {case_id: "REPLACE_WITH_CASE_ID"})-[r:SIMILAR_TO]->(b:Case)
OPTIONAL MATCH (a)-[other_rel]-(x)
WHERE type(other_rel) <> "SIMILAR_TO"
RETURN a, r, b, other_rel, x;
"""


# ==================================================================
# MAIN PIPELINE
# ==================================================================
def main():
    print("=" * 60)
    print("Pakistani Legal Research Assistant — Similarity Linker")
    print("=" * 60)

    # --- Pre-flight checks -----------------------------------------
    if not check_ollama():
        print("\n[ABORT] Start Ollama (e.g. `ollama serve`) and try again.")
        sys.exit(1)

    qdrant_client = get_qdrant_client()
    if qdrant_client is None:
        print("\n[ABORT] Fix Qdrant connection/collection and try again.")
        sys.exit(1)

    neo4j_driver = get_neo4j_driver()
    if neo4j_driver is None:
        print("\n[ABORT] Fix Neo4j connection/credentials and try again.")
        sys.exit(1)

    # --- Get user's legal problem ------------------------------------
    print("\nEnter the legal problem (e.g. 'The accused is involved in a")
    print("criminal case and wants pre-arrest bail.'):\n")
    legal_query = input("> ").strip()

    if not legal_query:
        print("[ABORT] Empty query provided.")
        sys.exit(1)

    # --- Optional explicit central case (manual override) ------------
    print("\nIf you want to MANUALLY anchor this search around a SPECIFIC")
    print("known case, enter its case number or case_id now.")
    print("Otherwise, press Enter and the TOP-SCORING case found in Neo4j")
    print("will automatically become the Central Case:\n")
    central_identifier = input("> ").strip() or None

    print(f"\nLEGAL QUERY:\n{legal_query}\n")

    # --- Embed query --------------------------------------------------
    query_vector = get_query_embedding(legal_query)
    if query_vector is None:
        print("[ABORT] Could not generate query embedding.")
        sys.exit(1)

    # --- Search Qdrant -------------------------------------------------
    raw_results = search_qdrant(qdrant_client, query_vector)
    print("QDRANT RESULTS:")
    print(f"Number of chunks retrieved: {len(raw_results)}")

    if not raw_results:
        print("[INFO] No chunks retrieved. Try a different query or check TOP_K.")
        neo4j_driver.close()
        sys.exit(0)

    # --- Group by case_id ------------------------------------------------
    cases = group_by_case_id(raw_results)
    ranked_cases = sorted(cases.values(), key=lambda c: c.score, reverse=True)

    if not ranked_cases:
        print("\n[INFO] No valid case_id found in retrieved chunks. Nothing to link.")
        neo4j_driver.close()
        sys.exit(0)

    print(f"UNIQUE CASES: {len(ranked_cases)}")

    # --- Match against Neo4j FIRST, so case_number shown below is always the
    # authoritative Neo4j value, never a synthesized/Qdrant-payload one -------
    case_ids = [c.case_id for c in ranked_cases]
    neo4j_info = find_cases_in_neo4j(neo4j_driver, case_ids)

    for c in ranked_cases:
        info = neo4j_info.get(c.case_id)
        # Overwrite whatever came from the Qdrant payload with the Neo4j
        # Case node's own case_number — or None if not found/empty, which
        # display_label() will render as "(not available)".
        c.case_number = info["case_number"] if info and info["exists"] else None

    print("\nTOP RELEVANT CASES:")
    for i, case in enumerate(ranked_cases, start=1):
        print_case_report(i, case)

    found_cases = [c for c in ranked_cases if neo4j_info.get(c.case_id, {}).get("exists")]
    missing_cases = [c for c in ranked_cases if not neo4j_info.get(c.case_id, {}).get("exists")]

    print("\nNEO4J RESULTS:\n")
    print(f"Cases found in Neo4j: {len(found_cases)}")
    print(f"Cases missing from Neo4j: {len(missing_cases)}")
    if missing_cases:
        print("\nMissing case_ids (skipped, no fake nodes created):")
        for c in missing_cases:
            print(f"  - {c.case_id}")

    # --- Determine the Central Case -----------------------------------
    # Priority 1: user explicitly named a case -> always wins.
    # Priority 2: no manual identifier -> auto-select the highest-scoring
    #             case that actually exists in Neo4j (found_cases is
    #             already sorted by score, since it's filtered from
    #             ranked_cases which is sorted descending).
    central_case_id = None
    central_case_number = None
    auto_selected = False

    if central_identifier:
        central = resolve_central_case(neo4j_driver, central_identifier)
        if central is None:
            print(
                f"\n[INFO] No case found in Neo4j matching '{central_identifier}' "
                f"(checked both case_id and case_number). No Central Case shown, "
                f"no SIMILAR_TO relationships created."
            )
            neo4j_driver.close()
            print("\nDone.")
            return
        central_case_id = central["case_id"]
        central_case_number = central["case_number"]
    else:
        if not found_cases:
            print(
                "\n[INFO] No manual case was given, and none of the retrieved "
                "cases exist in Neo4j — cannot auto-select a Central Case. "
                "No SIMILAR_TO relationships created."
            )
            neo4j_driver.close()
            print("\nDone.")
            return

        top_case = found_cases[0]  # highest-scoring case that exists in Neo4j
        central_case_id = top_case.case_id
        central_case_number = top_case.case_number
        auto_selected = True
        print(
            f"\n[AUTO] No manual case provided — automatically selecting the "
            f"top-scoring case as the Central Case:"
        )
        print(f"        Case ID: {central_case_id}")
        print(f"        Case Number: {display_label(top_case)}")
        print(f"        Similarity: {top_case.score:.4f}")

    other_cases = [c for c in found_cases if c.case_id != central_case_id]
    if not other_cases:
        print(
            f"\n[INFO] Central case resolved ({central_case_id}), but none of "
            f"the retrieved cases (besides itself) were found in Neo4j — "
            f"nothing to link."
        )
        neo4j_driver.close()
        print("\nDone.")
        return

    query_timestamp = datetime.now(timezone.utc).isoformat()

    # Update importance properties for visualization
    update_case_importance(neo4j_driver, central_case_id, 1.0, query_timestamp)

    created_any = False
    kept_edges = []
    for other in other_cases:
        if other.score < SIMILARITY_THRESHOLD:
            continue

        update_case_importance(neo4j_driver, other.case_id, other.score, query_timestamp)

        success = create_similar_to(neo4j_driver, central_case_id, other,
                                     legal_query, query_timestamp)
        if success:
            created_any = True
            kept_edges.append(other)

    print("\nSIMILAR CASES:\n")
    print(f"Central Case{' (auto-selected)' if auto_selected else ' (manual)'}:")
    print(f"Case ID: {central_case_id}")
    print(f"Case Number: {central_case_number if is_valid_case_number(central_case_number) else '(not available)'}\n")
    print("Related Cases:\n")

    if kept_edges:
        for i, other in enumerate(kept_edges, start=1):
            print(f"{i}. {display_label(other)}")
            print(f"   SIMILARITY: {other.score:.4f}")
            print(f"   MATCHED CHUNK: {other.matched_chunk_id}\n")
    else:
        print(f"(none met SIMILARITY_THRESHOLD = {SIMILARITY_THRESHOLD})\n")

    print(VERIFICATION_QUERIES.replace("REPLACE_WITH_CASE_ID", central_case_id))

    neo4j_driver.close()
    print("\nDone.")


if __name__ == "__main__":
    main()