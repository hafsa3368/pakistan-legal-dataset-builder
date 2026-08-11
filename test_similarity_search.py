"""
test_similarity_search.py
==========================

READ-ONLY testing script for the Pakistani Legal Research RAG project.

Purpose
-------
This script ONLY performs semantic similarity search against an EXISTING
Qdrant collection ("legal_chunks"). It does NOT:
    - generate or regenerate embeddings for stored data
    - insert, update, or delete any Qdrant points
    - recreate or modify the Qdrant collection
    - touch Neo4j in any way

Workflow
--------
User query (text)
    -> Ollama "nomic-embed-text" -> 768-dim query embedding
    -> Qdrant "legal_chunks" collection -> cosine similarity search
    -> Top-K chunk results (sorted by score, highest first)
    -> Display chunk-level results + payload metadata
    -> Display unique case_id list (case-level rollup)

Run:
    python test_similarity_search.py
"""

import sys
import json
import urllib.request
import urllib.error

# --------------------------------------------------------------------------
# CONFIG
# --------------------------------------------------------------------------

QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
COLLECTION_NAME = "legal_chunks"

OLLAMA_HOST = "localhost"
OLLAMA_PORT = 11434
EMBED_MODEL = "nomic-embed-text"
EXPECTED_VECTOR_DIM = 768

DEFAULT_TOP_K = 10

QDRANT_BASE_URL = f"http://{QDRANT_HOST}:{QDRANT_PORT}"
OLLAMA_BASE_URL = f"http://{OLLAMA_HOST}:{OLLAMA_PORT}"

# Fields we want to show for every result, in display order.
DISPLAY_FIELDS = [
    "case_id",
    "case_number",
    "chunk_id",
    "actual_filename",
    "generated_name",
    "court",
    "case_type",
    "year",
    "judge",
    "date_of_order",
    "sections_cited",
    "parties",
    "source_pages",
    "chunk_index",
    "chunk_text",
]


# --------------------------------------------------------------------------
# SMALL HTTP HELPERS (using only the standard library, no extra install
# needed for these calls; qdrant-client is used separately below)
# --------------------------------------------------------------------------

def _http_post_json(url, payload, timeout=30):
    """POST JSON and return parsed JSON response. Raises on failure."""
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}, method="POST"
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _http_get_json(url, timeout=10):
    """GET JSON and return parsed JSON response. Raises on failure."""
    req = urllib.request.Request(url, method="GET")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


# --------------------------------------------------------------------------
# STEP 1: CHECK OLLAMA IS RUNNING AND MODEL IS AVAILABLE
# --------------------------------------------------------------------------

def check_ollama_available():
    """
    Verifies Ollama is running and that the nomic-embed-text model exists.
    Does NOT modify anything. Exits with a clear message on failure.
    """
    try:
        tags = _http_get_json(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
    except (urllib.error.URLError, ConnectionRefusedError, TimeoutError) as e:
        print("ERROR: Could not connect to Ollama.")
        print(f"       Is Ollama running at {OLLAMA_BASE_URL}?")
        print(f"       Details: {e}")
        sys.exit(1)

    model_names = [m.get("name", "") for m in tags.get("models", [])]
    # Model tags sometimes appear as "nomic-embed-text:latest"
    if not any(EMBED_MODEL in name for name in model_names):
        print(f"ERROR: Ollama model '{EMBED_MODEL}' was not found locally.")
        print(f"       Available models: {model_names}")
        print(f"       Run: ollama pull {EMBED_MODEL}")
        sys.exit(1)

    print(f"[OK] Ollama is running and '{EMBED_MODEL}' is available.")


def get_query_embedding(query_text):
    """
    Sends the query text to Ollama's embedding endpoint and returns a
    768-dimensional vector (list of floats).
    """
    try:
        result = _http_post_json(
            f"{OLLAMA_BASE_URL}/api/embeddings",
            {"model": EMBED_MODEL, "prompt": query_text},
            timeout=60,
        )
    except (urllib.error.URLError, ConnectionRefusedError, TimeoutError) as e:
        print("ERROR: Failed to reach Ollama while generating the query embedding.")
        print(f"       Details: {e}")
        sys.exit(1)

    embedding = result.get("embedding")
    if not embedding:
        print("ERROR: Ollama did not return an embedding for this query.")
        print(f"       Raw response: {result}")
        sys.exit(1)

    if len(embedding) != EXPECTED_VECTOR_DIM:
        print(
            f"ERROR: Embedding dimension mismatch. Expected "
            f"{EXPECTED_VECTOR_DIM}, got {len(embedding)}."
        )
        print("       This would not match the vectors stored in Qdrant.")
        sys.exit(1)

    return embedding


# --------------------------------------------------------------------------
# STEP 2: CHECK QDRANT COLLECTION (READ-ONLY)
# --------------------------------------------------------------------------

def check_qdrant_collection(client):
    """
    Verifies the Qdrant collection exists and prints its info.
    Does NOT create or modify the collection.
    """
    from qdrant_client.http.exceptions import UnexpectedResponse

    try:
        existing = [c.name for c in client.get_collections().collections]
    except Exception as e:
        print("ERROR: Could not connect to Qdrant.")
        print(f"       Is Qdrant running at {QDRANT_BASE_URL}?")
        print(f"       Details: {e}")
        sys.exit(1)

    if COLLECTION_NAME not in existing:
        print(f"ERROR: Collection '{COLLECTION_NAME}' does not exist in Qdrant.")
        print(f"       Existing collections found: {existing}")
        print("       This script will NOT create it. Aborting.")
        sys.exit(1)

    try:
        info = client.get_collection(COLLECTION_NAME)
    except UnexpectedResponse as e:
        print(f"ERROR: Failed to fetch collection info. Details: {e}")
        sys.exit(1)

    vector_params = info.config.params.vectors
    # vector_params can be a single VectorParams or a dict of named vectors
    if hasattr(vector_params, "size"):
        vec_size = vector_params.size
        distance = vector_params.distance
    else:
        # named vectors - just grab the first one for display
        first_name = list(vector_params.keys())[0]
        vec_size = vector_params[first_name].size
        distance = vector_params[first_name].distance

    points_count = info.points_count

    print("-" * 50)
    print("QDRANT COLLECTION STATUS")
    print("-" * 50)
    print(f"Collection name : {COLLECTION_NAME}")
    print(f"Points count    : {points_count}")
    print(f"Vector dimension: {vec_size}")
    print(f"Distance metric : {distance}")
    print("-" * 50)

    if vec_size != EXPECTED_VECTOR_DIM:
        print(
            f"WARNING: Stored vector dimension ({vec_size}) does not match "
            f"expected {EXPECTED_VECTOR_DIM}."
        )


# --------------------------------------------------------------------------
# STEP 3: RUN SEARCH (READ-ONLY)
# --------------------------------------------------------------------------

def run_similarity_search(client, query_vector, top_k):
    """
    Performs a read-only cosine similarity search against the existing
    Qdrant collection. Returns a list of scored points.

    Newer qdrant-client versions (1.10+) removed the old `.search()` method
    in favor of `.query_points()`. This function tries the new method first
    and falls back to the old one, so it works regardless of which
    qdrant-client version is installed. Both are READ-ONLY calls.
    """
    # Try the newer API first (qdrant-client >= 1.10 style)
    if hasattr(client, "query_points"):
        try:
            response = client.query_points(
                collection_name=COLLECTION_NAME,
                query=query_vector,
                limit=top_k,
                with_payload=True,
            )
            return response.points
        except Exception as e:
            print("ERROR: Qdrant search failed (query_points).")
            print(f"       Details: {e}")
            sys.exit(1)

    # Fall back to the older API (qdrant-client < 1.10 style)
    if hasattr(client, "search"):
        try:
            return client.search(
                collection_name=COLLECTION_NAME,
                query_vector=query_vector,
                limit=top_k,
                with_payload=True,
            )
        except Exception as e:
            print("ERROR: Qdrant search failed (search).")
            print(f"       Details: {e}")
            sys.exit(1)

    print("ERROR: Installed qdrant-client has neither 'query_points' nor 'search'.")
    print("       Try: pip install --upgrade qdrant-client")
    sys.exit(1)


# --------------------------------------------------------------------------
# STEP 4: DISPLAY RESULTS
# --------------------------------------------------------------------------

def safe_get(payload, field):
    """Return payload[field] or 'N/A' if missing/empty, without crashing."""
    value = payload.get(field, None)
    if value is None or value == "":
        return "N/A"
    if isinstance(value, list):
        if not value:
            return "N/A"
        return "\n".join(str(v) for v in value)
    return str(value)


def display_results(results):
    """
    Prints each chunk-level result in the required format.
    Also validates presence of case_id / chunk_id / chunk_text and warns
    (without crashing) if any are missing.
    """
    if not results:
        print("No results found for this query.")
        return

    for rank, point in enumerate(results, start=1):
        payload = point.payload or {}

        # Non-fatal warnings for missing critical fields
        if "case_id" not in payload:
            print(f"WARNING: Result rank {rank} is missing 'case_id'.")
        if "chunk_id" not in payload:
            print(f"WARNING: Result rank {rank} is missing 'chunk_id'.")
        if "chunk_text" not in payload:
            print(f"WARNING: Result rank {rank} is missing 'chunk_text'.")

        print("=" * 40)
        print(f"RESULT {rank}")
        print("=" * 40)
        print()
        print(f"Similarity Score: {point.score:.4f}")
        print()

        for field in DISPLAY_FIELDS:
            if field == "chunk_text":
                continue  # print last, separately
            label = field.replace("_", " ").title()
            print(f"{label}:")
            print(safe_get(payload, field))
            print()

        print("Text:")
        print(safe_get(payload, "chunk_text"))
        print()


def display_unique_cases(results):
    """
    Rolls up chunk-level results into a unique, order-preserving list of
    case_id values (one Case -> many Chunks).
    """
    seen = set()
    unique_case_ids = []

    for point in results:
        payload = point.payload or {}
        case_id = payload.get("case_id")
        if not case_id:
            continue
        if case_id not in seen:
            seen.add(case_id)
            unique_case_ids.append(case_id)

    print("UNIQUE SIMILAR CASES")
    print("-" * 21)
    if not unique_case_ids:
        print("N/A (no case_id values found in results)")
    else:
        for i, case_id in enumerate(unique_case_ids, start=1):
            print(f"{i}. {case_id}")
    print()
    print(
        f"({len(results)} chunk(s) matched, representing "
        f"{len(unique_case_ids)} unique case(s))"
    )


# --------------------------------------------------------------------------
# MAIN
# --------------------------------------------------------------------------

def main():
    from qdrant_client import QdrantClient

    print("Pakistani Legal Research RAG - Similarity Search Test (READ-ONLY)")
    print("=" * 70)
    print(
        "This script does NOT regenerate or modify your existing 40,400+ "
        "Qdrant vectors. It only reads/searches the existing collection."
    )
    print()

    # 1. Check Ollama
    check_ollama_available()

    # 2. Connect to Qdrant and verify collection (no creation, no writes)
    try:
        client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    except Exception as e:
        print("ERROR: Could not create Qdrant client.")
        print(f"       Details: {e}")
        sys.exit(1)

    check_qdrant_collection(client)

    # 3. Get user query
    query_text = input("\nEnter your legal query (e.g. 'pre-arrest bail in a murder case'): ").strip()
    if not query_text:
        print("ERROR: Empty query entered. Exiting.")
        sys.exit(1)

    top_k_input = input(f"Number of results to return [default {DEFAULT_TOP_K}]: ").strip()
    if top_k_input:
        try:
            top_k = int(top_k_input)
            if top_k <= 0:
                raise ValueError
        except ValueError:
            print(f"Invalid input, using default TOP_K = {DEFAULT_TOP_K}.")
            top_k = DEFAULT_TOP_K
    else:
        top_k = DEFAULT_TOP_K

    # 4. Embed the query
    print("\nGenerating query embedding via Ollama...")
    query_vector = get_query_embedding(query_text)
    print(f"[OK] Query embedded into a {len(query_vector)}-dimensional vector.")

    # 5. Search Qdrant (read-only)
    print(f"\nSearching '{COLLECTION_NAME}' for top {top_k} similar chunks...\n")
    results = run_similarity_search(client, query_vector, top_k)

    # 6. Display
    display_results(results)
    display_unique_cases(results)

    print("\nDone. No data was modified, inserted, or deleted in Qdrant.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted by user (Ctrl+C). Exiting cleanly. No data was modified.")
        sys.exit(0)