"""
generate_embeddings.py
python generate_embeddings.py --json_dir "d:\hafsa_thesis material\supreme_court_scraper\extracted_text_clean"
-----------------------
.\qdrant.exe
STEP 1: Sirf embedding generation.

Kya karta hai:
1. Aapke repaired JSON files (repair_metadata.py ka output) padhta hai
2. Har chunk ka text Ollama (nomic-embed-text) se embedding banata hai
3. Embedding + metadata Qdrant mein store karta hai
4. Checkpoint rakhta hai taake beech mein ruk jaye to dobara shuru na karna pade

REQUIREMENTS (pehle ye install/run karein):
    pip install qdrant-client requests tqdm

    Ollama:
    ollama serve
    ollama pull nomic-embed-text

    Qdrant (.exe ya docker, jo bhi aapke paas ho):
    - agar .exe hai to bas usko run kar dein, default port 6333 pe chalta hai

USAGE:
    python generate_embeddings.py --json_dir "path/to/repaired/json/folder"
"""

import os
import json
import time
import hashlib
import argparse
import logging
import requests
from pathlib import Path
from tqdm import tqdm

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    PayloadSchemaType,
)

# ---------------------------------------------------------------------------
# CONFIG — abhi ke liye seedha yahan, baad mein .env mein move karenge
# ---------------------------------------------------------------------------
OLLAMA_URL = "http://localhost:11434/api/embeddings"
EMBED_MODEL = "nomic-embed-text"
EMBED_DIM = 768

QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
COLLECTION_NAME = "legal_chunks"

BATCH_SIZE = 50
CHECKPOINT_FILE = "embedding_checkpoint.json"
DEBUG_LOG_FILE = "embedding_debug.log"

MAX_RETRIES = 3
RETRY_DELAY = 5  # seconds

# ---------------------------------------------------------------------------
# LOGGING
# ---------------------------------------------------------------------------
logging.basicConfig(
    filename=DEBUG_LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
logging.getLogger().addHandler(console)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# REPAIRED FILES FILTER
# ---------------------------------------------------------------------------
def load_repaired_filenames(repaired_list_path: str):
    """
    repair_checkpoint.json ek simple JSON array hai jisme sirf un files
    ke naam hain jo repair ho chuki hain, e.g.:
    ["LHC_bail_1930_processed_141930.json", "LHC_bail_1931_processed_141931.json", ...]

    Ye function us list ko set mein load karta hai taake sirf inhi files
    ko embed karein, baaki (jo abhi repair nahi hui) skip ho jayen.
    """
    if not os.path.exists(repaired_list_path):
        log.error(f"Repaired files list not found: {repaired_list_path}")
        return set()

    with open(repaired_list_path, "r", encoding="utf-8") as f:
        names = json.load(f)

    return set(names)


# ---------------------------------------------------------------------------
# CHECKPOINT
# ---------------------------------------------------------------------------
def load_checkpoint():
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            return set(data.get("processed_ids", []))
    return set()


def save_checkpoint(processed_ids):
    with open(CHECKPOINT_FILE, "w", encoding="utf-8") as f:
        json.dump({"processed_ids": list(processed_ids)}, f)


# ---------------------------------------------------------------------------
# OLLAMA (with retry)
# ---------------------------------------------------------------------------
def is_ollama_alive():
    try:
        r = requests.get("http://localhost:11434", timeout=5)
        return r.status_code == 200
    except requests.exceptions.RequestException:
        return False


def get_embedding(text: str):
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.post(
                OLLAMA_URL,
                json={"model": EMBED_MODEL, "prompt": text},
                timeout=60,
            )
            resp.raise_for_status()
            return resp.json()["embedding"]
        except (requests.exceptions.RequestException, KeyError) as e:
            log.warning(f"Embedding attempt {attempt} failed: {e}")
            if not is_ollama_alive():
                log.warning("Ollama seems down, retrying...")
            time.sleep(RETRY_DELAY)
    log.error("Embedding failed after max retries, skipping this chunk.")
    return None


# ---------------------------------------------------------------------------
# STABLE POINT ID
# ---------------------------------------------------------------------------
def stable_point_id(chunk_id: str) -> int:
    """
    Python ka built-in hash() process-randomized hota hai (PYTHONHASHSEED),
    isliye har run mein alag point_id milta — same chunk dobara upsert
    hone ke bajaye duplicate ban jata. md5 se hamesha same, deterministic
    ID milta hai, chahe script kitni baar bhi chalayen.
    """
    digest = hashlib.md5(chunk_id.encode("utf-8")).hexdigest()
    return int(digest[:16], 16)  # first 16 hex chars -> 64-bit int, Qdrant-safe


# ---------------------------------------------------------------------------
# QDRANT SETUP
# ---------------------------------------------------------------------------
def setup_qdrant(client: QdrantClient):
    existing = [c.name for c in client.get_collections().collections]
    if COLLECTION_NAME not in existing:
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=EMBED_DIM, distance=Distance.COSINE),
        )
        log.info(f"Created Qdrant collection: {COLLECTION_NAME}")

    for field, schema in [
        ("court", PayloadSchemaType.KEYWORD),
        ("case_type", PayloadSchemaType.KEYWORD),
        ("year", PayloadSchemaType.INTEGER),
        ("case_number", PayloadSchemaType.KEYWORD),
        ("judge", PayloadSchemaType.KEYWORD),
        ("generated_name", PayloadSchemaType.KEYWORD),
    ]:
        try:
            client.create_payload_index(
                collection_name=COLLECTION_NAME,
                field_name=field,
                field_schema=schema,
            )
        except Exception:
            pass  # index already exists


# ---------------------------------------------------------------------------
# READ CHUNKS FROM YOUR ACTUAL JSON FORMAT
# ---------------------------------------------------------------------------
def iter_chunks(json_dir: str, repaired_filenames: set = None):
    """
    Matches repair_metadata.py output exactly:
    generated_name, actual_filename, court, case_type, year, case_number,
    date_of_order, judge, sections_cited, citations, parties,
    chunks: [{chunk_index, token_estimate, source_pages, text}, ...]

    Agar repaired_filenames diya gaya ho, to sirf unhi files ko process
    karega jo us set mein hain (baaki skip). Ye repair_checkpoint.json
    se match karta hai.
    """
    json_files = list(Path(json_dir).glob("*.json"))
    log.info(f"Found {len(json_files)} total JSON files in {json_dir}")

    if repaired_filenames is not None:
        json_files = [jf for jf in json_files if jf.name in repaired_filenames]
        log.info(f"Filtered to {len(json_files)} repaired files (per repair_checkpoint.json)")

    for jf in json_files:
        try:
            with open(jf, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            log.warning(f"Could not read {jf.name}: {e}")
            continue

        chunks = data.get("chunks", [])
        if not chunks:
            continue  # skip checkpoint files, failed_files.json etc.

        generated_name = data.get("generated_name", jf.stem)
        actual_filename = data.get("actual_filename", "")

        try:
            year_val = int(data.get("year", 0))
        except (ValueError, TypeError):
            year_val = 0

        for chunk in chunks:
            chunk_text = chunk.get("text", "").strip()
            if not chunk_text:
                continue

            chunk_index = chunk.get("chunk_index", 0)
            chunk_id = f"{jf.stem}_{chunk_index}"

            payload = {
                "chunk_id": chunk_id,
                "chunk_text": chunk_text,
                "generated_name": generated_name,
                "actual_filename": actual_filename,
                "file_path": str(jf),
                "court": data.get("court", "unknown"),
                "case_type": data.get("case_type", "unknown"),
                "case_number": data.get("case_number", ""),
                "date_of_order": data.get("date_of_order", ""),
                "year": year_val,
                "judge": data.get("judge", ""),
                "sections_cited": data.get("sections_cited", []),
                "citations": data.get("citations", []),
                "parties": data.get("parties", []),
                "chunk_index": chunk_index,
                "token_estimate": chunk.get("token_estimate", 0),
                "source_pages": chunk.get("source_pages", []),
            }
            yield chunk_id, chunk_text, payload


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def run(json_dir: str, repaired_list_path: str = None):
    processed_ids = load_checkpoint()
    log.info(f"Resuming with {len(processed_ids)} chunks already processed.")

    repaired_filenames = None
    if repaired_list_path:
        repaired_filenames = load_repaired_filenames(repaired_list_path)
        if not repaired_filenames:
            log.error("Repaired filenames list is empty. Nothing to process. Stopping.")
            return
        log.info(f"Loaded {len(repaired_filenames)} repaired filenames to process.")

    if not is_ollama_alive():
        log.error("Ollama is not running. Start it first: ollama serve")
        return

    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    setup_qdrant(client)

    batch_points, batch_ids = [], []
    newly_processed = set()

    # Summary counters
    stats = {"embedded": 0, "skipped_already_done": 0, "failed_embedding": 0, "failed_dimension": 0}

    # NOTE: iter_chunks() ek generator hai, list(...) mein convert NAHI kiya —
    # isse memory usage constant rehta hai chahe 59k+ docs ho ya kam.
    # tqdm ko total nahi pata hoga (progress bar sirf count dikhayega, % nahi),
    # ye trade-off hai jo bade dataset ke liye zaroori hai.
    for chunk_id, chunk_text, payload in tqdm(iter_chunks(json_dir, repaired_filenames), desc="Embedding chunks"):
        if chunk_id in processed_ids:
            stats["skipped_already_done"] += 1
            continue

        vector = get_embedding(chunk_text)
        if vector is None:
            stats["failed_embedding"] += 1
            continue

        if len(vector) != EMBED_DIM:
            log.error(
                f"Embedding dimension mismatch for {chunk_id}: "
                f"got {len(vector)}, expected {EMBED_DIM}. Skipping."
            )
            stats["failed_dimension"] += 1
            continue

        point_id = stable_point_id(chunk_id)
        batch_points.append(PointStruct(id=point_id, vector=vector, payload=payload))
        batch_ids.append(chunk_id)
        stats["embedded"] += 1

        if len(batch_points) >= BATCH_SIZE:
            _flush(client, batch_points, batch_ids, processed_ids, newly_processed)
            batch_points, batch_ids = [], []

    if batch_points:
        _flush(client, batch_points, batch_ids, processed_ids, newly_processed)

    # ---- Final summary ----
    log.info("=" * 50)
    log.info("EMBEDDING RUN SUMMARY")
    log.info(f"  Newly embedded chunks:        {stats['embedded']}")
    log.info(f"  Skipped (already processed):  {stats['skipped_already_done']}")
    log.info(f"  Failed (embedding API):       {stats['failed_embedding']}")
    log.info(f"  Failed (dimension mismatch):  {stats['failed_dimension']}")
    log.info(f"  Total chunks now in checkpoint: {len(processed_ids)}")
    log.info("=" * 50)


def _flush(client, batch_points, batch_ids, processed_ids, newly_processed):
    try:
        client.upsert(collection_name=COLLECTION_NAME, points=batch_points)
        processed_ids.update(batch_ids)
        newly_processed.update(batch_ids)
        save_checkpoint(processed_ids)
        log.info(f"Upserted batch of {len(batch_points)} points. Total: {len(processed_ids)}")
    except Exception as e:
        log.error(f"Qdrant upsert failed for batch: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate embeddings from repaired legal JSON files.")
    parser.add_argument("--json_dir", required=True, help="Folder containing your JSON files (repaired + unrepaired mixed)")
    parser.add_argument(
        "--repaired_list",
        default="repair_checkpoint.json",
        help="Path to repair_checkpoint.json (list of repaired filenames). "
             "Only these files will be embedded; pass empty string to disable filtering.",
    )
    args = parser.parse_args()

    repaired_list = args.repaired_list if args.repaired_list else None
    run(args.json_dir, repaired_list)