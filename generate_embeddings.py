r"""
generate_embeddings.py

C:\Users\Hamza\Desktop\qdrant
python generate_embeddings.py --json_dir "d:\hafsa_thesis material\supreme_court_scraper\extracted_text_clean"
-----------------------
.\qdrant.exe

ROOT CAUSE FOUND (via test_single_embed.py):
    Ollama ka error tha: {"error":"the input length exceeds the context length"}
    Matlab lambe chunks (~7500+ chars) nomic-embed-text ke context window
    (num_ctx) se zyada bade the, isliye Ollama unhe "500 Internal Server Error"
    ke roop mein reject kar raha tha — memory crash ya corrupt Urdu text ka
    masla NAHI tha.

FIX (dono cheezein):
    1. Har embedding request mein explicitly num_ctx=8192 bhejte hain,
       taake Ollama apna default (jo chhota ho sakta hai) use na kare.
    2. Agar phir bhi "context length" error aaye (bohot bade chunks ke liye),
       to chunk ko 2 hisso mein split karte hain, dono ka embedding lete hain,
       aur unka average (normalized) le kar final vector banate hain.
       Zaroorat pade to recursively aur split hota hai (max depth tak).
       Ye translation se zyada simple/fast/accurate hai — text same language
       mein rehta hai, bas chota ho jata hai.

Kya karta hai:
1. Aapke repaired JSON files (repair_metadata.py ka output) padhta hai
2. Har chunk ka text Ollama (nomic-embed-text) se embedding banata hai
3. Agar chunk context limit se bada ho, to split-and-average fallback use karta hai
4. Embedding + metadata Qdrant mein store karta hai
5. Checkpoint rakhta hai taake beech mein ruk jaye to dobara shuru na karna pade

REQUIREMENTS (pehle ye install/run karein):
    pip install qdrant-client requests tqdm numpy

    Ollama:
    ollama serve
    ollama pull nomic-embed-text

    Qdrant (.exe ya docker, jo bhi aapke paas ho):
    - agar .exe hai to bas usko run kar dein, default port 6333 pe chalta hai

USAGE:
    python generate_embeddings.py --json_dir "path/to/repaired/json/folder"

    # failed chunks ko baad mein sirf unhi ko retry karne ke liye:
    python generate_embeddings.py --json_dir "..." --retry_failed_only
"""
import os
import json
import time
import hashlib
import argparse
import logging
import requests
import numpy as np
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
# CONFIG
# ---------------------------------------------------------------------------
OLLAMA_EMBED_URL = "http://localhost:11434/api/embeddings"
EMBED_MODEL = "nomic-embed-text"
EMBED_DIM = 768

# FIX: explicitly context window batate hain Ollama ko har request mein,
# taake wo apna (chhota) default use na kare.
NUM_CTX = 8192

QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
COLLECTION_NAME = "legal_chunks"

BATCH_SIZE = 50
CHECKPOINT_FILE = "embedding_checkpoint.json"
DEBUG_LOG_FILE = "embedding_debug.log"
DEFAULT_REPAIR_CHECKPOINT_FILE = "repair_checkpoint.json"

# Permanently-skip hone wale chunks yahan save honge (chunk_id + reason)
FAILED_CHUNKS_FILE = "failed_chunks.json"

MAX_RETRIES = 3
RETRY_DELAY = 5  # seconds, base delay for exponential backoff

# FIX: agar embedding "context length exceeded" error de, to chunk ko is
# se zyada baar split nahi karenge (taake infinite recursion na ho).
# 3 splits = chunk 8 hisso tak toot sakta hai worst case mein.
MAX_SPLIT_DEPTH = 3

# Bohot hi zyada bade chunks (bug se ban jayen) ko hard-cap karne ke liye —
# safety net, is se bada koi bhi single chunk nahi jaana chahiye.
MAX_CHUNK_CHARS = 20000

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


def load_failed_chunks():
    if os.path.exists(FAILED_CHUNKS_FILE):
        with open(FAILED_CHUNKS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)  # dict: {chunk_id: reason}
    return {}


def save_failed_chunks(failed_dict):
    with open(FAILED_CHUNKS_FILE, "w", encoding="utf-8") as f:
        json.dump(failed_dict, f, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------------
# OLLAMA HELPERS
# ---------------------------------------------------------------------------
def is_ollama_alive():
    try:
        r = requests.get("http://localhost:11434", timeout=5)
        return r.status_code == 200
    except requests.exceptions.RequestException:
        return False


def _raw_embed_request(text: str):
    """
    Ollama ko ek hi request bhejta hai. Return: (vector_or_None, error_message_or_None)
    error_message mein Ollama ka asli reason hota hai (e.g. "context length exceeded"),
    generic "500" nahi — isse hum decide kar sakte hain ke split karna hai ya
    sirf retry karna hai.
    """
    try:
        resp = requests.post(
            OLLAMA_EMBED_URL,
            json={
                "model": EMBED_MODEL,
                "prompt": text,
                "options": {"num_ctx": NUM_CTX},
            },
            timeout=90,
        )
        if resp.status_code == 200:
            return resp.json()["embedding"], None
        # Ollama error body mein asli reason hota hai
        try:
            err_msg = resp.json().get("error", resp.text)
        except ValueError:
            err_msg = resp.text
        return None, err_msg
    except requests.exceptions.RequestException as e:
        return None, str(e)


def _average_vectors(vectors):
    """Multiple embeddings ko average karke ek normalized vector banata hai."""
    arr = np.array(vectors, dtype=np.float64)
    mean_vec = arr.mean(axis=0)
    norm = np.linalg.norm(mean_vec)
    if norm > 0:
        mean_vec = mean_vec / norm
    return mean_vec.tolist()


def get_embedding_with_split(text: str, chunk_id: str, depth: int = 0):
    """
    Pehle poora text embed karne ki koshish karta hai (retries ke sath).
    Agar Ollama "context length" error de, to text ko 2 hisso mein split
    karke recursively dono ka embedding leta hai, phir average kar deta hai.
    Doosre errors (network, timeout, etc.) ke liye normal retry hi karta hai,
    split nahi karta — kyunki split se wo masla theek nahi hoga.

    Return: (vector, was_split) tuple. vector None ho sakta hai agar sab
    kuch fail ho jaye.
    """
    last_error = None
    for attempt in range(1, MAX_RETRIES + 1):
        vector, error = _raw_embed_request(text)
        if vector is not None:
            return vector, (depth > 0)

        last_error = error or ""
        is_context_error = "context length" in last_error.lower()

        if is_context_error:
            # Retry karne ka koi fayda nahi — text hamesha itna hi bada rahega.
            # Seedha split logic pe jao.
            break

        # Doosre errors (network/500/timeout) — normal exponential backoff retry
        wait = RETRY_DELAY * (2 ** (attempt - 1))
        log.warning(f"[{chunk_id}] Embedding attempt {attempt} failed: {last_error} (waiting {wait}s)")
        if not is_ollama_alive():
            log.warning(f"[{chunk_id}] Ollama seems down, retrying...")
        time.sleep(wait)
    else:
        # Saare retries khatam, koi context error nahi tha, phir bhi fail —
        # normal failure
        log.error(f"[{chunk_id}] Embedding failed after max retries: {last_error}")
        return None, False

    # Yahan pahunche matlab context-length error mila
    if depth >= MAX_SPLIT_DEPTH:
        log.error(f"[{chunk_id}] Context length exceeded even after {depth} splits, giving up.")
        return None, False

    log.warning(
        f"[{chunk_id}] Context length exceeded ({len(text)} chars) — "
        f"splitting into 2 halves (split depth {depth + 1})."
    )

    mid = len(text) // 2
    # Word boundary ke qareeb split karo taake beech mein koi word na kate
    split_point = text.rfind(" ", 0, mid)
    if split_point == -1:
        split_point = mid

    first_half = text[:split_point].strip()
    second_half = text[split_point:].strip()

    vec1, _ = get_embedding_with_split(first_half, f"{chunk_id}_splitA", depth + 1)
    vec2, _ = get_embedding_with_split(second_half, f"{chunk_id}_splitB", depth + 1)

    if vec1 is None or vec2 is None:
        log.error(f"[{chunk_id}] One or both split halves failed to embed.")
        return None, False

    averaged = _average_vectors([vec1, vec2])
    log.info(f"[{chunk_id}] Embedding succeeded via split-and-average (depth {depth + 1}).")
    return averaged, True


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
def iter_chunks(json_dir: str, repaired_filenames: set = None, exclude_filenames: set = None):
    """
    Matches repair_metadata.py output exactly:
    generated_name, actual_filename, court, case_type, year, case_number,
    date_of_order, judge, sections_cited, citations, parties,
    chunks: [{chunk_index, token_estimate, source_pages, text}, ...]

    Agar repaired_filenames diya gaya ho, to sirf unhi files ko process
    karega jo us set mein hain (baaki skip). Ye repair_checkpoint.json
    se match karta hai.

    exclude_filenames (default: repair_checkpoint.json's files, see run())
    ko is dir mein filter hone se PEHLE hi skip kar diya jata hai -- inko
    khud JSON file kholna/parhna bhi nahi padta, kyunke ye already Qdrant
    mein hain (embedding_checkpoint.json ke through pehle se hi verify
    ho chuka hai). Ye large-scale reruns ko fast start karne deta hai.
    """
    json_files = list(Path(json_dir).glob("*.json"))
    log.info(f"Found {len(json_files)} total JSON files in {json_dir}")

    if repaired_filenames is not None:
        json_files = [jf for jf in json_files if jf.name in repaired_filenames]
        log.info(f"Filtered to {len(json_files)} repaired files (per repair_checkpoint.json)")

    if exclude_filenames:
        before = len(json_files)
        json_files = [jf for jf in json_files if jf.name not in exclude_filenames]
        log.info(
            f"Skipping {before - len(json_files)} file(s) already listed in "
            f"repair_checkpoint.json (already embedded) -- {len(json_files)} remaining."
        )

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

            # Bohot hi zyada bade chunks ke liye hard safety cap
            if len(chunk_text) > MAX_CHUNK_CHARS:
                log.warning(
                    f"[{chunk_id}] Chunk text too long ({len(chunk_text)} chars), "
                    f"truncating to {MAX_CHUNK_CHARS}."
                )
                chunk_text = chunk_text[:MAX_CHUNK_CHARS]

            payload = {
                "chunk_id": chunk_id,
                "chunk_text": chunk_text,
                "generated_name": generated_name,
                "actual_filename": actual_filename,
                # Same derivation add_case_id_to_qdrant.py used to apply as a
                # separate post-processing pass -- set directly at embed time
                # instead, so newly-embedded points never need that extra step.
                "case_id": actual_filename if actual_filename else generated_name,
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
def run(
    json_dir: str,
    repaired_list_path: str = None,
    retry_failed_only: bool = False,
    skip_already_embedded: bool = True,
):
    processed_ids = load_checkpoint()
    log.info(f"Resuming with {len(processed_ids)} chunks already processed.")

    failed_chunks = load_failed_chunks()
    retry_target_ids = set(failed_chunks.keys()) if retry_failed_only else None
    if retry_failed_only:
        log.info(f"Retry-only mode: {len(retry_target_ids)} previously failed chunks loaded.")
        if not retry_target_ids:
            log.info("No failed chunks to retry. Exiting.")
            return

    repaired_filenames = None
    if repaired_list_path:
        repaired_filenames = load_repaired_filenames(repaired_list_path)
        if not repaired_filenames:
            log.error("Repaired filenames list is empty. Nothing to process. Stopping.")
            return
        log.info(f"Loaded {len(repaired_filenames)} repaired filenames to process.")

    # By default, skip files listed in repair_checkpoint.json -- they were
    # already embedded in an earlier run (verified: embedding_checkpoint.json
    # already covers them), so re-opening/re-parsing those JSON files just to
    # have the per-chunk checkpoint check discard them again wastes time on
    # large reruns. Same convention as repair_metadata.py's --no-skip-repaired.
    exclude_filenames = None
    if skip_already_embedded and not repaired_list_path:
        exclude_filenames = load_repaired_filenames(DEFAULT_REPAIR_CHECKPOINT_FILE)

    if not is_ollama_alive():
        log.error("Ollama is not running. Start it first: ollama serve")
        return

    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    setup_qdrant(client)

    batch_points, batch_ids = [], []
    newly_processed = set()

    stats = {
        "embedded": 0,
        "embedded_via_split": 0,
        "skipped_already_done": 0,
        "failed_embedding": 0,
        "failed_dimension": 0,
    }

    for chunk_id, chunk_text, payload in tqdm(iter_chunks(json_dir, repaired_filenames, exclude_filenames), desc="Embedding chunks"):
        if chunk_id in processed_ids:
            stats["skipped_already_done"] += 1
            continue

        # retry-only mode mein sirf wahi chunks process karo jo pehle fail hue thay
        if retry_target_ids is not None and chunk_id not in retry_target_ids:
            continue

        # Chunk pehle se permanently fail ho chuka hai aur ye normal run hai
        # (retry-only nahi) — to skip karo, taake har normal run mein
        # dobara wahi crash na ho aur baaki chunks block na hon.
        if retry_target_ids is None and chunk_id in failed_chunks:
            stats["skipped_already_done"] += 1
            continue

        vector, was_split = get_embedding_with_split(chunk_text, chunk_id)
        if vector is None:
            stats["failed_embedding"] += 1
            failed_chunks[chunk_id] = "embedding_failed"
            save_failed_chunks(failed_chunks)
            continue

        if len(vector) != EMBED_DIM:
            log.error(
                f"Embedding dimension mismatch for {chunk_id}: "
                f"got {len(vector)}, expected {EMBED_DIM}. Skipping."
            )
            stats["failed_dimension"] += 1
            failed_chunks[chunk_id] = f"dimension_mismatch_{len(vector)}"
            save_failed_chunks(failed_chunks)
            continue

        # Agar chunk pehle failed_chunks mein tha aur ab kaamyab ho gaya,
        # to usko failed list se hata do
        if chunk_id in failed_chunks:
            del failed_chunks[chunk_id]
            save_failed_chunks(failed_chunks)

        payload["was_split_for_embedding"] = was_split

        point_id = stable_point_id(chunk_id)
        batch_points.append(PointStruct(id=point_id, vector=vector, payload=payload))
        batch_ids.append(chunk_id)
        stats["embedded"] += 1
        if was_split:
            stats["embedded_via_split"] += 1

        if len(batch_points) >= BATCH_SIZE:
            _flush(client, batch_points, batch_ids, processed_ids, newly_processed)
            batch_points, batch_ids = [], []

    if batch_points:
        _flush(client, batch_points, batch_ids, processed_ids, newly_processed)

    # ---- Final summary ----
    log.info("=" * 50)
    log.info("EMBEDDING RUN SUMMARY")
    log.info(f"  Newly embedded chunks:              {stats['embedded']}")
    log.info(f"    (of which via split-and-average): {stats['embedded_via_split']}")
    log.info(f"  Skipped (already processed/failed): {stats['skipped_already_done']}")
    log.info(f"  Failed (embedding):                 {stats['failed_embedding']}")
    log.info(f"  Failed (dimension mismatch):        {stats['failed_dimension']}")
    log.info(f"  Total chunks now in checkpoint:     {len(processed_ids)}")
    log.info(f"  Total chunks still in failed_chunks.json: {len(failed_chunks)}")
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
        default=None,
        help="Optional: path to a JSON file listing filenames to restrict "
             "embedding to (e.g. for re-running only a historical repair "
             "batch). By default (omitted) ALL files in --json_dir are "
             "embedded, since the embedding checkpoint already tracks what's "
             "done -- this must be opt-in, not the default, so newly-added "
             "files are never silently excluded.",
    )
    parser.add_argument(
        "--retry_failed_only",
        action="store_true",
        help="Sirf failed_chunks.json mein listed chunk_ids ko dobara try karo.",
    )
    parser.add_argument(
        "--no-skip-repaired",
        action="store_true",
        help="Also re-scan files listed in repair_checkpoint.json (skipped by "
             "default since they're already embedded -- verified against "
             "embedding_checkpoint.json). Same convention as "
             "repair_metadata.py's --no-skip-repaired.",
    )
    args = parser.parse_args()

    repaired_list = args.repaired_list if args.repaired_list else None
    run(
        args.json_dir,
        repaired_list,
        args.retry_failed_only,
        skip_already_embedded=not args.no_skip_repaired,
    )