"""
count_remaining.py
    python embedd_Count_remaining.py --json_dir "d:\hafsa_thesis material\supreme_court_scraper\extracted_text_clean"
-------------------
Batata hai ke embedding pipeline mein kitna kaam bacha hai — bina koi
Ollama/Qdrant call kiye. Sirf JSON files padh kar checkpoint ke against
count karta hai, isliye seconds mein result de deta hai.

USAGE:
    python count_remaining.py --json_dir "d:\\hafsa_thesis material\\supreme_court_scraper\\extracted_text_clean"

Optional (defaults match generate_embeddings.py ke defaults):
    --repaired_list repair_checkpoint.json
    --checkpoint embedding_checkpoint.json
    --failed failed_chunks.json
"""

import json
import argparse
from pathlib import Path


def load_set_from_json(path, key=None):
    p = Path(path)
    if not p.exists():
        return set()
    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)
    if key is not None and isinstance(data, dict):
        return set(data.get(key, []))
    if isinstance(data, list):
        return set(data)
    if isinstance(data, dict):
        return set(data.keys())
    return set()


def main():
    parser = argparse.ArgumentParser(description="Count remaining chunks/files for the embedding pipeline.")
    parser.add_argument("--json_dir", required=True, help="Folder containing your repaired JSON files")
    parser.add_argument("--repaired_list", default="repair_checkpoint.json")
    parser.add_argument("--checkpoint", default="embedding_checkpoint.json")
    parser.add_argument("--failed", default="failed_chunks.json")
    args = parser.parse_args()

    # --- load reference sets ---
    repaired_filenames = load_set_from_json(args.repaired_list)
    processed_ids = load_set_from_json(args.checkpoint, key="processed_ids")
    failed_ids = set(load_set_from_json(args.failed))  # dict keys = chunk_ids

    json_files = list(Path(args.json_dir).glob("*.json"))
    if repaired_filenames:
        json_files = [jf for jf in json_files if jf.name in repaired_filenames]

    # --- scan files, count chunks (no embedding calls, just reading text) ---
    total_files = 0
    total_chunks = 0
    done_chunks = 0
    failed_chunks_count = 0
    pending_chunks = 0
    fully_done_files = 0
    partially_done_files = 0
    untouched_files = 0

    for jf in json_files:
        try:
            with open(jf, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue

        chunks = data.get("chunks", [])
        if not chunks:
            continue

        total_files += 1
        file_chunk_ids = []
        for chunk in chunks:
            if not chunk.get("text", "").strip():
                continue
            chunk_index = chunk.get("chunk_index", 0)
            chunk_id = f"{jf.stem}_{chunk_index}"
            file_chunk_ids.append(chunk_id)

        total_chunks += len(file_chunk_ids)

        file_done = sum(1 for cid in file_chunk_ids if cid in processed_ids)
        file_failed = sum(1 for cid in file_chunk_ids if cid in failed_ids)

        done_chunks += file_done
        failed_chunks_count += file_failed
        pending_chunks += len(file_chunk_ids) - file_done - file_failed

        if file_done == len(file_chunk_ids) and len(file_chunk_ids) > 0:
            fully_done_files += 1
        elif file_done > 0:
            partially_done_files += 1
        else:
            untouched_files += 1

    print("=" * 55)
    print("CHUNK-LEVEL PROGRESS (ye asli embedding progress hai)")
    print(f"  Total chunks (repaired files mein)  : {total_chunks}")
    print(f"  Already embedded (checkpoint)        : {done_chunks}")
    print(f"  Permanently failed                   : {failed_chunks_count}")
    print(f"  Remaining (pending)                  : {pending_chunks}")
    if total_chunks:
        pct = 100 * done_chunks / total_chunks
        print(f"  Progress                             : {pct:.1f}%")
    print()
    print("FILE-LEVEL BREAKDOWN")
    print(f"  Total repaired files matched on disk : {total_files}")
    print(f"  Fully embedded (100% chunks done)    : {fully_done_files}")
    print(f"  Partially embedded                   : {partially_done_files}")
    print(f"  Not started yet                      : {untouched_files}")
    print("=" * 55)


if __name__ == "__main__":
    main()
