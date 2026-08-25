import json
import os
from collections import defaultdict

from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchAny

# =========================================================
# CONFIG
# =========================================================

OUTPUT_DIR = "extracted_text_clean"
REPAIR_CHECKPOINT_FILE = "fix_case_numbers_checkpoint.json"

QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
COLLECTION_NAME = "legal_chunks"

BATCH_SIZE = 250
CHUNK_SIZE = 200  # how many filenames go into one MatchAny filter query

CHECKPOINT_FILE = "sync_case_numbers_qdrant_checkpoint.json"


# =========================================================
# CHECKPOINT HELPERS
# =========================================================

def load_checkpoint():
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        return (
            set(data.get("done_chunks", [])),
            data.get("processed", 0),
            data.get("updated", 0),
            data.get("skipped", 0),
        )
    return set(), 0, 0, 0


def save_checkpoint(done_chunks, processed, updated, skipped):
    with open(CHECKPOINT_FILE, "w", encoding="utf-8") as f:
        json.dump(
            {
                "done_chunks": sorted(done_chunks),
                "processed": processed,
                "updated": updated,
                "skipped": skipped,
            },
            f,
        )


def chunked(seq, size):
    for i in range(0, len(seq), size):
        yield seq[i:i + size]


# =========================================================
# BUILD filename -> case_number MAP FROM JSON (source of truth)
# =========================================================

print("Building filename -> case_number map from repaired JSON files "
      f"listed in {REPAIR_CHECKPOINT_FILE}...", flush=True)

with open(REPAIR_CHECKPOINT_FILE, "r", encoding="utf-8") as f:
    repaired_names = json.load(f)

by_generated_name = {}
by_actual_filename = {}

json_files = [
    os.path.join(OUTPUT_DIR, os.path.basename(name))
    for name in repaired_names
]
json_files = [p for p in json_files if os.path.exists(p)]

for path in json_files:
    with open(path, "r", encoding="utf-8") as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError:
            continue

    case_number_raw = data.get("case_number", "")
    if not isinstance(case_number_raw, str):
        continue
    case_number = case_number_raw.strip()
    if not case_number:
        continue

    generated_name = str(data.get("generated_name") or "").strip()
    actual_filename = str(data.get("actual_filename") or "").strip()

    if generated_name:
        by_generated_name[generated_name] = case_number
    if actual_filename:
        by_actual_filename.setdefault(actual_filename, case_number)

# filenames only identifiable via actual_filename (generated_name missing
# or not the match key) still need a lookup pass, but keep that list small
actual_only = {k: v for k, v in by_actual_filename.items() if k not in by_generated_name}

print(f"Loaded {len(by_generated_name)} generated_name entries and "
      f"{len(actual_only)} actual_filename-only entries "
      f"from {len(json_files)} repaired JSON files.", flush=True)


# =========================================================
# CONNECT
# =========================================================

client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, timeout=30)
print("Connected to Qdrant", flush=True)

collection_info = client.get_collection(COLLECTION_NAME)
total_points = collection_info.points_count
print(f"Collection: {COLLECTION_NAME}", flush=True)
print(f"Points: {total_points}", flush=True)


# =========================================================
# SYNC PAYLOAD -- scoped ONLY to the repaired filenames, using the
# indexed generated_name / actual_filename fields to filter Qdrant
# server-side instead of scrolling the whole 55k-point collection.
# =========================================================

done_chunks, processed, updated, skipped = load_checkpoint()

if done_chunks:
    print(
        f"Resuming from checkpoint: {len(done_chunks)} chunk(s) already done, "
        f"processed={processed}, updated={updated}, skipped={skipped}",
        flush=True,
    )
else:
    print("No checkpoint found, starting fresh.", flush=True)

# build the list of (field, lookup_dict, values_chunk) work items
work_items = []
for field, lookup in (("generated_name", by_generated_name), ("actual_filename", actual_only)):
    values = list(lookup.keys())
    for chunk in chunked(values, CHUNK_SIZE):
        work_items.append((field, chunk))

total_chunks = len(work_items)
print(f"Total lookup chunks to process: {total_chunks} "
      f"(chunk size {CHUNK_SIZE})", flush=True)

for chunk_idx, (field, values) in enumerate(work_items):
    if chunk_idx in done_chunks:
        continue

    lookup = by_generated_name if field == "generated_name" else actual_only

    # accumulate groups across ALL pages of this chunk, then flush once --
    # avoids one set_payload round-trip per unique value PER PAGE
    chunk_groups = defaultdict(list)

    chunk_offset = None
    page = 0
    while True:
        page += 1
        print(f"  ... chunk {chunk_idx + 1}/{total_chunks} ({field}) "
              f"fetching page {page}", flush=True)
        points, next_offset = client.scroll(
            collection_name=COLLECTION_NAME,
            scroll_filter=Filter(
                must=[FieldCondition(key=field, match=MatchAny(any=values))]
            ),
            limit=BATCH_SIZE,
            offset=chunk_offset,
            with_payload=["generated_name", "actual_filename", "case_number"],
            with_vectors=False,
        )

        if not points:
            break

        for point in points:
            payload = point.payload or {}
            key = str(payload.get(field) or "").strip()

            new_case_number = lookup.get(key)
            if not new_case_number:
                skipped += 1
                continue

            old_case_number = payload.get("case_number")
            if old_case_number == new_case_number:
                continue

            chunk_groups[new_case_number].append(point.id)

        processed += len(points)
        print(f"  ... page {page} done: {len(points)} points fetched "
              f"(running total groups: {len(chunk_groups)})", flush=True)
        chunk_offset = next_offset
        if chunk_offset is None:
            break

    print(f"  Flushing {len(chunk_groups)} update group(s) for chunk "
          f"{chunk_idx + 1}/{total_chunks}...", flush=True)
    for group_i, (case_number, point_ids) in enumerate(chunk_groups.items(), 1):
        client.set_payload(
            collection_name=COLLECTION_NAME,
            payload={"case_number": case_number},
            points=point_ids,
        )
        updated += len(point_ids)
        if group_i % 20 == 0 or group_i == len(chunk_groups):
            print(f"      ... {group_i}/{len(chunk_groups)} groups flushed", flush=True)

    done_chunks.add(chunk_idx)
    save_checkpoint(done_chunks, processed, updated, skipped)

    print(
        f"Chunk {chunk_idx + 1}/{total_chunks} ({field}) done -> "
        f"processed={processed} | updated={updated} | skipped={skipped}",
        flush=True,
    )


# =========================================================
# FINISHED
# =========================================================

print("\n===================================", flush=True)
print("Qdrant case_number sync completed", flush=True)
print("===================================", flush=True)
print(f"Updated : {updated}", flush=True)
print(f"Skipped : {skipped}", flush=True)
print("Vectors were NOT regenerated.", flush=True)
print("===================================", flush=True)

if os.path.exists(CHECKPOINT_FILE):
    os.remove(CHECKPOINT_FILE)
    print("Checkpoint file removed (run completed fully).", flush=True)
