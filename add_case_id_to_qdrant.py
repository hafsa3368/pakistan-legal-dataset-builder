from collections import defaultdict
import json
import os

from qdrant_client import QdrantClient

# =========================================================
# CONFIG
# =========================================================

QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
COLLECTION_NAME = "legal_chunks"

BATCH_SIZE = 250  # FIX: bigger scroll batch is fine now since we group before upserting

# FIX: checkpoint file so script resumes from where it stopped
CHECKPOINT_FILE = "case_id_qdrant_checkpoint.json"


# =========================================================
# CHECKPOINT HELPERS
# =========================================================

def load_checkpoint():
    """Returns (offset, processed, updated, skipped) from last run, or fresh start."""
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        return (
            data.get("offset"),
            data.get("processed", 0),
            data.get("updated", 0),
            data.get("skipped", 0),
        )
    return None, 0, 0, 0


def save_checkpoint(offset, processed, updated, skipped):
    with open(CHECKPOINT_FILE, "w", encoding="utf-8") as f:
        json.dump(
            {
                "offset": offset,
                "processed": processed,
                "updated": updated,
                "skipped": skipped,
            },
            f,
        )


# =========================================================
# CONNECT
# =========================================================

client = QdrantClient(
    host=QDRANT_HOST,
    port=QDRANT_PORT
)

print("Connected to Qdrant")


# =========================================================
# CHECK COLLECTION
# =========================================================

collection_info = client.get_collection(COLLECTION_NAME)

print(f"Collection: {COLLECTION_NAME}")
print(f"Points: {collection_info.points_count}")


# =========================================================
# UPDATE PAYLOAD
# =========================================================

offset, processed, updated, skipped = load_checkpoint()
total_points = collection_info.points_count

if offset is not None:
    print(
        f"Resuming from checkpoint: processed={processed}, "
        f"updated={updated}, skipped={skipped}",
        flush=True,
    )
else:
    print("No checkpoint found, starting fresh.", flush=True)

while True:

    print(f"Fetching next batch (processed so far: {processed}/{total_points})...", flush=True)

    points, next_offset = client.scroll(
        collection_name=COLLECTION_NAME,
        limit=BATCH_SIZE,
        offset=offset,
        with_payload=True,
        with_vectors=False
    )

    if not points:
        break

    # -----------------------------------------------------
    # FIX: group point_ids by case_id within this batch,
    # so we can send ONE set_payload call per unique case_id
    # instead of one call per point (55k calls -> a few hundred).
    # -----------------------------------------------------
    groups = defaultdict(list)

    for point in points:

        payload = point.payload or {}

        actual_filename = payload.get("actual_filename")
        generated_name = payload.get("generated_name")

        actual_filename = (
            str(actual_filename).strip()
            if actual_filename
            else ""
        )

        generated_name = (
            str(generated_name).strip()
            if generated_name
            else ""
        )

        case_id = actual_filename if actual_filename else generated_name

        if not case_id:
            skipped += 1
            continue

        groups[case_id].append(point.id)

    # FIX: one set_payload call per unique case_id in this batch
    for case_id, point_ids in groups.items():
        client.set_payload(
            collection_name=COLLECTION_NAME,
            payload={
                "case_id": case_id
            },
            points=point_ids
        )
        updated += len(point_ids)

    processed += len(points)
    remaining = total_points - processed

    # FIX: plain print instead of tqdm bar, so progress is visible even
    # in terminals/consoles that don't render carriage-return progress bars.
    print(
        f"Batch done -> Done: {processed}/{total_points} | "
        f"Remaining: {remaining} | Updated: {updated} | Skipped: {skipped}",
        flush=True,
    )

    offset = next_offset

    # FIX: save checkpoint after every batch, so a Ctrl+C or crash
    # loses at most one batch of work, not the whole run.
    save_checkpoint(offset, processed, updated, skipped)

    if offset is None:
        break


# =========================================================
# FINISHED
# =========================================================

print("\n===================================")
print("Qdrant payload update completed")
print("===================================")
print(f"Updated : {updated}")
print(f"Skipped : {skipped}")
print("Vectors were NOT regenerated.")
print("===================================")

# FIX: clean up checkpoint on successful full completion
if os.path.exists(CHECKPOINT_FILE):
    os.remove(CHECKPOINT_FILE)
    print("Checkpoint file removed (run completed fully).")