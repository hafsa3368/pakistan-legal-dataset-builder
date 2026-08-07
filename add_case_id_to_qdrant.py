from qdrant_client import QdrantClient
from tqdm import tqdm

# =========================================================
# CONFIG
# =========================================================

QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
COLLECTION_NAME = "legal_chunks"

BATCH_SIZE = 100


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

offset = None
updated = 0
skipped = 0

while True:

    points, next_offset = client.scroll(
        collection_name=COLLECTION_NAME,
        limit=BATCH_SIZE,
        offset=offset,
        with_payload=True,
        with_vectors=False
    )

    if not points:
        break

    for point in points:

        payload = point.payload or {}

        # -------------------------------------------------
        # Get filename
        # -------------------------------------------------

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

        # -------------------------------------------------
        # Same case_id rule as Neo4j
        # -------------------------------------------------

        case_id = actual_filename if actual_filename else generated_name

        if not case_id:
            skipped += 1
            continue

        # -------------------------------------------------
        # Add case_id to existing payload
        # -------------------------------------------------

        client.set_payload(
            collection_name=COLLECTION_NAME,
            payload={
                "case_id": case_id
            },
            points=[point.id]
        )

        updated += 1

    print(
        f"Updated: {updated} | "
        f"Skipped: {skipped}"
    )

    offset = next_offset

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