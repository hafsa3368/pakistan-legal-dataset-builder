"""
sync_case_numbers_to_neo4j.py
==============================
Companion script to fix_case_numbers.py.

fix_case_numbers.py only rewrites the local JSON files under
extracted_text_clean/. It does NOT touch Neo4j. This script closes that
gap: it reads the (now-corrected) case_number field out of each JSON file
and pushes it into the matching Neo4j Case node -- and does STRICTLY
NOTHING else.

Guarantees / constraints (same spirit as every other script in this
project):
  - Touches ONLY the case_number property on (:Case) nodes.
  - Does NOT touch chunks, embeddings, Qdrant, SIMILAR_TO edges, or any
    other relationship or node label.
  - Does NOT create new nodes. Uses MATCH, not MERGE -- if a case_id from
    the JSON doesn't already exist as a Case node, it's skipped and
    logged, never silently created (creating a node here would be outside
    this script's job and could desync from the real import pipeline).
  - Dry run by default; nothing is written to Neo4j unless --apply is passed.
  - Checkpoint/resume + real Ctrl+C handling, same pattern as the rest of
    the pipeline.
  - Credentials read from .env (NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD),
    never hardcoded.

USAGE:
    python sync_case_numbers_to_neo4j.py                 # dry run, preview only
    python sync_case_numbers_to_neo4j.py --apply          # actually write to Neo4j
    python sync_case_numbers_to_neo4j.py --apply --case-ids-file case_ids.json
                                                            # restrict to a specific
                                                            # subset (e.g. the 1561
                                                            # confirmed-affected ids)
"""

import os
import re
import sys
import json
import glob
import signal
import argparse

from dotenv import load_dotenv
from neo4j import GraphDatabase

load_dotenv()

OUTPUT_DIR       = "extracted_text_clean"
CHECKPOINT_FILE  = "sync_case_numbers_checkpoint.json"
SKIPPED_FILE     = "sync_case_numbers_skipped.json"

NEO4J_URI      = os.getenv("NEO4J_URI")
NEO4J_USER     = os.getenv("NEO4J_USER")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

if not all([NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD]):
    print("❌ Missing NEO4J_URI / NEO4J_USER / NEO4J_PASSWORD in .env — aborting.")
    sys.exit(1)


# ==========================
# CTRL+C HANDLING
# ==========================
_INTERRUPTED = False

def _handle_sigint(signum, frame):
    global _INTERRUPTED
    if not _INTERRUPTED:
        print("\n⏸ Ctrl+C received — stopping after current file, saving checkpoint...", flush=True)
    _INTERRUPTED = True

signal.signal(signal.SIGINT, _handle_sigint)


# ==========================
# KNOWN-BAD case_number LITERALS
# (kept in sync with fix_case_numbers.py / legal_answer.py Revision 20.
#  Used here only for a pre-flight sanity check, never to gate the write.)
# ==========================
_KNOWN_BAD_CASE_NUMBERS = re.compile(
    r"Cr\.?\s*B\.?A\.?\s*No\.?\s*S-?994",
    re.IGNORECASE,
)

def is_known_bad_case_number(value) -> bool:
    if not value:
        return False
    return bool(_KNOWN_BAD_CASE_NUMBERS.search(str(value)))


# ==========================
# CHECKPOINT HELPERS
# ==========================
def load_checkpoint() -> set:
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, "r", encoding="utf-8") as f:
            return set(json.load(f))
    return set()


def save_checkpoint(done: set):
    with open(CHECKPOINT_FILE, "w", encoding="utf-8") as f:
        json.dump(sorted(done), f, ensure_ascii=False, indent=2)


def load_case_ids_filter(path):
    """Optional: restrict the sync run to a specific set of case_ids
    (e.g. output of the Cypher scan for the 1561 known-affected nodes)."""
    if not path:
        return None
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return set(data)
    if isinstance(data, dict) and "case_ids" in data:
        return set(data["case_ids"])
    raise ValueError(f"Could not parse case_ids from {path} — expected a list or {{'case_ids': [...]}}")


# ==========================
# NEO4J SYNC
# ==========================
def sync_one(tx, case_id: str, new_case_number: str) -> bool:
    """MATCH-only (never MERGE) -- returns True if a node was found and updated."""
    result = tx.run(
        """
        MATCH (c:Case {case_id: $case_id})
        SET c.case_number = $new_case_number
        RETURN c.case_id AS updated_id
        """,
        case_id=case_id,
        new_case_number=new_case_number,
    )
    return result.single() is not None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true",
                         help="Actually write to Neo4j. Without this, dry-run preview only.")
    parser.add_argument("--case-ids-file", default=None,
                         help="Optional JSON file (list, or {'case_ids': [...]}) restricting "
                              "the sync to a specific subset of case_ids.")
    args = parser.parse_args()

    case_ids_filter = load_case_ids_filter(args.case_ids_file)
    if case_ids_filter is not None:
        print(f"🔎 Restricting sync to {len(case_ids_filter)} case_id(s) from {args.case_ids_file}")

    all_files = glob.glob(os.path.join(OUTPUT_DIR, "*.json"))
    done = load_checkpoint() if args.apply else set()
    pending = [f for f in all_files if os.path.basename(f) not in done]

    print(f"📂 JSON files on disk: {len(all_files)} | pending: {len(pending)}")
    print(f"Mode: {'APPLY (writing to Neo4j)' if args.apply else 'DRY RUN (preview only)'}\n")

    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

    updated = 0
    not_found = 0
    skipped_no_case_id = 0
    still_bad_after_sync = 0
    skipped_entries = []
    i = 0

    try:
        with driver.session() as session:
            for i, path in enumerate(pending, 1):
                if _INTERRUPTED:
                    break

                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)

                case_id = data.get("case_id")
                case_number = data.get("case_number", "")

                if not case_id:
                    skipped_no_case_id += 1
                    skipped_entries.append({
                        "file": os.path.basename(path),
                        "reason": "missing_case_id",
                    })
                    if args.apply:
                        done.add(os.path.basename(path))
                        save_checkpoint(done)
                    continue

                if case_ids_filter is not None and case_id not in case_ids_filter:
                    if args.apply:
                        done.add(os.path.basename(path))
                        save_checkpoint(done)
                    continue

                if is_known_bad_case_number(case_number):
                    # The local JSON itself still holds a known-bad value --
                    # fix_case_numbers.py hasn't resolved this one yet.
                    # Sync it anyway (so Neo4j matches the JSON state) but
                    # count it separately so it's visible in the summary.
                    still_bad_after_sync += 1

                print(f"[{i}/{len(pending)}] {case_id}: syncing case_number = {case_number!r}"
                      f"{'  (still known-bad)' if is_known_bad_case_number(case_number) else ''}",
                      flush=True)

                if args.apply:
                    found = session.execute_write(sync_one, case_id, case_number)
                    if found:
                        updated += 1
                    else:
                        not_found += 1
                        skipped_entries.append({
                            "file": os.path.basename(path),
                            "case_id": case_id,
                            "reason": "no_matching_case_node_in_neo4j",
                        })
                    done.add(os.path.basename(path))
                    save_checkpoint(done)

                if i % 500 == 0:
                    print(f"    📊 Progress: {i}/{len(pending)} scanned | {updated} updated | "
                          f"{not_found} not found in Neo4j\n", flush=True)
    finally:
        driver.close()

    if skipped_entries:
        with open(SKIPPED_FILE, "w", encoding="utf-8") as f:
            json.dump(skipped_entries, f, ensure_ascii=False, indent=2)

    if _INTERRUPTED:
        print(f"\n⏸ Stopped early by Ctrl+C. Scanned {i}/{len(pending)} in this run.")
        if args.apply:
            print(f"💾 Checkpoint saved: {len(done)} files done total. Re-run to resume.")
        else:
            print("👉 Dry run does not checkpoint progress — re-run will rescan from the start.")
    else:
        print(f"\n✅ Done.")
        if args.apply:
            print(f"   {updated} Neo4j Case node(s) had case_number updated.")
            if not_found:
                print(f"   ⚠ {not_found} case_id(s) from JSON had no matching Case node in Neo4j "
                      f"— see {SKIPPED_FILE}.")
        if skipped_no_case_id:
            print(f"   ⚠ {skipped_no_case_id} JSON file(s) had no case_id field at all — skipped, see {SKIPPED_FILE}.")
        if still_bad_after_sync:
            print(f"   🛑 {still_bad_after_sync} synced value(s) were STILL a known-bad literal "
                  f"(fix_case_numbers.py hasn't resolved these yet) — Neo4j now correctly mirrors "
                  f"that unresolved state, but these need fix_case_numbers.py re-run (--use-ollama "
                  f"or manual fix) before this sync is re-run on them.")
        if not args.apply:
            print("👉 This was a DRY RUN. Re-run with --apply to write to Neo4j.")
        print(f"\nNext step: re-run your Cypher check —")
        print(f"  MATCH (c:Case) WHERE c.case_number CONTAINS 'S-994' RETURN count(c)")
        print(f"  Expect 0 (or only the {still_bad_after_sync} still-unresolved ones, if any).")


if __name__ == "__main__":
    main()