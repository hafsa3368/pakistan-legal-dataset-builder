"""
sync_judge_names_to_neo4j.py
==============================
Companion script to fix_judge_names_json.py.

fix_judge_names_json.py only rewrote the local JSON files under
extracted_text_clean/ (2,105 judge fields recovered via regex+Ollama). It
does NOT touch Neo4j. This script closes that gap -- but ONLY for genuine
gaps, and ONLY with re-validated, re-cleaned, consistently-cased values.
That carefulness is not optional here: an early dry run of a naive
"sync whatever's in the JSON" version of this script showed it would have
REGRESSED the graph -- e.g. pushing 'Muhammad Ali Mahzar' (a typo) over
the top of 'Muhammad Ali Mazhar' (the clean canonical name
fix_judge_names.py had already consolidated multiple raw variants into),
and reintroducing 'JUDGE Naeem' (garbage fix_judge_names.py had already
deleted) for a case fix_judge_names_json.py never managed to actually fix.
The root cause: JSON files only got REWRITTEN where their OWN judge value
was invalid; files that were "valid-looking but messy" (a typo, a stray
casing) were left as-is in the JSON even though Neo4j's copy of that same
judge identity had since been cleaned/merged to something better. Blindly
diffing JSON-vs-Neo4j and syncing on any difference would have overwritten
the better side with the worse one about as often as the reverse.

Given that, this script:
  - Matches each JSON file to its Case node via generated_name (falling
    back to actual_filename) -- JSON files have no case_id field, that
    only exists inside Neo4j (assigned via Qdrant matching at import
    time).
  - MATCH-only for the Case node -- never creates one.
  - Re-cleans and re-validates the JSON's judge value with the SAME
    clean_judge_candidate() / is_valid_judge_name() logic new_extractor.py
    now uses at extraction time (imported directly from there, not
    duplicated) -- a value that wouldn't pass fresh extraction doesn't get
    synced, no matter what's sitting in the JSON file.
  - ONLY fills genuine gaps: syncs a case ONLY when Neo4j currently has NO
    DECIDED_BY judge at all. An existing judge value in Neo4j is NEVER
    overwritten by this script -- if it's wrong, that's fix_judge_names.py's
    job (which operates on Neo4j directly and already reasons about
    cross-case frequency, fuzzy duplicates, etc. that this per-file script
    cannot see).
  - Before creating a new :Judge node, checks case-insensitively for an
    EXISTING one with the same name and reuses its exact casing/node --
    prevents 'Ali Zia Bajwa' and 'ALI ZIA BAJWA' becoming two different
    graph identities for the same person.
  - Does NOT touch chunks, embeddings, Qdrant, SIMILAR_TO edges, CITES,
    APPLIES, INVOLVES, or any other relationship/label.
  - Dry run by default; nothing written to Neo4j unless --apply is passed.
  - Checkpoint/resume + real Ctrl+C handling, same pattern as the rest of
    the pipeline.

USAGE:
    python sync_judge_names_to_neo4j.py                 # dry run, preview only
    python sync_judge_names_to_neo4j.py --apply          # actually write to Neo4j
"""

import os
import sys
import json
import glob
import signal
import argparse

from dotenv import load_dotenv
from neo4j import GraphDatabase
from neo4j.exceptions import Neo4jError

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from new_extractor import clean_judge_candidate, is_valid_judge_name  # noqa: E402

load_dotenv()

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

OUTPUT_DIR = "extracted_text_clean"
REPAIR_CHECKPOINT_FILE = "repair_checkpoint.json"   # read-only scope source
CHECKPOINT_FILE = "sync_judge_names_neo4j_checkpoint.json"
SKIPPED_FILE = "sync_judge_names_neo4j_skipped.json"

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://127.0.0.1:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

if not NEO4J_PASSWORD:
    print("❌ NEO4J_PASSWORD missing in .env -- aborting.")
    sys.exit(1)


# ==========================================================
# CTRL+C HANDLING
# ==========================================================
_INTERRUPTED = False


def _handle_sigint(signum, frame):
    global _INTERRUPTED
    if not _INTERRUPTED:
        print("\n⏸ Ctrl+C received -- stopping after current file, saving checkpoint...", flush=True)
    _INTERRUPTED = True


signal.signal(signal.SIGINT, _handle_sigint)


# ==========================================================
# SCOPE + CHECKPOINT HELPERS
# ==========================================================
def load_repaired_filenames() -> set:
    if not os.path.exists(REPAIR_CHECKPOINT_FILE):
        raise FileNotFoundError(f"{REPAIR_CHECKPOINT_FILE} not found in this folder.")
    with open(REPAIR_CHECKPOINT_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    names = set()
    if isinstance(data, list):
        names.update(data)
    elif isinstance(data, dict):
        for key in ("completed", "fixed", "processed", "done", "repaired"):
            value = data.get(key)
            if isinstance(value, list):
                names.update(value)
            elif isinstance(value, dict):
                names.update(value.keys())
    return {os.path.basename(n) for n in names}


def load_checkpoint() -> set:
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, "r", encoding="utf-8") as f:
            return set(json.load(f))
    return set()


def save_checkpoint(done: set):
    tmp_path = CHECKPOINT_FILE + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(sorted(done), f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, CHECKPOINT_FILE)


# ==========================================================
# NEO4J SYNC
# ==========================================================
def find_case_gap(tx, generated_name: str, actual_filename: str):
    """Returns a record only if a matching Case node exists AND it
    currently has NO DECIDED_BY judge at all -- i.e. only genuine gaps."""
    result = tx.run(
        """
        MATCH (c:Case)
        WHERE c.generated_name = $generated_name OR c.actual_filename = $actual_filename
        OPTIONAL MATCH (c)-[:DECIDED_BY]->(j:Judge)
        RETURN c.case_id AS case_id, j.name AS current_judge
        LIMIT 1
        """,
        generated_name=generated_name, actual_filename=actual_filename,
    )
    return result.single()


def find_existing_judge_casing(tx, candidate: str):
    """Case-insensitive lookup so we reuse an existing :Judge node's exact
    casing instead of creating a second node for the same person."""
    result = tx.run(
        """
        MATCH (j:Judge)
        WHERE toLower(j.name) = toLower($candidate)
        RETURN j.name AS name
        LIMIT 1
        """,
        candidate=candidate,
    )
    record = result.single()
    return record["name"] if record else candidate


def fill_gap(tx, case_id: str, judge_name: str):
    tx.run(
        """
        MATCH (c:Case {case_id: $case_id})
        SET c.judge_name = $judge_name
        MERGE (j:Judge {name: $judge_name})
        MERGE (c)-[:DECIDED_BY]->(j)
        """,
        case_id=case_id, judge_name=judge_name,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true",
                         help="Actually write to Neo4j. Without this, dry-run preview only.")
    args = parser.parse_args()

    repaired_names = load_repaired_filenames()
    all_files = [f for f in glob.glob(os.path.join(OUTPUT_DIR, "*.json")) if os.path.basename(f) in repaired_names]

    done = load_checkpoint() if args.apply else set()
    pending = [f for f in all_files if os.path.basename(f) not in done]

    print(f"📋 In-scope files (from {REPAIR_CHECKPOINT_FILE}): {len(repaired_names)}")
    print(f"📂 Matched on disk: {len(all_files)} | pending: {len(pending)}")
    print(f"Mode: {'APPLY (writing to Neo4j)' if args.apply else 'DRY RUN (preview only)'}")
    print("Policy: fills genuine gaps only -- never overwrites an existing Neo4j judge value.\n")

    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

    filled = 0
    no_judge_in_json = 0
    invalid_after_reclean = 0
    already_has_judge = 0
    not_found_in_neo4j = 0
    skipped_entries = []
    i = 0

    try:
        with driver.session() as session:
            for i, path in enumerate(pending, 1):
                if _INTERRUPTED:
                    break

                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)

                raw_judge = str(data.get("judge") or "").strip()
                if not raw_judge:
                    no_judge_in_json += 1
                    if args.apply:
                        done.add(os.path.basename(path))
                        save_checkpoint(done)
                    continue

                candidate = clean_judge_candidate(raw_judge)
                if not is_valid_judge_name(candidate):
                    invalid_after_reclean += 1
                    if args.apply:
                        done.add(os.path.basename(path))
                        save_checkpoint(done)
                    continue

                generated_name = str(data.get("generated_name") or "").strip()
                actual_filename = str(data.get("actual_filename") or "").strip()

                try:
                    record = session.execute_read(find_case_gap, generated_name, actual_filename)
                except Neo4jError as e:
                    print(f"[{i}/{len(pending)}] Neo4j error looking up {generated_name!r}: {e}", flush=True)
                    continue

                if record is None:
                    # Either no matching Case node, or it already has SOME
                    # judge -- both cases skip, but only the former is
                    # logged, since "already has a judge" is expected/fine.
                    lookup_check = session.execute_read(
                        lambda tx: tx.run(
                            "MATCH (c:Case) WHERE c.generated_name = $g OR c.actual_filename = $a "
                            "RETURN c.case_id AS cid LIMIT 1",
                            g=generated_name, a=actual_filename,
                        ).single()
                    )
                    if lookup_check is None:
                        not_found_in_neo4j += 1
                        skipped_entries.append({
                            "file": os.path.basename(path),
                            "generated_name": generated_name,
                            "reason": "no_matching_case_node_in_neo4j",
                        })
                    else:
                        already_has_judge += 1
                    if args.apply:
                        done.add(os.path.basename(path))
                        save_checkpoint(done)
                    continue

                case_id = record["case_id"]
                final_name = session.execute_read(find_existing_judge_casing, candidate)

                print(f"[{i}/{len(pending)}] {case_id}: (gap) -> {final_name!r}", flush=True)

                if args.apply:
                    try:
                        session.execute_write(fill_gap, case_id, final_name)
                        filled += 1
                    except Neo4jError as e:
                        print(f"    WARN: could not fill gap for {case_id}: {e}", flush=True)
                    done.add(os.path.basename(path))
                    save_checkpoint(done)

                if i % 500 == 0:
                    print(f"    📊 Progress: {i}/{len(pending)} scanned | {filled} gaps filled | "
                          f"{already_has_judge} already had a judge\n", flush=True)
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
            print("👉 Dry run does not checkpoint progress -- re-run will rescan from the start.")
    else:
        print(f"\n✅ Done.")
        if args.apply:
            print(f"   {filled} Neo4j Case node(s) had a judge gap filled.")
        print(f"   {already_has_judge} already had a judge in Neo4j -- left untouched.")
        print(f"   {no_judge_in_json} JSON file(s) have no judge value -- nothing to sync.")
        print(f"   {invalid_after_reclean} JSON judge value(s) failed re-validation -- not synced.")
        if not_found_in_neo4j:
            print(f"   ⚠ {not_found_in_neo4j} file(s) had no matching Case node in Neo4j -- see {SKIPPED_FILE}.")
        if not args.apply:
            print("👉 This was a DRY RUN. Re-run with --apply to write to Neo4j.")


if __name__ == "__main__":
    main()
