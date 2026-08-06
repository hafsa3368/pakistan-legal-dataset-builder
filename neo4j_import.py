import os
import json
import re
import signal
import time
from pathlib import Path
from typing import List, Dict, Any, Set

from neo4j import GraphDatabase
from neo4j.exceptions import ServiceUnavailable, SessionExpired, TransientError
from tqdm import tqdm

# =========================================================
# CONFIG  --->  SIRF YAHAN 2 CHEEZEIN CHANGE KARNI HAIN <---
# =========================================================

# 1) Ye folder jahan aapki extracted JSON files pari hain (data yahan se aayega)
JSON_FOLDER = r"D:\hafsa_thesis material\supreme_court_scraper\extracted_text_clean"

# 2) repair_checkpoint.json ka poora path (jo repair_metadata.py ne banayi thi)
#    Agar ye file kisi aur folder mein hai toh yahan sahi path daalein.
REPAIR_CHECKPOINT_FILE = r"D:\hafsa_thesis material\supreme_court_scraper\repair_checkpoint.json"

# =========================================================
# NEO4J LOGIN  --->  Neo4j Desktop mein "legal" DBMS banate waqt
#                     jo password AAPNE khud set kiya tha, wahi yahan daalein
# =========================================================
NEO4J_URI = "bolt://127.0.0.1:7687"   # local instance ka URI (screenshot mein yehi dikha tha)
NEO4J_USER = "neo4j"                  # ye default hi rehne dein
NEO4J_PASSWORD = "se310TJ@"   # <--- YAHAN apna real password daalein

BATCH_SIZE = 100
MAX_RETRIES = 3          # retries per batch on transient Neo4j errors
RETRY_BASE_DELAY = 2     # seconds, doubles each retry

CHECKPOINT_FILE = "neo4j_import_checkpoint.json"   # this script's OWN resume checkpoint
LOG_FILE = "neo4j_import_errors.log"

# =========================================================
# GLOBAL STOP FLAG (Ctrl+C)
# =========================================================
STOP_REQUESTED = False


def handle_sigint(signum, frame):
    global STOP_REQUESTED
    STOP_REQUESTED = True
    print("\n🛑 Ctrl+C received. Finishing current batch and saving checkpoint...")


signal.signal(signal.SIGINT, handle_sigint)

# =========================================================
# LOGGING
# =========================================================
def log_error(message: str):
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(message + "\n")


# =========================================================
# FIX: normalize_court() aur normalize_case_type() — fix_existing_metadata.py
# se hoobahoo copy. Pehle ye script raw "court" / "case_type" strings ko
# seedha Neo4j node banane ke liye use kar rahi thi (sirf .strip() ho raha
# tha), isliye "High Court of Sindh" ke 200+ OCR-garbled variants alag-alag
# Court nodes ban rahe thay, aur Topic (case_type) bhi normalize nahi ho
# raha tha. Ab dono fields Excel wale hi standard categories mein map
# hongi, taake Excel aur Neo4j dono mein same clean values ho.
# =========================================================
def normalize_court(raw, fallback="Unknown"):
    if not raw or not str(raw).strip():
        return fallback

    s = str(raw).upper().strip()

    # FIX: JSON files mein "court" field kabhi poora naam nahi, sirf
    # short abbreviation hoti hai (e.g. "LHC", "SHC") — jo purani checks
    # ("SINDH" in s, "LAHORE" in s) kabhi match nahi karti thin, isliye
    # sab kuch "Unknown" fallback mein gir raha tha. Ab exact-abbreviation
    # match sabse pehle check hota hai (word-boundary ke saath, taake
    # "LHC" kisi lambe string ke andar ghalati se match na ho jaye).
    abbreviations = {
        "LHC": "Lahore High Court",
        "SHC": "High Court of Sindh",
        "PHC": "Peshawar High Court",
        "IHC": "Islamabad High Court",
        "BHC": "Balochistan High Court",
        "SC": "Supreme Court of Pakistan",
        "SCP": "Supreme Court of Pakistan",
    }
    if s in abbreviations:
        return abbreviations[s]

    if "SINDH" in s:
        return "High Court of Sindh"
    if "LAHORE" in s:
        return "Lahore High Court"
    if "SUPREME COURT" in s:
        return "Supreme Court of Pakistan"
    if "BALOCHISTAN" in s:
        return "Balochistan High Court"
    if "PESHAWAR" in s:
        return "Peshawar High Court"
    if "ISLAMABAD" in s:
        return "Islamabad High Court"

    has_court_word = "COURT" in s
    junk_chars = sum(1 for ch in s if ch in "-*/_~^|\\")
    looks_garbled = junk_chars >= 2

    if not has_court_word or looks_garbled:
        return fallback

    return str(raw).strip()


def normalize_case_type(raw, fallback="Unknown"):
    if not raw or not str(raw).strip():
        return fallback

    s = str(raw).upper()
    s_compact = re.sub(r"[^A-Z]", "", s)

    if "BAIL" in s:
        return "Bail"
    if re.search(r"\bB\.?A\.?\b", raw) or "CRBA" in s_compact or "MBA" in s_compact:
        return "Bail"
    if "APPEAL" in s:
        return "Civil Appeal" if "CIVIL" in s else "Criminal Appeal"
    if "REVISION" in s:
        return "Criminal Revision"
    if "WRIT" in s:
        return "Writ Petition"
    if "CONSTITUTION" in s:
        return "Constitutional Petition"
    if "SUIT" in s:
        return "Civil Suit"
    if "REFERENCE" in s:
        return "Reference"
    if "APPLICATION" in s:
        return "Criminal Application"

    # FIX: bohat sari files ka case_type sirf plain "civil" ya "criminal"
    # hota hai (koi extra keyword jaisa appeal/suit/writ ke bagair) —
    # pehle koi bhi condition match nahi karti thi to seedha "Unknown"
    # fallback ban jata tha. Ye check aakhir mein hai (sirf tab chalega
    # jab upar wali specific categories mein se koi match na ho), taake
    # zyada specific info khone ka koi risk na ho.
    if "CIVIL" in s:
        return "Civil Case"
    if "CRIMINAL" in s or re.search(r"\bCR\.?\b", s):
        return "Criminal Case"

    return fallback


# =========================================================
# REPAIR CHECKPOINT (filenames already fixed by repair_metadata.py)
# =========================================================
def load_repaired_filenames() -> Set[str]:
    """
    Reads repair_checkpoint.json and returns a set of BASENAMES
    (e.g. 'LHC_bail_1930_processed_141930.json') that are safe to import.

    Handles a few possible shapes so it doesn't break if the checkpoint
    format changes slightly:
      - a plain JSON list:            ["file1.json", "file2.json", ...]
      - {"completed": [...]}
      - {"fixed": [...], "still_invalid": {...}}
      - {"processed": [...]} / {"done": [...]} / {"repaired": [...]}
    """
    if not os.path.exists(REPAIR_CHECKPOINT_FILE):
        raise FileNotFoundError(
            f"repair_checkpoint.json not found at: {REPAIR_CHECKPOINT_FILE}\n"
            f"Update REPAIR_CHECKPOINT_FILE at the top of this script."
        )

    with open(REPAIR_CHECKPOINT_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    names: Set[str] = set()

    if isinstance(data, list):
        names.update(data)
    elif isinstance(data, dict):
        for key in ("completed", "fixed", "processed", "done", "repaired"):
            value = data.get(key)
            if isinstance(value, list):
                names.update(value)
            elif isinstance(value, dict):
                names.update(value.keys())

    if not names:
        raise ValueError(
            "repair_checkpoint.json was loaded but no filenames were found in it. "
            "Open it and check the top-level structure — this script currently looks "
            "for a top-level list, or keys like 'completed' / 'fixed' / 'processed'."
        )

    # Normalize to basenames only, in case full paths were stored
    return {os.path.basename(n) for n in names}


# =========================================================
# CHECKPOINT (this script's own progress, for resuming)
# =========================================================
def load_checkpoint() -> set:
    if not os.path.exists(CHECKPOINT_FILE):
        return set()

    try:
        with open(CHECKPOINT_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        return set(data.get("completed", []))
    except Exception:
        return set()


def save_checkpoint(completed: set):
    with open(CHECKPOINT_FILE, "w", encoding="utf-8") as f:
        json.dump({"completed": sorted(completed)}, f, indent=2)


# =========================================================
# NEO4J SETUP
# =========================================================
def get_driver():
    if NEO4J_PASSWORD == "YOUR_PASSWORD_HERE":
        raise ValueError(
            "NEO4J_PASSWORD is still set to the placeholder value. "
            "Open this script and set your real Neo4j password before running."
        )

    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

    # Fail fast with a clear message instead of failing after scanning 58k files
    try:
        driver.verify_connectivity()
    except Exception as e:
        driver.close()
        raise ConnectionError(
            f"Could not connect to Neo4j at {NEO4J_URI}. "
            f"Check that Neo4j is running and the URI/user/password are correct.\n"
            f"Original error: {e}"
        )

    return driver


def create_constraints(driver):
    queries = [
        "CREATE CONSTRAINT case_id IF NOT EXISTS FOR (c:Case) REQUIRE c.case_id IS UNIQUE",
        "CREATE CONSTRAINT court_name IF NOT EXISTS FOR (c:Court) REQUIRE c.name IS UNIQUE",
        "CREATE CONSTRAINT judge_name IF NOT EXISTS FOR (j:Judge) REQUIRE j.name IS UNIQUE",
        "CREATE CONSTRAINT topic_name IF NOT EXISTS FOR (t:Topic) REQUIRE t.name IS UNIQUE",
        "CREATE CONSTRAINT section_name IF NOT EXISTS FOR (s:LawSection) REQUIRE s.name IS UNIQUE",
        "CREATE CONSTRAINT party_name IF NOT EXISTS FOR (p:Party) REQUIRE p.name IS UNIQUE",
        # NEW: Chunk nodes need a unique id so MERGE doesn't create duplicate
        # chunks on re-import/resume. chunk_id is derived from case_id +
        # chunk_index (see build_chunk_records()) so it's stable and unique
        # even across cases.
        "CREATE CONSTRAINT chunk_id IF NOT EXISTS FOR (ch:Chunk) REQUIRE ch.chunk_id IS UNIQUE",
        # NEW: Year is now a first-class node (DECIDED_IN relationship)
        # instead of just a plain property on Case, so queries like
        # "all cases decided in 2019" can traverse the graph directly.
        "CREATE CONSTRAINT year_value IF NOT EXISTS FOR (y:Year) REQUIRE y.value IS UNIQUE",
    ]

    with driver.session() as session:
        for q in queries:
            session.run(q)


# =========================================================
# HELPERS
# =========================================================
def normalize_str(value):
    if value is None:
        return ""
    return str(value).strip()


# =========================================================
# FIX: JSON files ki actual keys Excel-style hain (e.g. "Court",
# "Case Type", "Year" - capital, space) — script pehle sirf lowercase
# snake_case keys ("court", "case_type") dhoond rahi thi, jo kabhi match
# hi nahi hui, isliye data.get() hamesha None deta tha aur sab kuch
# normalize_court()/normalize_case_type() ke fallback ("Unknown") mein
# gir jata tha. get_field() ab ek saath saari possible key-spellings try
# karta hai (jo bhi pehle mile), taake chahe file mein "Court" ho,
# "court" ho, ya "Court Name" ho — sahi value uth jaye.
# =========================================================
def get_field(data: Dict[str, Any], *possible_keys):
    for key in possible_keys:
        if key in data and data.get(key) not in (None, ""):
            return data.get(key)
    return None


# =========================================================
# NEW: build_case_label() — produces a human-readable identity for the
# Case node, used ONLY as a display/caption property in Neo4j Browser.
#
# This is intentionally separate from case_id (the MERGE key, which stays
# filename-based — see the comment in build_case_record() explaining why
# case_number cannot be used as the identity key). case_label is not
# guaranteed unique and is never used in MERGE, so it carries zero risk
# of causing false-merge bugs like the ones case_number caused before.
#
# Priority order:
#   1) case_number, if present (e.g. "2019 PLD SC 45")
#   2) "<case_type> - <court> - <year>" built from whatever normalized
#      fields are available (skips any that are missing/"Unknown")
#   3) cleaned-up filename as an absolute last resort, so the graph never
#      shows a blank caption, but this path should rarely be hit.
# =========================================================
def build_case_label(case_number: str, case_type: str, court: str, year: str,
                      generated_name: str) -> str:
    if case_number:
        return case_number

    parts = [p for p in (case_type, court, year) if p and p != "Unknown"]
    if parts:
        return " - ".join(parts)

    if generated_name:
        cleaned = os.path.splitext(generated_name)[0].replace("_", " ").strip()
        if cleaned:
            return cleaned

    return "Unlabeled Case"


# =========================================================
# NEW: build_chunk_records() — JSON "chunks" array ke har object ko ek
# Chunk node record mein convert karta hai. chunk_id = case_id + chunk_index
# se banaya jata hai taake:
#   1) har chunk globally unique ho (do alag cases ke chunk_index=0 aapas
#      mein clash na karein), aur
#   2) MERGE resume-safe rahe — dobara import chalane par same chunk_id
#      se wahi Chunk node dobara MERGE hoga, duplicate nahi banega.
# source_pages ek list hoti hai, Neo4j properties list of primitives
# accept karta hai isliye seedha store ho sakti hai.
# =========================================================
def build_chunk_records(case_id: str, data: Dict[str, Any]) -> List[Dict[str, Any]]:
    raw_chunks = get_field(data, "chunks", "Chunks") or []
    if not isinstance(raw_chunks, list):
        return []

    chunk_records = []
    for chunk in raw_chunks:
        if not isinstance(chunk, dict):
            continue

        chunk_index = chunk.get("chunk_index")
        if chunk_index is None:
            # skip malformed chunk entries rather than fabricating an index
            continue

        chunk_id = f"{case_id}::chunk_{chunk_index}"

        source_pages = chunk.get("source_pages") or []
        if not isinstance(source_pages, list):
            source_pages = [source_pages]

        chunk_records.append({
            "chunk_id": chunk_id,
            "chunk_index": int(chunk_index),
            "text": normalize_str(chunk.get("text")),
            "token_estimate": int(chunk.get("token_estimate") or 0),
            "source_pages": [int(p) for p in source_pages if str(p).strip().lstrip("-").isdigit()],
        })

    return chunk_records


def build_case_record(data: Dict[str, Any]) -> Dict[str, Any]:
    case_number = normalize_str(get_field(data, "case_number", "Case Number", "CaseNumber"))
    generated_name = normalize_str(get_field(data, "generated_name", "Generated Name", "File Name", "generated_filename"))
    actual_filename = normalize_str(get_field(data, "actual_filename", "Actual File Path", "Actual Filename", "actual_file_path"))

    # IMPORTANT: case_id is based on the FILENAME, not case_number.
    # case_number extraction has known bugs (garbage/repeated values across
    # unrelated documents), which was causing unrelated cases to incorrectly
    # MERGE into a single node. The filename is guaranteed unique per source
    # document, so using it as the identity key prevents false merges.
    # case_number is kept as a plain property for search/reference only.
    case_id = actual_filename if actual_filename else generated_name

    # FIX: raw case_type -> normalize_case_type(), taake Topic nodes
    # Excel wali standard categories (Bail, Civil Appeal, Criminal
    # Appeal, Writ Petition, waghera) mein map hon, raw/inconsistent
    # strings mein nahi.
    case_type = normalize_case_type(get_field(data, "case_type", "Case Type", "CaseType"), fallback="Unknown")

    # FIX: raw court -> normalize_court(), taake OCR-garbled variants
    # ("HIGG GOURT OF STANDH...", "High Court of Indh", etc.) ek hi
    # clean Court node mein merge hon.
    court = normalize_court(get_field(data, "court", "Court", "Court Name"), fallback="Unknown")

    year = normalize_str(get_field(data, "year", "Year"))

    # NEW: case_label — see build_case_label() docstring/comment above.
    # This is what makes the Case node readable in Neo4j Browser
    # ("2019 PLD SC 45" instead of "LHC_bail_1930_processed_141930.pdf")
    # without disturbing the filename-based case_id used for MERGE.
    case_label = build_case_label(case_number, case_type, court, year, generated_name)

    return {
        "case_id": case_id,
        "case_label": case_label,
        "case_number": case_number,
        "generated_name": generated_name,
        "actual_filename": actual_filename,
        "case_type": case_type,
        "year": year,
        "date_of_order": normalize_str(get_field(data, "date_of_order", "Date of Order", "DateOfOrder")),
        "used_ocr": bool(get_field(data, "used_ocr", "Used OCR", "usedOcr") or False),
        "num_chunks": int(get_field(data, "num_chunks", "Num Chunks", "numChunks") or 0),
        "court": court,
        "judge": normalize_str(get_field(data, "judge", "Judge", "Judge Name")),
        "sections_cited": [normalize_str(x) for x in (get_field(data, "sections_cited", "Sections Cited", "Keywords") or []) if normalize_str(x)],
        "citations": [normalize_str(x) for x in (get_field(data, "citations", "Citations") or []) if normalize_str(x)],
        "parties": [normalize_str(x) for x in (get_field(data, "parties", "Parties") or []) if normalize_str(x)],
        # NEW: per-case list of Chunk records built from the JSON "chunks"
        # array. Each entry becomes its own Chunk node in Neo4j, linked to
        # this Case via HAS_CHUNK. This is what lets future Qdrant-based
        # retrieval attach (Query)-[:SIMILAR]->(Chunk) relationships later,
        # since retrieval works at chunk granularity, not whole-case.
        "chunks": build_chunk_records(case_id, data),
    }


# =========================================================
# BATCH IMPORT QUERY
#
# NEW Legal Knowledge Graph shape:
#   (Case)-[:HAS_CHUNK]->(Chunk)
#   (Case)-[:DECIDED_BY]->(Judge)
#   (Case)-[:HEARD_IN]->(Court)
#   (Case)-[:HAS_TOPIC]->(Topic)      <- renamed from BELONGS_TO
#   (Case)-[:DECIDED_IN]->(Year)      <- NEW, Year is now its own node
#   (Case)-[:INVOLVES]->(Party)
#   (Case)-[:APPLIES]->(LawSection)
#   (Case)-[:CITES]->(Case)
# =========================================================
BATCH_QUERY = """
UNWIND $rows AS row

MERGE (c:Case {case_id: row.case_id})
SET c.case_number    = row.case_number,
    c.case_label      = row.case_label,
    c.generated_name = row.generated_name,
    c.actual_filename= row.actual_filename,
    c.case_type      = row.case_type,
    c.year           = row.year,
    c.date_of_order  = row.date_of_order,
    c.used_ocr       = row.used_ocr,
    c.num_chunks     = row.num_chunks

FOREACH (_ IN CASE WHEN row.court <> '' THEN [1] ELSE [] END |
    MERGE (co:Court {name: row.court})
    MERGE (c)-[:HEARD_IN]->(co)
)

FOREACH (_ IN CASE WHEN row.judge <> '' THEN [1] ELSE [] END |
    MERGE (j:Judge {name: row.judge})
    MERGE (c)-[:DECIDED_BY]->(j)
)

FOREACH (_ IN CASE WHEN row.case_type <> '' THEN [1] ELSE [] END |
    MERGE (t:Topic {name: row.case_type})
    MERGE (c)-[:HAS_TOPIC]->(t)
)

// NEW: Year node + DECIDED_IN relationship
FOREACH (_ IN CASE WHEN row.year <> '' THEN [1] ELSE [] END |
    MERGE (y:Year {value: row.year})
    MERGE (c)-[:DECIDED_IN]->(y)
)

FOREACH (sec IN row.sections_cited |
    MERGE (s:LawSection {name: sec})
    MERGE (c)-[:APPLIES]->(s)
)

FOREACH (p IN row.parties |
    MERGE (pt:Party {name: p})
    MERGE (c)-[:INVOLVES]->(pt)
)

FOREACH (cit IN row.citations |
    MERGE (cited:Case {case_id: cit})
    MERGE (c)-[:CITES]->(cited)
)

// NEW: Chunk nodes, one per entry in row.chunks, linked via HAS_CHUNK.
// chunk_id is the MERGE key so re-running the import (resume / incremental
// growth) never creates duplicate chunks.
FOREACH (ch IN row.chunks |
    MERGE (chunk:Chunk {chunk_id: ch.chunk_id})
    SET chunk.chunk_index    = ch.chunk_index,
        chunk.text           = ch.text,
        chunk.token_estimate = ch.token_estimate,
        chunk.source_pages   = ch.source_pages
    MERGE (c)-[:HAS_CHUNK]->(chunk)
)
"""


def import_batch(driver, rows: List[Dict[str, Any]]):
    """Imports one batch, retrying on transient Neo4j connection issues."""
    delay = RETRY_BASE_DELAY
    last_error = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            with driver.session() as session:
                session.execute_write(lambda tx: tx.run(BATCH_QUERY, rows=rows).consume())
            return
        except (ServiceUnavailable, SessionExpired, TransientError) as e:
            last_error = e
            if attempt == MAX_RETRIES:
                break
            print(f"⚠️  Neo4j connection issue (attempt {attempt}/{MAX_RETRIES}): {e}. "
                  f"Retrying in {delay}s...")
            time.sleep(delay)
            delay *= 2

    # exhausted retries
    raise last_error


# =========================================================
# MAIN
# =========================================================
def main():
    print("🚀 Neo4j Import Started (repaired files only)")

    driver = None
    try:
        driver = get_driver()
        create_constraints(driver)

        # --- Filter to only files that repair_metadata.py has verified ---
        repaired_names = load_repaired_filenames()
        print(f"📋 Repaired filenames in checkpoint: {len(repaired_names)}")

        all_files = [f for f in Path(JSON_FOLDER).rglob("*.json") if f.name in repaired_names]

        found_names = {f.name for f in all_files}
        missing_on_disk = repaired_names - found_names
        if missing_on_disk:
            print(f"⚠️  {len(missing_on_disk)} filenames from checkpoint were NOT found in {JSON_FOLDER}")
            log_error(f"Missing on disk ({len(missing_on_disk)}): {sorted(missing_on_disk)}")

        completed = load_checkpoint()
        total_files = len(all_files)
        pending_files = [f for f in all_files if str(f) not in completed]

        print(f"📁 Repaired files matched on disk : {total_files}")
        print(f"✅ Already imported (this script)  : {len(completed)}")
        print(f"⏳ Remaining                        : {len(pending_files)}\n")

        imported = 0
        skipped = 0
        failed = 0
        # FIX: agar zyada tar records "Unknown" ban rahe hain, ho sakta hai
        # JSON key-names dobara mismatch ho gaye hon (jaisa is baar hua) —
        # ye counter aisi problem ko turant pakadne mein madad karega.
        unknown_court_count = 0
        # NEW: total chunk nodes created this run, printed in the summary
        # so it's easy to sanity-check chunk import is actually happening.
        total_chunks = 0
        # NEW: counts cases where JSON's stated "num_chunks" doesn't match
        # the actual length of the "chunks" array we built Chunk nodes
        # from. Mismatches get logged to LOG_FILE with the filename so they
        # can be traced back to specific files (e.g. stale metadata from an
        # earlier extraction run).
        chunk_count_mismatches = 0

        batch = []
        batch_paths = []

        pbar = tqdm(total=len(pending_files), desc="Importing", unit="file")

        for file_path in pending_files:

            if STOP_REQUESTED:
                break

            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)

                record = build_case_record(data)

                if record["court"] == "Unknown":
                    unknown_court_count += 1

                if not record["case_id"]:
                    skipped += 1
                    completed.add(str(file_path))
                    pbar.update(1)
                    continue

                actual_chunk_count = len(record["chunks"])
                total_chunks += actual_chunk_count

                # NEW: flag when JSON's own "num_chunks" field disagrees
                # with how many chunk objects we actually found/parsed.
                # This does NOT block the import — it only logs — since a
                # mismatch is a metadata quality issue, not a reason to
                # skip importing the chunks that ARE present.
                if record["num_chunks"] != actual_chunk_count:
                    chunk_count_mismatches += 1
                    log_error(
                        f"{file_path} | chunk count mismatch: "
                        f"num_chunks field={record['num_chunks']} "
                        f"actual chunks parsed={actual_chunk_count}"
                    )

                batch.append(record)
                batch_paths.append(str(file_path))

                if len(batch) >= BATCH_SIZE:
                    import_batch(driver, batch)

                    for p in batch_paths:
                        completed.add(p)

                    save_checkpoint(completed)

                    imported += len(batch)
                    batch.clear()
                    batch_paths.clear()

            except Exception as e:
                failed += 1
                log_error(f"{file_path} | {type(e).__name__}: {e}")

            pbar.set_postfix({
                "imported": imported,
                "skipped": skipped,
                "failed": failed
            })

            pbar.update(1)

        # Import remaining records
        if batch and not STOP_REQUESTED:
            try:
                import_batch(driver, batch)

                for p in batch_paths:
                    completed.add(p)

                save_checkpoint(completed)

                imported += len(batch)

            except Exception as e:
                failed += len(batch)
                log_error(f"FINAL BATCH | {type(e).__name__}: {e}")

        pbar.close()
        save_checkpoint(completed)

        print("\n📊 Import Summary")
        print(f"Imported : {imported}")
        print(f"Skipped  : {skipped}")
        print(f"Failed   : {failed}")
        print(f"Chunks   : {total_chunks}")
        print(f"Completed: {len(completed)} / {total_files}")
        print(f"Court = 'Unknown': {unknown_court_count} / {imported + skipped}")
        print(f"num_chunks mismatches: {chunk_count_mismatches} / {imported + skipped}")

        if imported > 0 and unknown_court_count > 0.5 * imported:
            print("\n⚠️  50% se zyada records mein Court 'Unknown' hai — mumkin hai")
            print("    JSON files ki keys get_field() ki list mein match nahi ho rahi.")
            print("    Ek JSON file khol kar uski actual keys check karo.")

        if chunk_count_mismatches > 0:
            print(f"\n⚠️  {chunk_count_mismatches} files mein JSON ka 'num_chunks' field")
            print("    aur actual 'chunks' array ki length match nahi karti.")
            print(f"    Details {LOG_FILE} mein 'chunk count mismatch' lines mein hain.")

        if STOP_REQUESTED:
            print("\n🛑 Import stopped by user. Resume by running the same command again.")
        else:
            print("\n🎉 Neo4j import completed successfully.")

    finally:
        if driver is not None:
            driver.close()


if __name__ == "__main__":
    main()