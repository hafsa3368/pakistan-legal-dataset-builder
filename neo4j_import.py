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

    return {
        "case_id": case_id,
        "case_number": case_number,
        "generated_name": generated_name,
        "actual_filename": actual_filename,
        # FIX: raw case_type -> normalize_case_type(), taake Topic nodes
        # Excel wali standard categories (Bail, Civil Appeal, Criminal
        # Appeal, Writ Petition, waghera) mein map hon, raw/inconsistent
        # strings mein nahi.
        "case_type": normalize_case_type(get_field(data, "case_type", "Case Type", "CaseType"), fallback="Unknown"),
        "year": normalize_str(get_field(data, "year", "Year")),
        "date_of_order": normalize_str(get_field(data, "date_of_order", "Date of Order", "DateOfOrder")),
        "used_ocr": bool(get_field(data, "used_ocr", "Used OCR", "usedOcr") or False),
        "num_chunks": int(get_field(data, "num_chunks", "Num Chunks", "numChunks") or 0),
        # FIX: raw court -> normalize_court(), taake OCR-garbled variants
        # ("HIGG GOURT OF STANDH...", "High Court of Indh", etc.) ek hi
        # clean Court node mein merge hon.
        "court": normalize_court(get_field(data, "court", "Court", "Court Name"), fallback="Unknown"),
        "judge": normalize_str(get_field(data, "judge", "Judge", "Judge Name")),
        "sections_cited": [normalize_str(x) for x in (get_field(data, "sections_cited", "Sections Cited", "Keywords") or []) if normalize_str(x)],
        "citations": [normalize_str(x) for x in (get_field(data, "citations", "Citations") or []) if normalize_str(x)],
        "parties": [normalize_str(x) for x in (get_field(data, "parties", "Parties") or []) if normalize_str(x)],
    }


# =========================================================
# BATCH IMPORT QUERY
# =========================================================
BATCH_QUERY = """
UNWIND $rows AS row

MERGE (c:Case {case_id: row.case_id})
SET c.case_number    = row.case_number,
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
    MERGE (c)-[:BELONGS_TO]->(t)
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
        print(f"Completed: {len(completed)} / {total_files}")
        print(f"Court = 'Unknown': {unknown_court_count} / {imported + skipped}")

        if imported > 0 and unknown_court_count > 0.5 * imported:
            print("\n⚠️  50% se zyada records mein Court 'Unknown' hai — mumkin hai")
            print("    JSON files ki keys get_field() ki list mein match nahi ho rahi.")
            print("    Ek JSON file khol kar uski actual keys check karo.")

        if STOP_REQUESTED:
            print("\n🛑 Import stopped by user. Resume by running the same command again.")
        else:
            print("\n🎉 Neo4j import completed successfully.")

    finally:
        if driver is not None:
            driver.close()


if __name__ == "__main__":
    main()