import os
import json
import re
import signal
import time
from pathlib import Path
from typing import List, Dict, Any, Set, Optional
from dotenv import load_dotenv
from neo4j import GraphDatabase
from neo4j.exceptions import ServiceUnavailable, SessionExpired, TransientError
from qdrant_client import QdrantClient
from tqdm import tqdm

# =========================================================
# CONFIG  --->  SIRF YAHAN CHEEZEIN CHANGE KARNI HAIN <---
# =========================================================
#C:\Users\Hamza\Desktop\qdrant
# .\qdrant.exe
# python neo4j_import.py
# 1) Ye folder jahan aapki extracted JSON files pari hain (data yahan se aayega)
JSON_FOLDER = r"D:\hafsa_thesis material\supreme_court_scraper\extracted_text_clean"

# 2) repair_checkpoint.json ka poora path (jo repair_metadata.py ne banayi thi)
#    Agar ye file kisi aur folder mein hai toh yahan sahi path daalein.
REPAIR_CHECKPOINT_FILE = r"D:\hafsa_thesis material\supreme_court_scraper\repair_checkpoint.json"

# =========================================================
# NEO4J LOGIN
# =========================================================
load_dotenv()  # <--- YAHAN apna real password daalein
import os
import json
import re
import signal
import time
from pathlib import Path
from typing import List, Dict, Any, Set, Optional

from neo4j import GraphDatabase
from neo4j.exceptions import ServiceUnavailable, SessionExpired, TransientError
from qdrant_client import QdrantClient
from tqdm import tqdm


# =========================================================
# CONFIG
# =========================================================

JSON_FOLDER = r"D:\hafsa_thesis material\supreme_court_scraper\extracted_text_clean"

REPAIR_CHECKPOINT_FILE = (
    r"D:\hafsa_thesis material\supreme_court_scraper\repair_checkpoint.json"
)

# =========================================================
# NEO4J LOGIN
# =========================================================

NEO4J_URI = "bolt://127.0.0.1:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

# =========================================================
# QDRANT
# READ ONLY
# =========================================================

QDRANT_HOST = "127.0.0.1"  # force IPv4 -- "localhost" intermittently resolves to
                            # IPv6 ::1 on this machine, where nothing is listening
                            # (Qdrant binds 0.0.0.0 / IPv4 only), which hangs the
                            # connection forever instead of failing fast
QDRANT_PORT = 6333
QDRANT_COLLECTION = "legal_chunks"
QDRANT_SCROLL_BATCH = 500

# =========================================================
# IMPORT SETTINGS
# =========================================================

BATCH_SIZE = 100

MAX_RETRIES = 3
RETRY_BASE_DELAY = 2

CHECKPOINT_FILE = "neo4j_import_checkpoint.json"
LOG_FILE = "neo4j_import_errors.log"


# =========================================================
# GLOBAL STOP FLAG
# =========================================================

STOP_REQUESTED = False


def handle_sigint(signum, frame):
    global STOP_REQUESTED
    STOP_REQUESTED = True

    print(
        "\nCtrl+C received. "
        "Finishing current batch and saving checkpoint..."
    )


signal.signal(signal.SIGINT, handle_sigint)


# =========================================================
# LOGGING
# =========================================================

def log_error(message: str):

    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(message + "\n")


# =========================================================
# NORMALIZE COURT
# =========================================================

def normalize_court(raw, fallback="Unknown"):

    if not raw or not str(raw).strip():
        return fallback

    s = str(raw).upper().strip()

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

    junk_chars = sum(
        1 for ch in s
        if ch in "-*/_~^|\\"
    )

    looks_garbled = junk_chars >= 2

    if not has_court_word or looks_garbled:
        return fallback

    return str(raw).strip()


# =========================================================
# NORMALIZE CASE TYPE
# =========================================================

def normalize_case_type(raw, fallback="Unknown"):

    if not raw or not str(raw).strip():
        return fallback

    s = str(raw).upper()
    s_compact = re.sub(r"[^A-Z]", "", s)

    if "BAIL" in s:
        return "Bail"

    if re.search(r"\bB\.?A\.?\b", str(raw)) \
            or "CRBA" in s_compact \
            or "MBA" in s_compact:
        return "Bail"

    if "APPEAL" in s:
        if "CIVIL" in s:
            return "Civil Appeal"
        return "Criminal Appeal"

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

    if "CIVIL" in s:
        return "Civil Case"

    if "CRIMINAL" in s or re.search(r"\bCR\.?\b", s):
        return "Criminal Case"

    return fallback


# =========================================================
# LOAD REPAIR CHECKPOINT
# =========================================================

def load_repaired_filenames() -> Set[str]:

    if not os.path.exists(REPAIR_CHECKPOINT_FILE):

        raise FileNotFoundError(
            f"\nrepair_checkpoint.json not found:\n"
            f"{REPAIR_CHECKPOINT_FILE}\n\n"
            f"Please check REPAIR_CHECKPOINT_FILE at the top."
        )

    with open(
        REPAIR_CHECKPOINT_FILE,
        "r",
        encoding="utf-8"
    ) as f:

        data = json.load(f)

    names: Set[str] = set()

    if isinstance(data, list):

        names.update(data)

    elif isinstance(data, dict):

        for key in (
            "completed",
            "fixed",
            "processed",
            "done",
            "repaired",
        ):

            value = data.get(key)

            if isinstance(value, list):

                names.update(value)

            elif isinstance(value, dict):

                names.update(value.keys())

    if not names:

        raise ValueError(
            "\nrepair_checkpoint.json was loaded, "
            "but no filenames were found."
        )

    return {
        os.path.basename(n)
        for n in names
    }


# =========================================================
# BUILD QDRANT CASE_ID MAPPING
#
# Qdrant is the SOURCE OF TRUTH for case_id.
#
# Qdrant is READ ONLY here.
# =========================================================

def build_qdrant_case_id_mapping() -> Dict[str, str]:

    print(
        "\nFetching finalized case_id mapping "
        "from Qdrant (READ ONLY)..."
    )

    client = QdrantClient(
        host=QDRANT_HOST,
        port=QDRANT_PORT
    )

    existing = [
        c.name
        for c in client.get_collections().collections
    ]

    if QDRANT_COLLECTION not in existing:

        raise RuntimeError(
            f"\nQdrant collection '{QDRANT_COLLECTION}' "
            f"was not found.\n"
            f"Available collections: {existing}"
        )

    mapping: Dict[str, str] = {}

    conflicts = 0
    missing_case_id = 0

    offset = None

    while True:

        points, next_offset = client.scroll(
            collection_name=QDRANT_COLLECTION,
            limit=QDRANT_SCROLL_BATCH,
            offset=offset,
            with_payload=[
                "case_id",
                "actual_filename",
                "generated_name",
            ],
            with_vectors=False,
        )

        if not points:
            break

        for point in points:

            payload = point.payload or {}

            case_id = payload.get("case_id")

            if not case_id:

                missing_case_id += 1

                continue

            case_id = str(case_id).strip()

            actual_filename = str(
                payload.get("actual_filename") or ""
            ).strip()

            generated_name = str(
                payload.get("generated_name") or ""
            ).strip()

            key = (
                actual_filename
                if actual_filename
                else generated_name
            )

            if not key:
                continue

            if key in mapping:

                if mapping[key] != case_id:

                    conflicts += 1

                    log_error(
                        "QDRANT MAPPING CONFLICT | "
                        f"key={key!r} | "
                        f"existing={mapping[key]!r} | "
                        f"new={case_id!r}"
                    )

                    continue

            else:

                mapping[key] = case_id

        offset = next_offset

        if offset is None:
            break

    print(
        f"Qdrant mapping built successfully: "
        f"{len(mapping)} unique filename keys"
    )

    if missing_case_id:

        print(
            f"WARNING: {missing_case_id} Qdrant points "
            f"had no case_id."
        )

    if conflicts:

        print(
            f"WARNING: {conflicts} conflicting mappings found."
        )

    return mapping


# =========================================================
# IMPORT CHECKPOINT
# =========================================================

def load_checkpoint() -> Set[str]:

    if not os.path.exists(CHECKPOINT_FILE):

        return set()

    try:

        with open(
            CHECKPOINT_FILE,
            "r",
            encoding="utf-8"
        ) as f:

            data = json.load(f)

        return set(
            data.get("completed", [])
        )

    except Exception:

        return set()


def save_checkpoint(completed: Set[str]):

    with open(
        CHECKPOINT_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            {
                "completed": sorted(completed)
            },
            f,
            indent=2
        )


# =========================================================
# NEO4J CONNECTION
# =========================================================

def get_driver():

    if NEO4J_PASSWORD == "YOUR_PASSWORD_HERE":

        raise ValueError(
            "NEO4J_PASSWORD is still a placeholder."
        )

    driver = GraphDatabase.driver(
        NEO4J_URI,
        auth=(
            NEO4J_USER,
            NEO4J_PASSWORD
        )
    )

    try:

        driver.verify_connectivity()

    except Exception as e:

        driver.close()

        raise ConnectionError(
            f"\nCould not connect to Neo4j.\n"
            f"URI: {NEO4J_URI}\n"
            f"Original error: {e}"
        )

    print("Connected to Neo4j")

    return driver


# =========================================================
# NEO4J CONSTRAINTS
# =========================================================

def create_constraints(driver):

    queries = [

        # Actual legal cases
        """
        CREATE CONSTRAINT case_id IF NOT EXISTS
        FOR (c:Case)
        REQUIRE c.case_id IS UNIQUE
        """,

        # Courts
        """
        CREATE CONSTRAINT court_name IF NOT EXISTS
        FOR (c:Court)
        REQUIRE c.name IS UNIQUE
        """,

        # Judges
        """
        CREATE CONSTRAINT judge_name IF NOT EXISTS
        FOR (j:Judge)
        REQUIRE j.name IS UNIQUE
        """,

        # Topics
        """
        CREATE CONSTRAINT topic_name IF NOT EXISTS
        FOR (t:Topic)
        REQUIRE t.name IS UNIQUE
        """,

        # Law sections
        """
        CREATE CONSTRAINT section_name IF NOT EXISTS
        FOR (s:LawSection)
        REQUIRE s.name IS UNIQUE
        """,

        # Parties
        """
        CREATE CONSTRAINT party_name IF NOT EXISTS
        FOR (p:Party)
        REQUIRE p.name IS UNIQUE
        """,

        # Chunks
        """
        CREATE CONSTRAINT chunk_id IF NOT EXISTS
        FOR (ch:Chunk)
        REQUIRE ch.chunk_id IS UNIQUE
        """,

        # Years
        """
        CREATE CONSTRAINT year_value IF NOT EXISTS
        FOR (y:Year)
        REQUIRE y.value IS UNIQUE
        """,

        # Legal citations
        # IMPORTANT:
        # Citation is NOT a Case.
        """
        CREATE CONSTRAINT citation_id IF NOT EXISTS
        FOR (c:Citation)
        REQUIRE c.citation_id IS UNIQUE
        """
    ]

    with driver.session() as session:

        for query in queries:

            session.run(query)

    print("Neo4j constraints created/verified.")


# =========================================================
# HELPERS
# =========================================================

def normalize_str(value):

    if value is None:
        return ""

    return str(value).strip()


def get_field(
    data: Dict[str, Any],
    *possible_keys
):

    for key in possible_keys:

        if (
            key in data
            and data.get(key) not in (None, "")
        ):

            return data.get(key)

    return None


# =========================================================
# CASE LABEL
# =========================================================

def build_case_label(
    case_number: str,
    case_type: str,
    court: str,
    year: str,
    generated_name: str
) -> str:

    if case_number:

        return case_number

    parts = [
        p
        for p in (
            case_type,
            court,
            year
        )
        if p and p != "Unknown"
    ]

    if parts:

        return " - ".join(parts)

    if generated_name:

        cleaned = (
            os.path.splitext(
                generated_name
            )[0]
            .replace("_", " ")
            .strip()
        )

        if cleaned:

            return cleaned

    return "Unlabeled Case"


# =========================================================
# BUILD CHUNK RECORDS
# =========================================================

def build_chunk_records(
    case_id: str,
    data: Dict[str, Any]
) -> List[Dict[str, Any]]:

    raw_chunks = get_field(
        data,
        "chunks",
        "Chunks"
    ) or []

    if not isinstance(raw_chunks, list):

        return []

    chunk_records = []

    for chunk in raw_chunks:

        if not isinstance(chunk, dict):

            continue

        chunk_index = chunk.get(
            "chunk_index"
        )

        if chunk_index is None:

            continue

        chunk_id = (
            f"{case_id}::chunk_{chunk_index}"
        )

        source_pages = (
            chunk.get("source_pages")
            or []
        )

        if not isinstance(
            source_pages,
            list
        ):

            source_pages = [
                source_pages
            ]

        clean_pages = []

        for p in source_pages:

            if str(p).strip().lstrip("-").isdigit():

                clean_pages.append(
                    int(p)
                )

        chunk_records.append(
            {
                "chunk_id": chunk_id,

                "chunk_index": int(
                    chunk_index
                ),

                "text": normalize_str(
                    chunk.get("text")
                ),

                "token_estimate": int(
                    chunk.get(
                        "token_estimate"
                    ) or 0
                ),

                "source_pages": clean_pages,
            }
        )

    return chunk_records


# =========================================================
# RESOLVE CASE_ID FROM QDRANT
# =========================================================

def resolve_case_id_from_qdrant(
    data: Dict[str, Any],
    qdrant_mapping: Dict[str, str]
) -> Optional[str]:

    generated_name = normalize_str(
        get_field(
            data,
            "generated_name",
            "Generated Name",
            "File Name",
            "generated_filename"
        )
    )

    actual_filename = normalize_str(
        get_field(
            data,
            "actual_filename",
            "Actual File Path",
            "Actual Filename",
            "actual_file_path"
        )
    )

    # First try actual_filename
    if (
        actual_filename
        and actual_filename in qdrant_mapping
    ):

        return qdrant_mapping[
            actual_filename
        ]

    # Then generated_name
    if (
        generated_name
        and generated_name in qdrant_mapping
    ):

        return qdrant_mapping[
            generated_name
        ]

    # IMPORTANT:
    # Do NOT invent a case_id.
    return None


# =========================================================
# BUILD CASE RECORD
# =========================================================

def build_case_record(
    data: Dict[str, Any],
    case_id: str
) -> Dict[str, Any]:

    case_number = normalize_str(
        get_field(
            data,
            "case_number",
            "Case Number",
            "CaseNumber"
        )
    )

    generated_name = normalize_str(
        get_field(
            data,
            "generated_name",
            "Generated Name",
            "File Name",
            "generated_filename"
        )
    )

    actual_filename = normalize_str(
        get_field(
            data,
            "actual_filename",
            "Actual File Path",
            "Actual Filename",
            "actual_file_path"
        )
    )

    case_type = normalize_case_type(
        get_field(
            data,
            "case_type",
            "Case Type",
            "CaseType"
        ),
        fallback="Unknown"
    )

    court = normalize_court(
        get_field(
            data,
            "court",
            "Court",
            "Court Name"
        ),
        fallback="Unknown"
    )

    year = normalize_str(
        get_field(
            data,
            "year",
            "Year"
        )
    )

    case_label = build_case_label(
        case_number,
        case_type,
        court,
        year,
        generated_name
    )

    sections = (
        get_field(
            data,
            "sections_cited",
            "Sections Cited",
            "Keywords"
        )
        or []
    )

    citations = (
        get_field(
            data,
            "citations",
            "Citations"
        )
        or []
    )

    parties = (
        get_field(
            data,
            "parties",
            "Parties"
        )
        or []
    )

    return {

        "case_id": case_id,

        "case_label": case_label,

        "case_number": case_number,

        "generated_name": generated_name,

        "actual_filename": actual_filename,

        "case_type": case_type,

        "year": year,

        "date_of_order": normalize_str(
            get_field(
                data,
                "date_of_order",
                "Date of Order",
                "DateOfOrder"
            )
        ),

        "used_ocr": bool(
            get_field(
                data,
                "used_ocr",
                "Used OCR",
                "usedOcr"
            )
            or False
        ),

        "num_chunks": int(
            get_field(
                data,
                "num_chunks",
                "Num Chunks",
                "numChunks"
            )
            or 0
        ),

        "court": court,

        "judge": normalize_str(
            get_field(
                data,
                "judge",
                "Judge",
                "Judge Name"
            )
        ),

        "sections_cited": [
            normalize_str(x)
            for x in sections
            if normalize_str(x)
        ],

        "citations": [
            normalize_str(x)
            for x in citations
            if normalize_str(x)
        ],

        "parties": [
            normalize_str(x)
            for x in parties
            if normalize_str(x)
        ],

        "chunks": build_chunk_records(
            case_id,
            data
        )
    }


# =========================================================
# NEO4J BATCH QUERY
#
# IMPORTANT CORRECTION:
#
# citations are Citation nodes.
# They are NOT Case nodes.
# =========================================================

BATCH_QUERY = """
UNWIND $rows AS row

MERGE (c:Case {case_id: row.case_id})

SET
    c.case_number = row.case_number,
    c.case_label = row.case_label,
    c.generated_name = row.generated_name,
    c.actual_filename = row.actual_filename,
    c.case_type = row.case_type,
    c.year = row.year,
    c.date_of_order = row.date_of_order,
    c.used_ocr = row.used_ocr,
    c.num_chunks = row.num_chunks,
    c.judge_name = row.judge


FOREACH (_ IN CASE
    WHEN row.court <> ''
    AND row.court <> 'Unknown'
    THEN [1]
    ELSE []
END |

    MERGE (co:Court {name: row.court})

    MERGE (c)-[:HEARD_IN]->(co)
)


FOREACH (_ IN CASE
    WHEN row.judge <> ''
    AND row.judge <> 'Unknown'
    THEN [1]
    ELSE []
END |

    MERGE (j:Judge {name: row.judge})

    MERGE (c)-[:DECIDED_BY]->(j)
)


FOREACH (_ IN CASE
    WHEN row.case_type <> ''
    AND row.case_type <> 'Unknown'
    THEN [1]
    ELSE []
END |

    MERGE (t:Topic {name: row.case_type})

    MERGE (c)-[:HAS_TOPIC]->(t)
)


FOREACH (_ IN CASE
    WHEN row.year <> ''
    AND row.year <> 'Unknown'
    THEN [1]
    ELSE []
END |

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

    MERGE (citation:Citation {
        citation_id: cit
    })

    MERGE (c)-[:CITES]->(citation)
)


FOREACH (ch IN row.chunks |

    MERGE (chunk:Chunk {
        chunk_id: ch.chunk_id
    })

    SET
        chunk.chunk_index = ch.chunk_index,
        chunk.text = ch.text,
        chunk.token_estimate = ch.token_estimate,
        chunk.source_pages = ch.source_pages

    MERGE (c)-[:HAS_CHUNK]->(chunk)
)
"""


# =========================================================
# IMPORT BATCH
# =========================================================

def import_batch(
    driver,
    rows: List[Dict[str, Any]]
):

    delay = RETRY_BASE_DELAY
    last_error = None

    for attempt in range(
        1,
        MAX_RETRIES + 1
    ):

        try:

            with driver.session() as session:

                session.execute_write(
                    lambda tx:
                    tx.run(
                        BATCH_QUERY,
                        rows=rows
                    ).consume()
                )

            return

        except (
            ServiceUnavailable,
            SessionExpired,
            TransientError
        ) as e:

            last_error = e

            if attempt == MAX_RETRIES:

                break

            print(
                f"\nNeo4j connection issue "
                f"(attempt {attempt}/{MAX_RETRIES}). "
                f"Retrying in {delay}s..."
            )

            time.sleep(delay)

            delay *= 2

    raise last_error


# =========================================================
# MAIN
# =========================================================

def main():

    print(
        "\n=========================================="
    )

    print(
        "Neo4j Legal Case Import"
    )

    print(
        "Qdrant case_id = SOURCE OF TRUTH"
    )

    print(
        "Citations = Citation nodes"
    )

    print(
        "==========================================\n"
    )

    driver = None

    try:

        # -------------------------------------------------
        # STEP 1
        # Build Qdrant mapping
        # -------------------------------------------------

        qdrant_mapping = (
            build_qdrant_case_id_mapping()
        )

        # -------------------------------------------------
        # STEP 2
        # Connect Neo4j
        # -------------------------------------------------

        driver = get_driver()

        # -------------------------------------------------
        # STEP 3
        # Constraints
        # -------------------------------------------------

        create_constraints(driver)

        # -------------------------------------------------
        # STEP 4
        # Repair checkpoint
        # -------------------------------------------------

        repaired_names = (
            load_repaired_filenames()
        )

        print(
            f"Repaired filenames in checkpoint: "
            f"{len(repaired_names)}"
        )

        # -------------------------------------------------
        # STEP 5
        # Find JSON files
        # -------------------------------------------------

        all_files = [

            f

            for f in Path(
                JSON_FOLDER
            ).rglob("*.json")

            if f.name in repaired_names

        ]

        found_names = {
            f.name
            for f in all_files
        }

        missing_on_disk = (
            repaired_names
            - found_names
        )

        if missing_on_disk:

            print(
                f"WARNING: "
                f"{len(missing_on_disk)} files "
                f"from checkpoint were not found."
            )

            log_error(
                f"Missing on disk "
                f"({len(missing_on_disk)}): "
                f"{sorted(missing_on_disk)}"
            )

        # -------------------------------------------------
        # STEP 6
        # Load own checkpoint
        # -------------------------------------------------

        completed = load_checkpoint()

        total_files = len(all_files)

        pending_files = [

            f

            for f in all_files

            if str(f) not in completed

        ]

        print(
            f"\nJSON files found : {total_files}"
        )

        print(
            f"Already imported: {len(completed)}"
        )

        print(
            f"Remaining       : {len(pending_files)}\n"
        )

        # -------------------------------------------------
        # Counters
        # -------------------------------------------------

        imported = 0
        skipped = 0
        failed = 0

        no_qdrant_match = 0

        unknown_court_count = 0

        total_chunks = 0

        chunk_count_mismatches = 0

        # -------------------------------------------------
        # Batch
        # -------------------------------------------------

        batch = []

        batch_paths = []

        pbar = tqdm(
            total=len(pending_files),
            desc="Importing",
            unit="file"
        )

        # -------------------------------------------------
        # PROCESS FILES
        # -------------------------------------------------

        for file_path in pending_files:

            if STOP_REQUESTED:

                break

            try:

                # -----------------------------------------
                # Read JSON
                # -----------------------------------------

                with open(
                    file_path,
                    "r",
                    encoding="utf-8"
                ) as f:

                    data = json.load(f)

                # -----------------------------------------
                # Resolve case_id from Qdrant
                # -----------------------------------------

                case_id = (
                    resolve_case_id_from_qdrant(
                        data,
                        qdrant_mapping
                    )
                )

                # -----------------------------------------
                # No match = SKIP
                # Never invent case_id
                # -----------------------------------------

                if not case_id:

                    no_qdrant_match += 1

                    skipped += 1

                    completed.add(
                        str(file_path)
                    )

                    log_error(
                        f"{file_path} | "
                        f"SKIPPED: no matching "
                        f"case_id in Qdrant"
                    )

                    pbar.update(1)

                    continue

                # -----------------------------------------
                # Build record
                # -----------------------------------------

                record = build_case_record(
                    data,
                    case_id
                )

                # -----------------------------------------
                # Court check
                # -----------------------------------------

                if (
                    record["court"]
                    == "Unknown"
                ):

                    unknown_court_count += 1

                # -----------------------------------------
                # Chunk check
                # -----------------------------------------

                actual_chunk_count = len(
                    record["chunks"]
                )

                total_chunks += (
                    actual_chunk_count
                )

                if (
                    record["num_chunks"]
                    != actual_chunk_count
                ):

                    chunk_count_mismatches += 1

                    log_error(
                        f"{file_path} | "
                        f"chunk count mismatch | "
                        f"JSON={record['num_chunks']} | "
                        f"actual={actual_chunk_count}"
                    )

                # -----------------------------------------
                # Add to batch
                # -----------------------------------------

                batch.append(record)

                batch_paths.append(
                    str(file_path)
                )

                # -----------------------------------------
                # Import batch
                # -----------------------------------------

                if len(batch) >= BATCH_SIZE:

                    import_batch(
                        driver,
                        batch
                    )

                    for p in batch_paths:

                        completed.add(p)

                    save_checkpoint(
                        completed
                    )

                    imported += len(
                        batch
                    )

                    batch.clear()

                    batch_paths.clear()

            except Exception as e:

                failed += 1

                log_error(
                    f"{file_path} | "
                    f"{type(e).__name__}: "
                    f"{e}"
                )

            pbar.set_postfix(
                {
                    "imported": imported,
                    "skipped": skipped,
                    "failed": failed
                }
            )

            pbar.update(1)

        # -------------------------------------------------
        # FINAL BATCH
        # -------------------------------------------------

        if batch and not STOP_REQUESTED:

            try:

                import_batch(
                    driver,
                    batch
                )

                for p in batch_paths:

                    completed.add(p)

                save_checkpoint(
                    completed
                )

                imported += len(
                    batch
                )

            except Exception as e:

                failed += len(batch)

                log_error(
                    f"FINAL BATCH | "
                    f"{type(e).__name__}: "
                    f"{e}"
                )

        # -------------------------------------------------
        # CLOSE PROGRESS BAR
        # -------------------------------------------------

        pbar.close()

        save_checkpoint(
            completed
        )

        # -------------------------------------------------
        # SUMMARY
        # -------------------------------------------------

        print(
            "\n=========================================="
        )

        print(
            "IMPORT SUMMARY"
        )

        print(
            "=========================================="
        )

        print(
            f"Imported : {imported}"
        )

        print(
            f"Skipped  : {skipped}"
        )

        print(
            f"  No Qdrant case_id match: "
            f"{no_qdrant_match}"
        )

        print(
            f"Failed   : {failed}"
        )

        print(
            f"Chunks   : {total_chunks}"
        )

        print(
            f"Completed: "
            f"{len(completed)} / {total_files}"
        )

        print(
            f"Court Unknown: "
            f"{unknown_court_count}"
        )

        print(
            f"Chunk mismatches: "
            f"{chunk_count_mismatches}"
        )

        print(
            "=========================================="
        )

        if no_qdrant_match > 0:

            print(
                "\nWARNING:"
            )

            print(
                f"{no_qdrant_match} files "
                f"were skipped because Qdrant "
                f"had no matching case_id."
            )

            print(
                f"Check: {LOG_FILE}"
            )

        if chunk_count_mismatches > 0:

            print(
                "\nWARNING:"
            )

            print(
                f"{chunk_count_mismatches} files "
                f"had chunk-count mismatches."
            )

            print(
                f"Check: {LOG_FILE}"
            )

        if STOP_REQUESTED:

            print(
                "\nImport stopped by user."
            )

            print(
                "Run the same command again "
                "to resume."
            )

        else:

            print(
                "\nNeo4j import completed successfully."
            )

    finally:

        if driver is not None:

            driver.close()


# =========================================================
# RUN
# =========================================================

if __name__ == "__main__":

    main()