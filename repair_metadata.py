"""

===================
repair_metadata.py chunks ko kabhi generate/modify nahi karta — ye sirf jo chunks pehle se extractor.py ne bana rakhe hain unko read karke jodta hai (reconstruct_full_text() function — chunk_index ke hisaab se sort karke sab chunks ka text ek saath jod deta hai)
In-place metadata repair for already-generated JSON files in extracted_text_clean/."""

"""
repair_metadata.py
===================
In-place metadata repair for already-generated JSON files in extracted_text_clean/.

WHAT THIS DOES:
  - Scans every *.json file in OUTPUT_DIR
  - Rebuilds full_text from that file's own "chunks" (already-extracted, already-cleaned
    text) -- NOT from the PDF. No PDF is opened. No OCR runs. No network calls.
  - Re-runs the SAME extract_document_metadata() logic from extractor.py on
    that reconstructed text.
  - Replaces a field ONLY if the CURRENT value is empty OR fails validation
    (e.g. "judge": "Ahmed/Pa" -- contains "/", or is a garbled fragment).
  - Any field that is already valid is left completely untouched.
  - "chunks", "case_number" (if valid), "generated_name", "actual_filename",
    "court", "case_type", "year", "used_ocr", "num_chunks" are NEVER modified
    unless a field itself is one of the ones being validated/fixed.
  - Writes the fix back to the SAME filename (json.dump overwrite). No new file,
    no backup file, nothing deleted.

WHAT THIS DOES NOT DO:
  - Does NOT delete any .json file.
  - Does NOT touch extractor_checkpoint.json.
  - Does NOT re-run OCR or re-open PDFs.
  - Does NOT force a value if nothing valid can be found -- it just leaves the
    field as-is and reports it at the end so you can review manually.

HOW TO RUN:
  1. Put this file in the SAME folder as extractor.py
     (D:\\hafsa_thesis material\\supreme_court_scraper)
  2. First do a dry run (default) to see what WOULD change, nothing is written:
         python repair_metadata.py
  3. If the preview looks right, apply for real:
         python repair_metadata.py --apply
"""

import os
import sys
import re
import json
import glob
import signal
import argparse

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

_INTERRUPTED = False

def _handle_sigint(signum, frame):
    global _INTERRUPTED
    if not _INTERRUPTED:
        print("\n⏸ Ctrl+C received -- current file finish hote hi ruk jayega "
              "(koi file adhoori nahi likhi jayegi)...", flush=True)
    _INTERRUPTED = True

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from new_extractor import extract_document_metadata, OUTPUT_DIR
except Exception as e:
    print(f"❌ Could not import new_extractor.py from this folder: {e}")
    print("   Make sure repair_metadata.py sits in the same directory as new_extractor.py")
    sys.exit(1)

signal.signal(signal.SIGINT, _handle_sigint)

# Scope used to be restricted to repair_checkpoint.json's original ~10.5k
# filenames (a one-time historical repair batch). neo4j_import.py no longer
# depends on that file at all (it now imports every JSON in OUTPUT_DIR).
# This script's own PROGRESS_FILE (below, unrelated to repair_checkpoint.json)
# tracks resume state on its own, so a full run is always safe. As a speed
# shortcut, repair_checkpoint.json's files are treated as an EXCLUDE list by
# default (opposite of the old read-only INCLUDE-only filter) -- they were
# already fixed by the original repair pass, so skipping them lets this
# script start directly on the files that were never touched. Pass
# --no-skip-repaired to disable this and re-check every file instead.
PROGRESS_FILE = "repair_metadata_progress.json"
REPAIR_CHECKPOINT_FILE = "repair_checkpoint.json"


def load_repair_checkpoint_filenames() -> set:
    """Filenames listed in repair_checkpoint.json -- used as an EXCLUDE
    list (already fixed by the original repair pass), not a scope filter."""
    if not os.path.exists(REPAIR_CHECKPOINT_FILE):
        return set()
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


def load_repair_checkpoint() -> set:
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, "r", encoding="utf-8") as f:
            return set(json.load(f))
    return set()


def save_repair_checkpoint(done_set: set):
    with open(PROGRESS_FILE, "w", encoding="utf-8") as f:
        json.dump(sorted(done_set), f, ensure_ascii=False, indent=2)


# ==========================
# FIELD VALIDATORS
# (define what "empty/garbled" means for each field)
# ==========================
BAD_JUDGE_TOKENS = {
    "PA", "PS", "APG", "DPG", "ASC", "ORDER", "DATE", "JUDGE", "PRESENT",
    "COURT", "SHEET", "HEARING", "COUNSEL"
}

def _coerce_scalar(val) -> str:
    if val is None:
        return ""
    if isinstance(val, str):
        return val.strip()
    if isinstance(val, list):
        for item in val:
            if isinstance(item, str) and item.strip():
                return item.strip()
        return ""
    return str(val).strip()


def is_valid_judge(name) -> bool:
    name = _coerce_scalar(name)
    if not name:
        return False
    if len(name) < 4:
        return False
    if any(ch.isdigit() for ch in name):
        return False
    if "/" in name or "\\" in name:
        return False
    words = [w for w in re.split(r"\s+", name) if w]
    if not words:
        return False
    for w in words:
        letters_only = re.sub(r"[^A-Za-z]", "", w)
        if len(letters_only) < 2:
            return False
        if letters_only.upper() in BAD_JUDGE_TOKENS:
            return False
    return True


# Same prefix list used by extract_document_metadata()'s case_number regex.
# A real case number must START with one of these AND contain "No." followed
# by an alphanumeric identifier. This replaces the old length-only check
# (len >= 5), which let garbage sentence fragments like "Constitution is not"
# or "W.P. Nos" pass through as "valid" and never get repaired.
CASE_NUMBER_PREFIX_RE = re.compile(
    r"^(?:Crl\.?|CRL\.?|Cr\.B\.A\.?|Cr\.R\.A\.?|Cr\.A\.?|Civil|W\.P\.?|Const\.?|Criminal|Misc\.?)",
    re.I
)
# Requires an actual DIGIT to appear after "No." (with at most one
# leading letter + separator in between, e.g. "No.S-1234" or "No.1234"),
# not just any word character. Fixes a real gap: source PDF text sometimes
# has a corrupted/mangled hyphen character right after the prefix letter
# (e.g. "Cr. Misc. App. No. S <corrupted-char> 507 of 2023"), which stops
# the extraction regex right at "No. S" -- a bare trailing letter with no
# digit anywhere is always a truncated fragment, never a real case number.
CASE_NUMBER_HAS_NO_RE = re.compile(r"No\.?\s*[A-Za-z]?[\s\-]?\d", re.I)


def is_valid_case_number(val) -> bool:
    val = _coerce_scalar(val)
    if not val:
        return False
    if len(val) < 5 or len(val) > 60:
        return False
    if not CASE_NUMBER_PREFIX_RE.match(val):
        return False
    if not CASE_NUMBER_HAS_NO_RE.search(val):
        return False
    return True


def is_valid_date(val) -> bool:
    val = _coerce_scalar(val)
    if not val:
        return False
    if re.match(r"^\d{1,2}[.\-/]\d{1,2}[.\-/]\d{2,4}$", val):
        return True
    if re.match(r"^\d{1,2}(?:st|nd|rd|th)?\s+\w+,?\s+\d{4}$", val, re.I):
        return True
    return False


def is_valid_list(val) -> bool:
    return isinstance(val, list) and len(val) > 0


FIELD_VALIDATORS = {
    "judge":          is_valid_judge,
    "case_number":    is_valid_case_number,
    "date_of_order":  is_valid_date,
    "sections_cited": is_valid_list,
    "citations":      is_valid_list,
    "parties":        is_valid_list,
}


def reconstruct_full_text(data: dict) -> str:
    chunks = data.get("chunks", [])
    chunks_sorted = sorted(chunks, key=lambda c: c.get("chunk_index", 0))
    return " ".join(c.get("text", "") for c in chunks_sorted)


MAX_TEXT_CHARS = 200_000


def _invalid_fields(data: dict) -> list:
    """Cheap pre-check (no PDF/Ollama) -- which FIELD_VALIDATORS fields on
    this file's CURRENT stored data are already invalid. Lets repair_file()
    skip the expensive extract_document_metadata() call (which can trigger
    an Ollama call whenever ANY of judge/case_number/date_of_order/
    sections_cited/parties is missing, regardless of whether the fields
    this script actually cares about are already fine) for files that
    don't need any repair at all -- across the full 58k-file corpus this
    is the difference between a multi-day run and a fast one, since most
    files already have valid data from earlier repair passes."""
    invalid = []
    for field, validator in FIELD_VALIDATORS.items():
        current_val = data.get(field, "" if field not in ("sections_cited", "citations", "parties") else [])
        if not validator(current_val):
            invalid.append(field)
    return invalid


def repair_file(path: str, apply_changes: bool):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not _invalid_fields(data):
        return {"file": os.path.basename(path), "status": "no_change_needed",
                "changes": {}, "still_invalid": []}

    full_text = reconstruct_full_text(data)
    if not full_text.strip():
        return {"file": os.path.basename(path), "status": "skipped_no_chunk_text",
                "changes": {}, "still_invalid": []}

    if len(full_text) > MAX_TEXT_CHARS:
        full_text = full_text[:MAX_TEXT_CHARS]

    row = {
        "court":     data.get("court", ""),
        "case_type": data.get("case_type", ""),
        "year":      data.get("year", ""),
    }
    fresh_meta = extract_document_metadata(full_text, row)

    changes = {}
    still_invalid = []

    for field, validator in FIELD_VALIDATORS.items():
        empty_val = "" if field not in ("sections_cited", "citations", "parties") else []
        current_val = data.get(field, empty_val)
        if validator(current_val):
            continue

        new_val = fresh_meta.get(field)
        if validator(new_val):
            changes[field] = {"old": current_val, "new": new_val}
            if apply_changes:
                data[field] = new_val
        else:
            still_invalid.append(field)
            # current_val is CONFIRMED garbage (that's why we're here -- it
            # failed validation above) and nothing better could be found.
            # Keeping known-garbage (e.g. "JUDGE Rafiq/PA") is worse than an
            # honest empty field -- downstream consumers (Neo4j/Qdrant/the UI)
            # can't tell confirmed-garbage from real data, whereas an empty
            # field unambiguously means "not found." Only clear it if there
            # was actually something there to begin with (skip the no-op of
            # "clearing" an already-empty field).
            if current_val != empty_val:
                changes[field] = {"old": current_val, "new": empty_val, "cleared": True}
                if apply_changes:
                    data[field] = empty_val

    if changes and apply_changes:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    return {
        "file": os.path.basename(path),
        "status": "changed" if changes else "no_change_needed",
        "changes": changes,
        "still_invalid": still_invalid,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true",
                         help="Actually write fixes. Without this flag, runs as a dry-run preview only.")
    parser.add_argument("--reset", action="store_true",
                         help="Clear repair progress checkpoint and rescan ALL files from the start.")
    parser.add_argument("--no-skip-repaired", action="store_true",
                         help="Also re-check files listed in repair_checkpoint.json "
                              "(skipped by default since they were already fixed by "
                              "the original repair pass).")
    args = parser.parse_args()

    if args.reset and os.path.exists(PROGRESS_FILE):
        os.remove(PROGRESS_FILE)
        print(f"🔄 Cleared {PROGRESS_FILE} -- will rescan all in-scope files from scratch.\n")

    all_json_files = sorted(glob.glob(os.path.join(OUTPUT_DIR, "*.json")))
    total_on_disk = len(all_json_files)

    if not args.no_skip_repaired:
        already_repaired = load_repair_checkpoint_filenames()
        if already_repaired:
            all_json_files = [p for p in all_json_files if os.path.basename(p) not in already_repaired]
            print(f"⏭  Skipping {total_on_disk - len(all_json_files)} file(s) already listed in "
                  f"{REPAIR_CHECKPOINT_FILE} (already fixed by the original repair pass).")

    total = len(all_json_files)
    print(f"📂 {total} JSON files in scope (of {total_on_disk} total in '{OUTPUT_DIR}/').")

    done = load_repair_checkpoint() if args.apply else set()

    pending_files = [p for p in all_json_files if os.path.basename(p) not in done]
    already_done_count = total - len(pending_files)

    if args.apply and already_done_count:
        print(f"⏭  Resuming: {already_done_count} already scanned in a previous run, "
              f"{len(pending_files)} remaining.")
    print(f"🔧 Mode: {'APPLY (writing changes)' if args.apply else 'DRY RUN (preview only, nothing written)'}\n")

    total_changed = 0
    total_still_invalid = []
    processed_this_run = 0

    for offset, path in enumerate(pending_files, start=1):
        idx = already_done_count + offset
        print(f"[{idx}/{total}] ⏳ processing {os.path.basename(path)} ...", flush=True)
        result = repair_file(path, apply_changes=args.apply)
        processed_this_run = offset

        if result["status"] == "skipped_no_chunk_text":
            print(f"[{idx}/{total}] ⚠  {os.path.basename(path)} -- no chunk text found, skipped", flush=True)
        elif result["changes"]:
            total_changed += 1
            print(f"[{idx}/{total}] ✅ {result['file']}", flush=True)
            for field, cv in result["changes"].items():
                tag = " (cleared -- confirmed garbage, nothing better found)" if cv.get("cleared") else ""
                print(f"     {field}: {cv['old']!r} -> {cv['new']!r}{tag}", flush=True)
        else:
            print(f"[{idx}/{total}] ✔ {os.path.basename(path)} -- already valid, untouched", flush=True)

        if result["still_invalid"]:
            total_still_invalid.append((result["file"], result["still_invalid"]))

        if args.apply:
            done.add(os.path.basename(path))
            save_repair_checkpoint(done)

        remaining = total - idx
        print(f"    📊 Progress: {idx}/{total} done | {remaining} remaining\n", flush=True)

        if _INTERRUPTED:
            print(f"⏸ Ctrl+C confirmed -- rok raha hoon ({idx}/{total} scanned, "
                  f"{remaining} remaining). Koi file delete/adhoori nahi hai -- "
                  f"script dobara chalao to checkpoint se yahi se resume hoga "
                  f"(file 1 se dobara shuru NAHI hoga).", flush=True)
            break

    print(f"\n🎉 This run scanned {processed_this_run} file(s). "
          f"Total progress: {min(already_done_count + processed_this_run, total)}/{total}. "
          f"Files fixed this run: {total_changed}")

    if total_still_invalid:
        print(f"\n⚠  {len(total_still_invalid)} file(s) still have fields that could not be "
              f"recovered from stored text (left untouched -- review manually):")
        for fname, fields in total_still_invalid:
            print(f"   - {fname}: {', '.join(fields)}")

    if not args.apply:
        print("\n👉 This was a DRY RUN. Nothing was written to disk.")
        print("   Run again with:  python repair_metadata.py --apply")
    elif already_done_count + processed_this_run >= total:
        print(f"\n✅ All {total} files scanned. Repair complete.")
        print(f"   (If you ever want to rescan everything again: python repair_metadata.py --apply --reset)")


if __name__ == "__main__":
    main()