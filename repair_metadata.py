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

# --------------------------------------------------------------------------
# Safe Ctrl+C handling: sets a flag instead of killing mid-write.
# The loop checks this flag AFTER each file finishes (read + possible
# write is done in one quick step), so no file is ever left half-written.
#
# IMPORTANT: extractor.py ALSO registers its own signal.signal(SIGINT, ...)
# handler at import time. If we register ours before importing extractor,
# extractor's import silently overwrites ours and Ctrl+C stops responding
# to THIS script (it would print extractor.py's own message instead and
# set extractor's internal flag, which this script never checks). So we
# define the handler now, import extractor.py below, and only THEN
# register our handler -- as the LAST thing to touch signal.signal, so it
# wins.
# --------------------------------------------------------------------------
_INTERRUPTED = False

def _handle_sigint(signum, frame):
    global _INTERRUPTED
    if not _INTERRUPTED:
        print("\n⏸ Ctrl+C received -- current file finish hote hi ruk jayega "
              "(koi file adhoori nahi likhi jayegi)...", flush=True)
    _INTERRUPTED = True

# --------------------------------------------------------------------------
# Import the exact same extraction logic already used by extractor.py,
# so the repair uses IDENTICAL rules -- no logic duplication/drift.
# --------------------------------------------------------------------------
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from extractor import extract_document_metadata, OUTPUT_DIR
except Exception as e:
    print(f"❌ Could not import extractor.py from this folder: {e}")
    print("   Make sure repair_metadata.py sits in the same directory as extractor.py")
    sys.exit(1)

# Register AFTER the import so extractor.py's own signal.signal() call
# (which runs during import) cannot override this script's handler.
signal.signal(signal.SIGINT, _handle_sigint)


# --------------------------------------------------------------------------
# Checkpoint: tracks which files this repair script has ALREADY scanned,
# so Ctrl+C / a fresh run resumes from where it left off instead of
# starting over at file 1 every time. This checkpoint only tracks repair
# progress -- it never touches extractor_checkpoint.json (the main
# pipeline's own checkpoint) and never deletes any .json output file.
# --------------------------------------------------------------------------
REPAIR_CHECKPOINT_FILE = "repair_checkpoint.json"


def load_repair_checkpoint() -> set:
    if os.path.exists(REPAIR_CHECKPOINT_FILE):
        with open(REPAIR_CHECKPOINT_FILE, "r", encoding="utf-8") as f:
            return set(json.load(f))
    return set()


def save_repair_checkpoint(done_set: set):
    with open(REPAIR_CHECKPOINT_FILE, "w", encoding="utf-8") as f:
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
    """
    Safely turn ANY value (str, list, None, int, etc.) into a plain string
    for validation/comparison, so a type mismatch between what's stored in
    the JSON and what extractor.py's extract_document_metadata() currently
    returns can never crash the script.
    """
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


def is_valid_case_number(val) -> bool:
    val = _coerce_scalar(val)
    if not val:
        return False
    return len(val) >= 5


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


# Safety cap: judge/case_number/date/sections/citations/parties always appear
# early in a Pakistani court judgment. Capping avoids rare pathological/slow
# regex behavior on unusually large reconstructed documents, without losing
# any real extraction accuracy.
MAX_TEXT_CHARS = 200_000


def repair_file(path: str, apply_changes: bool):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

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
        current_val = data.get(field, "" if field not in ("sections_cited", "citations", "parties") else [])
        if validator(current_val):
            continue  # already valid -- do not touch

        new_val = fresh_meta.get(field)
        if validator(new_val):
            changes[field] = {"old": current_val, "new": new_val}
            if apply_changes:
                data[field] = new_val
        else:
            still_invalid.append(field)

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
    args = parser.parse_args()

    if args.reset and os.path.exists(REPAIR_CHECKPOINT_FILE):
        os.remove(REPAIR_CHECKPOINT_FILE)
        print(f"🔄 Cleared {REPAIR_CHECKPOINT_FILE} -- will rescan all files from scratch.\n")

    all_json_files = sorted(glob.glob(os.path.join(OUTPUT_DIR, "*.json")))
    total = len(all_json_files)

    # Checkpoint only matters in --apply mode (that's the long-running pass
    # you actually need to resume). Dry runs always preview everything.
    done = load_repair_checkpoint() if args.apply else set()

    pending_files = [p for p in all_json_files if os.path.basename(p) not in done]
    already_done_count = total - len(pending_files)

    print(f"📂 Found {total} JSON files in '{OUTPUT_DIR}/'")
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
                print(f"     {field}: {cv['old']!r} -> {cv['new']!r}", flush=True)
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
    