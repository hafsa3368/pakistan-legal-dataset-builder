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
# --------------------------------------------------------------------------
_INTERRUPTED = False

def _handle_sigint(signum, frame):
    global _INTERRUPTED
    if not _INTERRUPTED:
        print("\n⏸ Ctrl+C received -- current file finish hote hi ruk jayega "
              "(koi file adhoori nahi likhi jayegi)...", flush=True)
    _INTERRUPTED = True

signal.signal(signal.SIGINT, _handle_sigint)

# --------------------------------------------------------------------------
# Import the exact same extraction logic already used by legal_extractor.py,
# so the repair uses IDENTICAL rules -- no logic duplication/drift.
# --------------------------------------------------------------------------
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
   from extractor import extract_document_metadata, OUTPUT_DIR
except Exception as e:
    print(f"❌ Could not import extractor.py from this folder: {e}")
    print("   Make sure repair_metadata.py sits in the same directory as extractor.py")


# ==========================
# FIELD VALIDATORS
# (define what "empty/garbled" means for each field)
# ==========================
BAD_JUDGE_TOKENS = {
    "PA", "PS", "APG", "DPG", "ASC", "ORDER", "DATE", "JUDGE", "PRESENT",
    "COURT", "SHEET", "HEARING", "COUNSEL"
}

def is_valid_judge(name: str) -> bool:
    if not name or not name.strip():
        return False
    name = name.strip()
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


def is_valid_case_number(val: str) -> bool:
    if not val or not val.strip():
        return False
    return len(val.strip()) >= 5


def is_valid_date(val: str) -> bool:
    if not val or not val.strip():
        return False
    val = val.strip()
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


def repair_file(path: str, apply_changes: bool):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    full_text = reconstruct_full_text(data)
    if not full_text.strip():
        return {"file": path, "status": "skipped_no_chunk_text", "changes": {}}

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
    args = parser.parse_args()

    json_files = sorted(glob.glob(os.path.join(OUTPUT_DIR, "*.json")))
    print(f"📂 Found {len(json_files)} JSON files in '{OUTPUT_DIR}/'")
    print(f"🔧 Mode: {'APPLY (writing changes)' if args.apply else 'DRY RUN (preview only, nothing written)'}\n")

    total = len(json_files)
    total_changed = 0
    total_still_invalid = []
    processed = 0

    for idx, path in enumerate(json_files, start=1):
        result = repair_file(path, apply_changes=args.apply)
        processed = idx

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

        remaining = total - idx
        print(f"    📊 Progress: {idx}/{total} done | {remaining} remaining\n", flush=True)

        if _INTERRUPTED:
            print(f"⏸ Ctrl+C confirmed -- rok raha hoon ({idx}/{total} scanned, "
                  f"{remaining} remaining). Koi file delete/adhoori nahi hai -- "
                  f"script dobara chalao to yehi loop bache hue files se shuru ho jayega "
                  f"(already-fixed files ko dobara touch nahi karega, kyunki wo ab valid hain).", flush=True)
            break

    print(f"\n🎉 Stopped after scanning {processed}/{total} files. Files fixed: {total_changed}/{processed}")

    if total_still_invalid:
        print(f"\n⚠  {len(total_still_invalid)} file(s) still have fields that could not be "
              f"recovered from stored text (left untouched -- review manually):")
        for fname, fields in total_still_invalid:
            print(f"   - {fname}: {', '.join(fields)}")

    if not args.apply:
        print("\n👉 This was a DRY RUN. Nothing was written to disk.")
        print("   Run again with:  python repair_metadata.py --apply")


if __name__ == "__main__":
    main()