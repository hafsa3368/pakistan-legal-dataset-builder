"""
finalize_unresolved_case_numbers.py
=====================================
Run this AFTER fix_case_numbers.py --apply. It reads
fix_case_numbers_needs_review.json (the leftover unresolved files) and does
THREE things, all DRY RUN by default:

1. NULL OUT confirmed-contaminated case_number values that have no safe
   replacement. Leaving the old wrong ghost value (e.g. repeated
   'Cr.B.A.No.S-994 of 2019') in place is worse than an empty string.

2. SPLITS the leftovers into THREE buckets:
     - unresolved_no_content_extracted.json  -- reconstructed text is empty,
       or is ONLY the "CamScanner" watermark (with nothing else). This is a
       total EXTRACTION failure, not a case-number-format problem: there is
       no document content to search at all. The source PDF was almost
       certainly a scanned image with no usable text layer -- these need
       re-OCR of the original PDF, not a regex fix. (Revision 2: this
       bucket was previously getting lumped into "clean header" by mistake,
       since "CamScanner" alone doesn't trip the OCR-garble heuristic.)
     - unresolved_ocr_garbled.json    -- there IS real document text, but it
       reads as OCR-corrupted (mixed-script noise, broken words). Also
       needs re-OCR, but at least some content was extracted, unlike
       bucket 1.
     - unresolved_clean_header.json   -- header text reads fine and has
       real document content, but truly has no case-number caption found.
       This is the highest-value bucket for a second look at the regex.

USAGE:
    python finalize_unresolved_case_numbers.py                 # dry run, just reports the split
    python finalize_unresolved_case_numbers.py --apply          # actually null out case_number
                                                                  # for confirmed-unresolved files
"""

import os
import re
import json
import argparse

OUTPUT_DIR = "extracted_text_clean"
REVIEW_FILE = "fix_case_numbers_needs_review.json"
NO_CONTENT_FILE = "unresolved_no_content_extracted.json"
OCR_GARBLED_FILE = "unresolved_ocr_garbled.json"
CLEAN_HEADER_FILE = "unresolved_clean_header.json"
HEADER_WINDOW = 3000

# Heuristics for "this header text itself looks OCR-garbled".
_GARBLE_PATTERNS = re.compile(
    r"("
    r"[A-Za-z]\{[A-Za-z]|"           # stray brace mid-word, e.g. "l{r:r'"
    r"[A-Za-z]'[a-z]{1,3}\b|"        # stray apostrophe mid-word
    r"\b[A-Z]{2,}[a-z]{1,2}[A-Z]{2,}\b|"  # weird case-swapping like "FIIGH"
    r"[\u0600-\u06FF]{1,3}[A-Za-z]|[A-Za-z][\u0600-\u06FF]{1,3}"  # Arabic/Urdu
                                                                    # chars glued
                                                                    # to Latin
                                                                    # letters
    r")"
)

# REVISION 2: detects "extraction produced essentially nothing except the
# CamScanner watermark" -- a distinct failure mode from OCR-garbled-but-
# present text. After stripping every occurrence of the word "camscanner"
# (case-insensitive) and surrounding whitespace/punctuation, if what's left
# is very short, there was no real document content to work with at all.
_CAMSCANNER_RE = re.compile(r"camscanner", re.I)
_NON_ALNUM_RE = re.compile(r"[^0-9A-Za-z]+")


def reconstruct_full_text(data: dict) -> str:
    chunks = sorted(data.get("chunks", []), key=lambda c: c.get("chunk_index", 0))
    return " ".join(c.get("text", "") for c in chunks)


def has_no_real_content(text: str, min_remaining_chars: int = 30) -> bool:
    """True if, after stripping the CamScanner watermark and all
    non-alphanumeric noise, fewer than `min_remaining_chars` characters
    remain -- meaning there's essentially no real document content here."""
    if not text:
        return True
    stripped = _CAMSCANNER_RE.sub("", text)
    stripped = _NON_ALNUM_RE.sub("", stripped)
    return len(stripped) < min_remaining_chars


def looks_ocr_garbled(text: str) -> bool:
    """Heuristic only -- flags text with an unusually high density of the
    garble patterns above. Used purely to sort the review queue."""
    if not text:
        return False
    hits = len(_GARBLE_PATTERNS.findall(text))
    density = hits / max(len(text), 1)
    return hits >= 3 and density > 0.002


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true",
                         help="Actually null out case_number for confirmed-unresolved "
                              "files. Without this, dry-run preview only.")
    args = parser.parse_args()

    if not os.path.exists(REVIEW_FILE):
        print(f"❌ {REVIEW_FILE} not found. Run fix_case_numbers.py first.")
        return

    with open(REVIEW_FILE, "r", encoding="utf-8") as f:
        rows = json.load(f)

    no_replacement_rows = [
        r for r in rows
        if r.get("reason") == "contaminated_stored_value_not_found_in_own_header_no_replacement"
    ]
    other_rows = [r for r in rows if r not in no_replacement_rows]

    print(f"📋 {len(rows)} total unresolved rows.")
    print(f"   {len(no_replacement_rows)} confirmed-contaminated with no replacement "
          f"(candidates for nulling).")
    print(f"   {len(other_rows)} other (low-confidence candidate -- left untouched, "
          f"still in {REVIEW_FILE}).\n")

    no_content = []
    garbled = []
    clean_header = []
    nulled = 0

    for row in no_replacement_rows:
        fname = row["file"]
        path = os.path.join(OUTPUT_DIR, fname)
        if not os.path.exists(path):
            continue
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            print(f"⚠ Could not read {fname}: {e}")
            continue

        full_text = reconstruct_full_text(data)
        header_text = full_text[:HEADER_WINDOW]

        if has_no_real_content(header_text):
            no_content.append(row)
        elif looks_ocr_garbled(header_text):
            garbled.append(row)
        else:
            clean_header.append(row)

        if args.apply:
            data["case_number"] = ""
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            nulled += 1

    with open(NO_CONTENT_FILE, "w", encoding="utf-8") as f:
        json.dump(no_content, f, ensure_ascii=False, indent=2)
    with open(OCR_GARBLED_FILE, "w", encoding="utf-8") as f:
        json.dump(garbled, f, ensure_ascii=False, indent=2)
    with open(CLEAN_HEADER_FILE, "w", encoding="utf-8") as f:
        json.dump(clean_header, f, ensure_ascii=False, indent=2)

    print(f"⚪ {len(no_content)} file(s) have essentially NO extracted content "
          f"(just the CamScanner watermark, nothing else) -- see {NO_CONTENT_FILE}. "
          f"The original scan needs proper OCR from scratch; there is no text here "
          f"for any regex or LLM to work with.")
    print(f"🔴 {len(garbled)} file(s) HAVE real text but it looks OCR-garbled -- see "
          f"{OCR_GARBLED_FILE}. Needs the source PDF re-OCR'd with a better engine.")
    print(f"🟡 {len(clean_header)} file(s) have a clean-looking header with genuinely "
          f"no case-number caption found -- see {CLEAN_HEADER_FILE}. Worth a quick "
          f"manual glance; some may reveal one more missing regex format.")

    if args.apply:
        print(f"\n✅ case_number nulled (set to \"\") for {nulled} confirmed-contaminated "
              f"file(s) -- they no longer carry a wrong ghost value.")
    else:
        print("\n👉 This was a DRY RUN. Re-run with --apply to null out case_number "
              "for the confirmed-contaminated files listed above.")


if __name__ == "__main__":
    main()