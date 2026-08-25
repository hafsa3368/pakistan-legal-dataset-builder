"""
inspect_clean_header_unresolved.py
====================================
Inspects the "clean header but no case number found" bucket produced by
finalize_unresolved_case_numbers.py (unresolved_clean_header.json). This is
the highest-value bucket to eyeball: the header text reads fine (not OCR
garbage), but CASE_NO_RE still didn't find anything -- meaning either:

  (a) the real case number IS there, in a format the regex still doesn't
      cover (-> worth expanding the regex further), or
  (b) the header genuinely has no case-number caption on this particular
      page (e.g. a continuation page, an unusual order format, etc. ->
      genuinely nothing to extract, leave it empty).

Does NOT call Ollama, does NOT write anything -- purely for reading.

USAGE:
    python inspect_review_headers.py                 # sample 20
    python inspect_review_headers.py --sample 40
    python inspect_review_headers.py --filter "Cr."   # only rows whose
                                                                 # old_case_number
                                                                 # contains this
"""

import os
import json
import argparse

OUTPUT_DIR = "extracted_text_clean"
INPUT_FILE = "unresolved_clean_header.json"
HEADER_WINDOW = 3000
PREVIEW_CHARS = 700


def reconstruct_full_text(data: dict) -> str:
    chunks = sorted(data.get("chunks", []), key=lambda c: c.get("chunk_index", 0))
    return " ".join(c.get("text", "") for c in chunks)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=int, default=20)
    parser.add_argument("--filter", type=str, default=None,
                         help="Only show rows whose old_case_number contains this substring "
                              "(case-insensitive).")
    args = parser.parse_args()

    if not os.path.exists(INPUT_FILE):
        print(f"❌ {INPUT_FILE} not found. Run finalize_unresolved_case_numbers.py first.")
        return

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        rows = json.load(f)

    if args.filter:
        needle = args.filter.lower()
        rows = [r for r in rows if needle in str(r.get("old_case_number", "")).lower()]

    print(f"📋 {len(rows)} rows in {INPUT_FILE} "
          f"({'filtered by ' + repr(args.filter) if args.filter else 'unfiltered'}).")

    shown = 0
    for row in rows:
        if shown >= args.sample:
            break
        fname = row["file"]
        path = os.path.join(OUTPUT_DIR, fname)
        if not os.path.exists(path):
            continue
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            print(f"\n--- {fname} --- (could not read: {e})")
            continue

        full_text = reconstruct_full_text(data)
        header_text = full_text[:HEADER_WINDOW]

        shown += 1
        print(f"\n--- {fname} ---")
        print(f"  old_case_number (now nulled in the file): {row.get('old_case_number')!r}")
        preview = header_text[:PREVIEW_CHARS].replace("\n", " ⏎ ")
        print(f"  header preview: {preview!r}")

    if shown == 0:
        print("  Nothing matched -- try without --filter.")
    else:
        print(f"\n  Shown {shown} file(s). For each: is there a real case-number caption "
              f"CASE_NO_RE is still missing? If a pattern repeats across several, that's "
              f"worth adding as a new prefix.")


if __name__ == "__main__":
    main()