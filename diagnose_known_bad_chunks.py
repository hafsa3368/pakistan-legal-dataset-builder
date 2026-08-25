"""
diagnose_known_bad_chunks.py
==============================
Diagnostic (READ-ONLY, no writes) for the 1554 files that
fix_case_numbers.py flagged as "unresolved known-bad" -- i.e. files where
the header-regex kept RE-DERIVING the exact same "Cr.B.A.No.S-994 of 2019"
string from the file's own reconstructed chunk text, run after run.

WHY THIS SCRIPT EXISTS
-----------------------
If the regex genuinely re-extracts the same bad case number from a file's
OWN header text every time, that means the bad string isn't just sitting
in a metadata field -- it's present in the actual `chunks` array that
reconstruct_full_text() stitches together. That's a much bigger problem
than a metadata regex bug: it means these 1554 files may be storing the
WRONG document's text entirely (e.g. all pointing at the same source PDF,
or a duplicate/template page), not just a wrong case_number label on
otherwise-correct content.

No regex "pattern" can fix that -- if the underlying stored text for a
file genuinely is a different case's content, the only real fixes are:
  (a) re-scrape / re-extract that file from its correct source PDF, or
  (b) if it turns out these are genuine duplicate downloads (same PDF
      saved under many filenames), deduplicate and drop/relink them.

This script does NOT change anything. It only reports, for each flagged
file:
  - the first ~300 chars of its reconstructed header text (so you can see
    with your own eyes whether it's genuinely S-994's content)
  - a hash of the full reconstructed text, so you can see how many of the
    1554 files share IDENTICAL content vs merely similar/overlapping
    content
  - the file's recorded source PDF filename/path, if present in the JSON,
    so you can check whether it's really the same source file reused

USAGE:
    python diagnose_known_bad_chunks.py
    python diagnose_known_bad_chunks.py --sample 20     # print full detail for first 20 only
"""

import os
import re
import json
import glob
import hashlib
import argparse
from collections import defaultdict, Counter

OUTPUT_DIR   = "extracted_text_clean"
REVIEW_FILE  = "fix_case_numbers_needs_review.json"

HEADER_WINDOW = 3000
PREVIEW_CHARS = 300


def reconstruct_full_text(data: dict) -> str:
    chunks = sorted(data.get("chunks", []), key=lambda c: c.get("chunk_index", 0))
    return " ".join(c.get("text", "") for c in chunks)


def load_flagged_files(reason_filter):
    if not os.path.exists(REVIEW_FILE):
        raise FileNotFoundError(f"{REVIEW_FILE} not found. Run fix_case_numbers.py (dry run) first.")
    with open(REVIEW_FILE, "r", encoding="utf-8") as f:
        entries = json.load(f)
    return [e for e in entries if e.get("reason") == reason_filter]


def guess_source_pdf_field(data: dict):
    """Best-effort: look for whatever field this pipeline used to record
    the original source PDF path/filename, since the exact key name isn't
    guaranteed across pipeline versions."""
    for key in ("source_pdf", "source_file", "pdf_path", "original_filename",
                "source", "file_path", "pdf_filename"):
        if key in data and data[key]:
            return key, data[key]
    return None, None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=int, default=10,
                         help="How many files to print full header-text preview for (default 10).")
    parser.add_argument("--reason", default="regex_no_match_known_bad_value_retained",
                         choices=["regex_no_match_known_bad_value_retained",
                                  "regex_reextracted_identical_known_bad_value"],
                         help="Which flagged bucket to inspect. Default is the "
                              "'no match, header doesn't contain the pattern' bucket -- "
                              "these are the real hallucination-victim candidates. "
                              "Use 'regex_reextracted_identical_known_bad_value' to check "
                              "files whose header genuinely DOES contain the pattern "
                              "(possible true positives / real S-994 duplicates).")
    args = parser.parse_args()

    flagged = load_flagged_files(args.reason)
    print(f"🔎 {len(flagged)} flagged file(s) to inspect (reason = {args.reason})\n")
    report_file = f"diagnose_known_bad_chunks_report__{args.reason}.json"

    text_hash_to_files = defaultdict(list)
    source_pdf_counter = Counter()
    missing_on_disk = []
    report_rows = []

    for idx, entry in enumerate(flagged, 1):
        fname = entry["file"]
        path = os.path.join(OUTPUT_DIR, fname)

        if not os.path.exists(path):
            missing_on_disk.append(fname)
            continue

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        full_text = reconstruct_full_text(data)
        header_text = full_text[:HEADER_WINDOW]
        text_hash = hashlib.sha256(full_text.encode("utf-8")).hexdigest()
        text_hash_to_files[text_hash].append(fname)

        src_key, src_val = guess_source_pdf_field(data)
        if src_val:
            source_pdf_counter[str(src_val)] += 1

        row = {
            "file": fname,
            "case_id": data.get("case_id"),
            "stored_case_number": data.get("case_number"),
            "source_field_used": src_key,
            "source_value": src_val,
            "full_text_sha256": text_hash,
            "header_preview": header_text[:PREVIEW_CHARS],
        }
        report_rows.append(row)

        if idx <= args.sample:
            print(f"── [{idx}] {fname} ─────────────────────────────")
            print(f"    case_id           : {data.get('case_id')}")
            print(f"    stored case_number: {data.get('case_number')!r}")
            print(f"    source field/value: {src_key} = {src_val!r}")
            print(f"    header preview    : {header_text[:PREVIEW_CHARS]!r}")
            print(f"    full_text sha256  : {text_hash}\n")

    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(report_rows, f, ensure_ascii=False, indent=2)

    # ── Summary ──────────────────────────────────────────────────────
    print("\n================ SUMMARY ================")
    print(f"Total flagged files inspected : {len(report_rows)}")
    if missing_on_disk:
        print(f"⚠ Missing on disk             : {len(missing_on_disk)} (not counted below)")

    unique_texts = len(text_hash_to_files)
    print(f"Unique full-text content hashes among these files: {unique_texts}")

    if unique_texts == 1:
        print("🛑 ALL flagged files share IDENTICAL stored text content.")
        print("   This confirms it's not a metadata bug -- these files are literally")
        print("   storing the same document's chunks under different case_ids/filenames.")
    elif unique_texts < len(report_rows) * 0.1:
        print("🛑 A SMALL number of distinct texts cover almost all flagged files --")
        print("   strong sign of a small set of source PDFs being reused/duplicated")
        print("   across many case_ids, not a per-file extraction bug.")
    else:
        print("ℹ Content hashes are mostly distinct -- these files do NOT share identical")
        print("  text. This would mean each file's OWN header genuinely contains an")
        print("  S-994-like pattern independently (less likely, but check the header")
        print("  previews above/in the report to confirm what's actually there).")

    # Show the largest clusters of duplicate content, if any.
    clusters = sorted(text_hash_to_files.items(), key=lambda kv: -len(kv[1]))
    top_clusters = [c for c in clusters if len(c[1]) > 1][:5]
    if top_clusters:
        print(f"\nTop duplicate-content clusters (same full_text shared across files):")
        for h, files in top_clusters:
            print(f"  {h[:12]}...  -> {len(files)} files, e.g. {files[:3]}")

    if source_pdf_counter:
        print(f"\nMost common recorded source PDF value(s) among flagged files:")
        for src, count in source_pdf_counter.most_common(5):
            print(f"  {count:>5}  {src}")
    else:
        print("\nℹ No recognizable source-PDF field found in these JSON files "
              "(checked: source_pdf, source_file, pdf_path, original_filename, "
              "source, file_path, pdf_filename). If your pipeline uses a different "
              "field name, tell me and I'll adjust guess_source_pdf_field().")

    print(f"\n📄 Full per-file detail written to {report_file}")


if __name__ == "__main__":
    main()