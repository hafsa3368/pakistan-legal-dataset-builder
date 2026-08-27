"""
reprocess_unresolved_ocr.py
=============================
Re-runs the OCR extraction pipeline (now at 300 DPI -- see new_extractor.py's
OCR_DPI/OCR_ZOOM) on the ~1900 files flagged by finalize_unresolved_case_numbers.py
as "no_content_extracted" or "ocr_garbled". Those were extracted under the OLD
~72 DPI default, which was too low-resolution for EasyOCR to read reliably.

Does NOT touch the main extractor_checkpoint.json or new_extractor.py's own
run() flow -- this is a separate, targeted re-pass with its own checkpoint,
safely interruptible/resumable without disturbing the main pipeline.

Only OVERWRITES the existing (bad) output JSON in extracted_text_clean/ if the
new OCR pass produces MORE chunks than the old one had -- never replaces
something with a worse result.

USAGE:
    python reprocess_unresolved_ocr.py                # process everything, resumable
    python reprocess_unresolved_ocr.py --limit 10      # process only the first N (testing)
"""

import os
import sys
import json
import signal
import argparse

import pandas as pd

import new_extractor as ext

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
except Exception:
    pass

UNRESOLVED_FILES = [
    "unresolved_no_content_extracted.json",
    "unresolved_ocr_garbled.json",
]
EXCEL_FILE = "pdf_name_metadata.xlsx"
OUTPUT_DIR = "extracted_text_clean"
CHECKPOINT_FILE = "reprocess_unresolved_ocr_checkpoint.json"
LOG_FILE = "reprocess_unresolved_ocr_errors.log"

_INTERRUPTED = False


def _handle_sigint(signum, frame):
    global _INTERRUPTED
    if not _INTERRUPTED:
        print("\nCtrl+C received -- stopping after current file, checkpoint will be saved...", flush=True)
    _INTERRUPTED = True


signal.signal(signal.SIGINT, _handle_sigint)


def load_checkpoint() -> set:
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, "r", encoding="utf-8") as f:
            return set(json.load(f))
    return set()


def save_checkpoint(done: set):
    with open(CHECKPOINT_FILE, "w", encoding="utf-8") as f:
        json.dump(sorted(done), f, indent=2)


def log_error(name: str, error) -> None:
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"{name} -> {error}\n")


def reprocess_one(json_name: str, row: dict) -> tuple:
    """Returns (status, new_chunk_count). status in
    {'improved', 'unchanged', 'no_text', 'error'}."""
    pdf_path = row["actual_path"]
    output_file = os.path.join(OUTPUT_DIR, json_name)

    old_chunk_count = 0
    if os.path.exists(output_file):
        try:
            with open(output_file, "r", encoding="utf-8") as f:
                old_chunk_count = json.load(f).get("num_chunks", 0)
        except Exception:
            old_chunk_count = 0

    ocr_pages = ext.ocr_text_pages(pdf_path, max_pages=ext.OCR_MAX_PAGES)
    text_parts, page_nums = [], []
    for page_num, page_text in ocr_pages:
        clean_page = ext.clean_text(page_text)
        if clean_page:
            text_parts.append(clean_page)
            page_nums.append(page_num)
    full_text = " ".join(text_parts)

    if not full_text.strip():
        return "no_text", 0

    doc_meta = ext.extract_document_metadata(full_text, row)
    page_paras = ext.split_page_paragraphs(list(zip(page_nums, text_parts)))
    chunks = ext.build_chunks(page_paras)

    if len(chunks) <= old_chunk_count:
        return "unchanged", len(chunks)

    output_data = {
        "generated_name": row["generated_name"],
        "actual_filename": row["actual_filename"],
        "court": row["court"],
        "case_type": row["case_type"],
        "year": row["year"],
        "used_ocr": True,
        "num_chunks": len(chunks),
        "case_number": doc_meta["case_number"],
        "date_of_order": doc_meta["date_of_order"],
        "judge": doc_meta["judge"],
        "sections_cited": doc_meta["sections_cited"],
        "citations": doc_meta["citations"],
        "parties": doc_meta["parties"],
        "chunks": chunks,
    }
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    return "improved", len(chunks)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None,
                         help="Only process the first N pending files (for testing).")
    args = parser.parse_args()

    files = set()
    for fname in UNRESOLVED_FILES:
        with open(fname, encoding="utf-8") as f:
            for row in json.load(f):
                files.add(row["file"])

    df = pd.read_excel(EXCEL_FILE)
    df["expected_json"] = df["generated_name"].astype(str).str.replace(".pdf", ".json", regex=False)
    df = df[df["expected_json"].isin(files)]
    row_by_json = {r["expected_json"]: r for r in df.to_dict("records")}

    done = load_checkpoint()
    pending = [f for f in sorted(files) if f not in done]
    if args.limit:
        pending = pending[: args.limit]

    print(f"Total unresolved files: {len(files)}", flush=True)
    print(f"Already reprocessed:    {len(done)}", flush=True)
    print(f"Processing this run:    {len(pending)}\n", flush=True)

    if ext.OCR_SUPPORTED:
        ext.get_ocr_reader()

    improved, unchanged, no_text, failed = 0, 0, 0, 0

    for i, json_name in enumerate(pending, 1):
        if _INTERRUPTED:
            break

        row = row_by_json.get(json_name)
        if row is None:
            print(f"[{i}/{len(pending)}] SKIP (no Excel row match): {json_name}", flush=True)
            done.add(json_name)
            save_checkpoint(done)
            continue

        try:
            status, new_count = reprocess_one(json_name, row)
            if status == "improved":
                improved += 1
                print(f"[{i}/{len(pending)}] IMPROVED -> {new_count} chunks: {json_name}", flush=True)
            elif status == "unchanged":
                unchanged += 1
                print(f"[{i}/{len(pending)}] no improvement ({new_count} chunks): {json_name}", flush=True)
            else:
                no_text += 1
                print(f"[{i}/{len(pending)}] still no usable text: {json_name}", flush=True)
        except Exception as e:
            failed += 1
            log_error(json_name, e)
            print(f"[{i}/{len(pending)}] ERROR: {json_name} -> {e}", flush=True)

        done.add(json_name)
        save_checkpoint(done)

        if i % 25 == 0:
            print(f"    Progress: {i}/{len(pending)} | improved={improved} unchanged={unchanged} no_text={no_text} failed={failed}\n", flush=True)

    print(f"\nDone this run. improved={improved} unchanged={unchanged} no_text={no_text} failed={failed}", flush=True)
    print(f"Total progress: {len(done)}/{len(files)}", flush=True)
    if _INTERRUPTED:
        print("Stopped early by Ctrl+C -- re-run to resume from checkpoint.", flush=True)


if __name__ == "__main__":
    main()
