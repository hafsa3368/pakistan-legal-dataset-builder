import json
from pathlib import Path

CHECKPOINT = Path("checkpoint.json")
EXTRACT_DIR = Path("extracted_pdfs")

# Output filenames (as in Excel) that are still "Unknown" and need retry
target_output_names = {
    "pdfs_lhc_unknown_case_2681.pdf",
    "pdfs_lhc_unknown_case_2687.pdf",
    "SHC_pdfs_unknown_case_2694.pdf",
    "SHC_pdfs_unknown_case_2694_1.pdf",
    "SHC_pdfs_unknown_case_2695.pdf",
    "SHC_pdfs_unknown_case_2695_1.pdf",
    "SHC_pdfs_unknown_case_2696.pdf",
    "SHC_pdfs_unknown_case_2696_1.pdf",
    "SHC_pdfs_unknown_case_2697.pdf",
    "SHC_pdfs_unknown_case_2697_1.pdf",
    "SHC_pdfs_unknown_case_2698.pdf",
    "SHC_pdfs_unknown_case_2698_1.pdf",
    "SHC_pdfs_unknown_case_2699.pdf",
    "SHC_pdfs_unknown_case_2699_1.pdf",
    "SHC_pdfs_unknown_case_269_1.pdf",
    "SHC_pdfs_unknown_case_270.pdf",
    "SHC_pdfs_unknown_case_2700.pdf",
    "SHC_pdfs_unknown_case_2700_1.pdf",
    "SHC_pdfs_unknown_case_2701.pdf",
    "SHC_pdfs_unknown_case_2701_1.pdf",
    "SHC_pdfs_unknown_case_2702.pdf",
    "SHC_pdfs_unknown_case_2702_1.pdf",
    "SHC_pdfs_unknown_case_2703.pdf",
    "SHC_pdfs_unknown_case_2703_1.pdf",
    "SHC_pdfs_unknown_case_2704.pdf",
    "SHC_pdfs_unknown_case_2705.pdf",
    "SHC_pdfs_unknown_case_2705_1.pdf",
    "SHC_pdfs_unknown_case_2706.pdf",
    "SHC_pdfs_unknown_case_2706_1.pdf",
    "SHC_pdfs_unknown_case_2707.pdf",
    "SHC_pdfs_unknown_case_2707_1.pdf",
}

done = set(json.load(open(CHECKPOINT)))
print(f"Checkpoint entries before: {len(done)}")

# A checkpoint entry is the relative path like "pdfs/<...>.pdf" or "shc/<...>.pdf"
# We need to find which checkpoint keys correspond to the unknown output files.
# Strategy: scan all PDFs, compute output filename, match against target set, remove from done.

import sys
sys.path.insert(0, ".")

# Re-implement minimal output-filename logic (must match process_pdfs_fixed.py)
import re

def clean(text):
    return re.sub(r'[^a-zA-Z0-9_-]', '_', str(text)).strip('_')

def make_output_filename(path, extract_dir, folder, tp, year=None):
    if year is None:
        m = re.search(r"(19|20)\d{2}", path.name)
        year = m.group() if m else ""

    folder_label = path.relative_to(extract_dir / Path(folder)).parent.as_posix().replace('/', '_')
    if folder_label == '.':
        folder_label = folder

    safe_name = clean(path.stem)[:40]
    parts = [tp, folder_label]
    if year:
        parts.append(year)
    parts.append(safe_name)
    return "_".join([clean(part) for part in parts if part]) + ".pdf"

ZIP_FILES = ["pdfs.zip", "shc.zip"]
removed = 0
for z in ZIP_FILES:
    root = EXTRACT_DIR / Path(z).stem
    if not root.exists():
        continue
    tp = "LHC" if "lhc" in z.lower() else "SHC"
    for pdf in root.rglob("*.pdf"):
        out_name = make_output_filename(pdf, EXTRACT_DIR, root.name, tp)
        if out_name in target_output_names:
            key = str(pdf.relative_to(EXTRACT_DIR)).replace('\\', '/')
            if key in done:
                done.discard(key)
                removed += 1
                print(f"Removed from checkpoint: {key} -> {out_name}")

print(f"\nRemoved {removed} entries")
print(f"Checkpoint entries after: {len(done)}")

json.dump(sorted(done), open(CHECKPOINT, "w"))
print("Saved checkpoint.json")