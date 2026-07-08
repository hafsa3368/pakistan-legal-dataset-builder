"""
Cleanup Script — Run this BEFORE re-running legal_extractor.py
Deletes:
  1. All .json files in extracted_text_clean/
  2. extractor_checkpoint.json (so nothing is skipped)
"""

import os
import glob

OUTPUT_DIR      = "extracted_text_clean"
CHECKPOINT_FILE = "extractor_checkpoint.json"

def cleanup():
    deleted_jsons = 0
    deleted_checkpoint = 0

    # ── 1. Delete all JSONs in extracted_text_clean/ ──────────────────
    if os.path.exists(OUTPUT_DIR):
        json_files = glob.glob(os.path.join(OUTPUT_DIR, "*.json"))
        total = len(json_files)

        if total == 0:
            print(f"📂 '{OUTPUT_DIR}' is already empty — nothing to delete.")
        else:
            print(f"🗑  Found {total} JSON files in '{OUTPUT_DIR}/' — deleting...")
            for f in json_files:
                os.remove(f)
                deleted_jsons += 1
            print(f"✅ Deleted {deleted_jsons} JSON files.")
    else:
        print(f"⚠  Folder '{OUTPUT_DIR}' does not exist — skipping.")

    # ── 2. Delete checkpoint ───────────────────────────────────────────
    if os.path.exists(CHECKPOINT_FILE):
        os.remove(CHECKPOINT_FILE)
        deleted_checkpoint = 1
        print(f"✅ Deleted checkpoint: {CHECKPOINT_FILE}")
    else:
        print(f"⚠  Checkpoint file not found — skipping.")

    print(f"\n🎉 Cleanup done! Now run:  python legal_extractor.py")

if __name__ == "__main__":
    confirm = input("⚠  This will permanently delete all old JSONs and checkpoint. Type YES to continue: ")
    if confirm.strip().upper() == "YES":
        cleanup()
    else:
        print("❌ Cancelled. Nothing was deleted.")