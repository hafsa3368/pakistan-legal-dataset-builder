import json
import pandas as pd
import shutil

EXCEL_FILE = "supreme_court_ai_metadata.xlsx"
CHECKPOINT_FILE = "checkpoint.json"

# 1. Excel se saari processed filenames nikaalo
df = pd.read_excel(EXCEL_FILE)
excel_filenames = set(df['File Name'].dropna().astype(str).tolist())
print(f"Excel mein {len(excel_filenames)} unique filenames mili")

# 2. Purana checkpoint padh lo
with open(CHECKPOINT_FILE, 'r') as f:
    old_checkpoint = json.load(f)
print(f"Purane checkpoint mein {len(old_checkpoint['processed'])} filenames thi")

# 3. Backup pehle lo
shutil.copy(CHECKPOINT_FILE, CHECKPOINT_FILE + ".backup")

# 4. Naya checkpoint = excel ki filenames
new_checkpoint = {"processed": sorted(excel_filenames)}
with open(CHECKPOINT_FILE, 'w') as f:
    json.dump(new_checkpoint, f, indent=2)

print(f"✅ Naya checkpoint likh diya: {len(new_checkpoint['processed'])} filenames")
print(f"✅ Purana checkpoint '{CHECKPOINT_FILE}.backup' mein safe hai")