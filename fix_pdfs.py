import os, pdfplumber, shutil, re

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PDF_DIR = os.path.join(BASE_DIR, "pdfs")

FOLDERS = ["civil", "criminal", "bail", "contempt", "service", "tax", "writ"]

def classify(text):
    text = (text or "").lower()
    words = re.findall(r'\b\w+\b', text)

    if "criminal" in words:
        return "criminal"
    if "bail" in words:
        return "bail"
    if "service" in words:
        return "service"
    if "tax" in words or "taxation" in words:
        return "tax"
    if "contempt" in words:
        return "contempt"
    if "writ" in words:
        return "writ"

    return "civil"

def move_file(path, folder):
    name = os.path.basename(path)
    target = os.path.join(PDF_DIR, folder, name)

    base, ext = os.path.splitext(name)
    i = 1
    while os.path.exists(target):
        target = os.path.join(PDF_DIR, folder, f"{base}_{i}{ext}")
        i += 1

    shutil.move(path, target)

# ============================================================
for folder in FOLDERS:

    folder_path = os.path.join(PDF_DIR, folder)

    for file in os.listdir(folder_path):

        if not file.endswith(".pdf"):
            continue

        file_path = os.path.join(folder_path, file)

        try:
            with pdfplumber.open(file_path) as p:
                text = p.pages[0].extract_text() or ""
        except:
            continue

        correct_folder = classify(text)

        if correct_folder != folder:
            print(f"🔁 Moving {file} → {correct_folder}")
            move_file(file_path, correct_folder)

print("🎯 RE-CLEAN DONE")