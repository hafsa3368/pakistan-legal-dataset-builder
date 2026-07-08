"""
Local PDFs → Google Drive
Koi website nahi, koi browser nahi, sirf local files.
"""

import pickle
import pdfplumber
from pathlib import Path
from collections import Counter

from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request

BASE_DIR   = Path(__file__).parent
TOKEN_PATH = BASE_DIR / "token.pkl"
CREDS_PATH = BASE_DIR / "credentials.json"
SCOPES     = ["https://www.googleapis.com/auth/drive"]


# ============================================================
# AUTH
# ============================================================
def get_drive_service():
    creds = None
    if TOKEN_PATH.exists():
        with open(TOKEN_PATH, "rb") as f:
            creds = pickle.load(f)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(str(CREDS_PATH), SCOPES)
            creds = flow.run_local_server(port=0)
        with open(TOKEN_PATH, "wb") as f:
            pickle.dump(creds, f)
    return build("drive", "v3", credentials=creds)


# ============================================================
# FOLDER BANANA
# ============================================================
def get_or_create_folder(service, name, parent_id=None):
    query = f"name='{name}' and mimeType='application/vnd.google-apps.folder' and trashed=false"
    if parent_id:
        query += f" and '{parent_id}' in parents"
    res = service.files().list(q=query, fields="files(id)").execute()
    files = res.get("files", [])
    if files:
        return files[0]["id"]
    meta = {"name": name, "mimeType": "application/vnd.google-apps.folder"}
    if parent_id:
        meta["parents"] = [parent_id]
    folder = service.files().create(body=meta, fields="id").execute()
    print(f"  📁 Folder bana: {name}")
    return folder["id"]


# ============================================================
# PDF PADHNA
# ============================================================
def read_pdf_text(path: Path) -> str:
    try:
        with pdfplumber.open(str(path)) as pdf:
            return "\n".join(
                page.extract_text() or "" for page in pdf.pages[:2]
            ).lower()
    except Exception as e:
        print(f"  ⚠️  {path.name} nahi parha: {e}")
        return ""


# ============================================================
# CLASSIFY
# ============================================================
def classify(text: str) -> str:
    if any(k in text for k in ["section 302", "murder", "qatl"]):
        return "criminal"
    if "bail" in text:
        return "bail"
    if any(k in text for k in ["service tribunal", "federal public service"]):
        return "service"
    if any(k in text for k in ["income tax", "sales tax", "fbr", "customs"]):
        return "tax"
    if "contempt of court" in text:
        return "contempt"
    return "civil"


# ============================================================
# UPLOAD
# ============================================================
def upload_file(service, file_path: Path, folder_id: str):
    meta  = {"name": file_path.name, "parents": [folder_id]}
    media = MediaFileUpload(str(file_path), mimetype="application/pdf", resumable=True)
    service.files().create(body=meta, media_body=media, fields="id").execute()


# ============================================================
# MAIN
# ============================================================
def main():

    # Local folder
    raw = input("📂 Local folder ka path daalo: ").strip().strip('"')
    local_folder = Path(raw)

    if not local_folder.is_dir():
        print("❌ Folder nahi mila.")
        return

    pdfs = sorted(local_folder.rglob("*.pdf"))
    if not pdfs:
        print("❌ Koi PDF nahi mili.")
        return

    print(f"\n✅ {len(pdfs)} PDF mili hain.\n")

    # Drive folder naam
    root_name = input("☁️  Drive pe main folder ka naam (Enter = Court_PDFs): ").strip()
    if not root_name:
        root_name = "Court_PDFs"

    # Drive connect
    print("\n🔐 Google Drive se connect ho raha hai...")
    service = get_drive_service()
    print("✅ Connected!\n")

    # Root folder
    root_id = get_or_create_folder(service, root_name)

    # Classify
    print("🔍 PDFs classify ho rahi hain...\n")
    classified = []
    for pdf in pdfs:
        category = classify(read_pdf_text(pdf))
        classified.append((pdf, category))
        print(f"  {pdf.name:50s} → {category}")

    # Upload
    print(f"\n⬆️  Upload shuru...\n")
    subfolder_cache = {}
    success = 0
    failed  = 0

    for i, (pdf_path, category) in enumerate(classified, 1):
        print(f"  [{i}/{len(classified)}] {pdf_path.name} ({category})", end="  ")
        if category not in subfolder_cache:
            subfolder_cache[category] = get_or_create_folder(service, category, root_id)
        try:
            upload_file(service, pdf_path, subfolder_cache[category])
            print("✅")
            success += 1
        except Exception as e:
            print(f"❌ ({e})")
            failed += 1

    # Summary
    print(f"\n{'─' * 40}")
    print(f"🎯 Mukammal!  ✅ {success} uploaded   ❌ {failed} failed")
    print(f"\n📁 {root_name}/")
    for cat, count in sorted(Counter(c for _, c in classified).items()):
        print(f"   └─ 📂 {cat}/  ({count} file)")


if __name__ == "__main__":
    main()