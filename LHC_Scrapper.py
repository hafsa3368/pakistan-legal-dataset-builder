from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from webdriver_manager.chrome import ChromeDriverManager
import time, os, pdfplumber, re, shutil

# ============================================================
# 📁 SETUP
# ============================================================
BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
DOWNLOAD_DIR = os.path.join(BASE_DIR, "downloads_temp")
PDF_DIR      = os.path.join(BASE_DIR, "pdfs")
LOG_FILE     = os.path.join(BASE_DIR, "downloaded_urls.txt")

os.makedirs(DOWNLOAD_DIR, exist_ok=True)

FOLDERS = ["civil", "criminal", "bail", "contempt", "service", "tax", "uncategorized"]
for f in FOLDERS:
    os.makedirs(os.path.join(PDF_DIR, f), exist_ok=True)

# ============================================================
# 📋 RESUME SUPPORT
# ============================================================
def load_seen_urls():
    if not os.path.exists(LOG_FILE):
        return set()
    with open(LOG_FILE, "r", encoding="utf-8") as f:
        return set(line.strip().lower() for line in f if line.strip())

def save_url(url):
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(url.strip().lower() + "\n")

seen_urls = load_seen_urls()
print(f"📋 {len(seen_urls)} files already downloaded — skip hongi")

# ============================================================
# 🌐 BROWSER
# ============================================================
options = webdriver.ChromeOptions()
options.add_argument("--no-sandbox")
options.add_argument("--window-size=1920,1080")
options.add_experimental_option("prefs", {
    "download.default_directory": DOWNLOAD_DIR,
    "plugins.always_open_pdf_externally": True
})

driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

# ============================================================
# 📖 READ PDF
# ============================================================
def read_pdf(path):
    try:
        with pdfplumber.open(path) as pdf:
            return "\n".join(p.extract_text() or "" for p in pdf.pages[:2]).lower()
    except:
        return ""

# ============================================================
# 🏷 CITATION FROM URL
# ============================================================
def extract_citation(url):
    name = os.path.splitext(os.path.basename(url))[0]
    if re.match(r'^\d{4}LHC\d+$', name, re.IGNORECASE):
        return name.upper()
    return re.sub(r'[^a-zA-Z0-9]', '', name) or "case"

# ============================================================
# 🔥 NEW: FILE ALREADY EXIST CHECK
# ============================================================
def file_exists_by_citation(citation):
    for root, dirs, files in os.walk(PDF_DIR):
        for f in files:
            if citation.lower() in f.lower():
                return True
    return False

# ============================================================
# 🧠 CLASSIFY (FIXED)
# ============================================================
def classify(text):
    text = text.lower()

    if "section 302" in text or "murder" in text or "p.p.c" in text:
        return "criminal"

    if "bail" in text:
        return "bail"

    if "service tribunal" in text or "government servant" in text:
        return "service"

    # ⚠️ FIX: Taxila vs Tax
    if "income tax" in text or "sales tax" in text or "customs" in text or "fbr" in text:
        return "tax"

    if "contempt of court" in text:
        return "contempt"

    return "civil"

def topic(text):
    if "contract" in text: return "contract"
    if "election" in text: return "election"
    if "appeal" in text: return "appeal"
    return "general"

def year(text):
    m = re.findall(r"\b(19\d{2}|20\d{2})\b", text)
    return m[-1] if m else "unknown"

# ============================================================
# 📦 SAVE FILE
# ============================================================
def save_file(old, folder, name):
    target = os.path.join(PDF_DIR, folder)
    os.makedirs(target, exist_ok=True)

    final = os.path.join(target, name)

    i = 1
    while os.path.exists(final):
        final = os.path.join(target, name.replace(".pdf", f"_{i}.pdf"))
        i += 1

    shutil.move(old, final)
    return final

# ============================================================
# ⏳ WAIT FOR DOWNLOAD
# ============================================================
def wait_for_new_pdf(before_files, timeout=60):
    for _ in range(timeout):
        current = set(
            f for f in os.listdir(DOWNLOAD_DIR)
            if f.endswith(".pdf") and not f.endswith(".crdownload")
        )
        new = current - before_files
        if new:
            time.sleep(1)
            return os.path.join(DOWNLOAD_DIR, new.pop())
        time.sleep(1)
    return ""

# ============================================================
# 🔄 SCRAPE PAGE
# ============================================================
def scrape_page():
    main_window = driver.window_handles[0]
    rows = driver.find_elements(By.CSS_SELECTOR, "tr")

    for r in rows:
        pdf_url = ""

        for a in r.find_elements(By.TAG_NAME, "a"):
            href = a.get_attribute("href") or ""
            if ".pdf" in href.lower():
                pdf_url = href.strip()
                break

        if not pdf_url:
            continue

        citation = extract_citation(pdf_url)

        # ✅ URL duplicate
        if pdf_url.lower() in seen_urls:
            print("⏭ Skip URL:", citation)
            continue

        # ✅ FILE duplicate (MAIN FIX)
        if file_exists_by_citation(citation):
            print("⏭ Already exists:", citation)
            continue

        before_files = set(f for f in os.listdir(DOWNLOAD_DIR) if f.endswith(".pdf"))

        driver.execute_script(f"window.open('{pdf_url}', '_blank');")
        time.sleep(2)

        try:
            driver.switch_to.window(driver.window_handles[-1])
        except:
            driver.switch_to.window(main_window)
            continue

        file_path = wait_for_new_pdf(before_files)

        if file_path:
            text = read_pdf(file_path)

            folder = classify(text)
            tp     = topic(text)
            yr     = year(text)

            filename = f"{folder}_{tp}_lhc_{yr}_{citation}.pdf"
            final    = save_file(file_path, folder, filename)

            seen_urls.add(pdf_url.lower())
            save_url(pdf_url)

            print("✅ Saved:", os.path.basename(final), "→", folder)
        else:
            print("⚠️ Failed:", citation)

        # 🔁 close tab safely
        try:
            if len(driver.window_handles) > 1:
                driver.close()
        except:
            pass

        try:
            driver.switch_to.window(main_window)
        except:
            driver.switch_to.window(driver.window_handles[0])

        time.sleep(1)

# ============================================================
# 🚀 MAIN
# ============================================================
driver.get("https://data.lhc.gov.pk/reported_judgments/judgments_approved_for_reporting")

print("\n👉 Filter/search apply karo → ENTER dabao")
input()

while True:
    scrape_page()
    nxt = input("\n➡️ Next page → ENTER | Stop = n: ")
    if nxt.lower() == "n":
        break

driver.quit()
print("\n🎯 DONE — PERFECT DATASET READY")