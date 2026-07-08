from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from webdriver_manager.chrome import ChromeDriverManager
import time, os, shutil, pdfplumber, re

# ================= PATHS =================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DOWNLOAD_DIR = os.path.join(BASE_DIR, "downloads_temp")
PDF_DIR = os.path.join(BASE_DIR, "pdfs")

os.makedirs(DOWNLOAD_DIR, exist_ok=True)

for f in ["civil","criminal","bail","contempt","service","tax"]:
    os.makedirs(os.path.join(PDF_DIR, f), exist_ok=True)

# ================= DRIVER =================
options = webdriver.ChromeOptions()
options.add_experimental_option("prefs", {
    "download.default_directory": DOWNLOAD_DIR,
    "plugins.always_open_pdf_externally": True
})

driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

# ================= FAST DOWNLOAD =================
def wait_download(before, timeout=15):
    for _ in range(timeout):
        now = set(os.listdir(DOWNLOAD_DIR))
        new = now - before

        for f in new:
            if f.endswith(".pdf") and not f.endswith(".crdownload"):
                return os.path.join(DOWNLOAD_DIR, f)

        time.sleep(1)
    return ""

# ================= MOVE =================
def read_pdf(path):
    try:
        with pdfplumber.open(path) as pdf:
            if not pdf.pages:
                return ""
            first_page = pdf.pages[0]
            text = first_page.extract_text() or ""
            text = text.replace('\r', '\n').strip()
            # Split into paragraphs by blank lines or repeated line breaks
            paragraphs = [p.strip() for p in re.split(r"\n\s*\n+", text) if p.strip()]
            if paragraphs:
                return paragraphs[0].lower()
            # Fallback: first 3 non-empty lines if no paragraphs found
            lines = [line.strip() for line in text.splitlines() if line.strip()]
            return " ".join(lines[:3]).lower()
    except Exception:
        return ""


def extract_citation_from_row(row):
    try:
        # Try to get case number from table columns (usually td[1] or td[2])
        for col_idx in [1, 2, 3]:
            try:
                cell = row.find_element(By.XPATH, f".//td[{col_idx}]")
                text = cell.text.strip()
                if text and len(text) > 0:
                    # Sanitize: remove spaces, special chars, keep alphanumeric
                    citation = re.sub(r'[^a-zA-Z0-9]', '', text)
                    if citation:
                        return citation
            except:
                continue
    except:
        pass
    return "unknown"

def classify(text):
    text = text.lower()
    
    if "section 302" in text or "murder" in text or "p.p.c" in text:
        return "criminal"
    
    if "bail" in text:
        return "bail"
    
    if "service tribunal" in text or "government servant" in text:
        return "service"
    
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

def save_file(old, folder, name):
    target = os.path.join(PDF_DIR, folder)
    os.makedirs(target, exist_ok=True)
    
    final = os.path.join(target, name)
    
    i = 1
    while os.path.exists(final):
        final = os.path.join(target, name.replace(".pdf", f"_{i}.pdf"))
        i += 1
    
    shutil.move(old, final)
    print(f"📁 Saved → {folder} → {os.path.basename(final)}")
    return final

# ================= SCRAPER =================
def scrape():

    driver.get("https://caselaw.shc.gov.pk/caselaw/search-all/search")
    input("👉 ENTER after filter")

    time.sleep(5)

    page = 1
    total_rows = 0

    while True:

        print(f"\n📄 Page {page}")

        rows = driver.find_elements(By.XPATH, "//table[@id='tblExport']//tbody//tr")
        print(f"📊 Found {len(rows)} rows on page {page}")

        total_rows += len(rows)

        start_row =0 if page == 1 else 0  # 🔥 ONLY FIRST PAGE START 690

        for i in range(start_row, len(rows)):

            try:
                rows = driver.find_elements(By.XPATH, "//table[@id='tblExport']//tbody//tr")
                row = rows[i]

                btn = row.find_element(By.XPATH, ".//td[16]//a//button")

                before = set(os.listdir(DOWNLOAD_DIR))

                # ================= IMPORTANT FIX =================
                # 🔥 NO NEW TAB, NO SWITCH
                driver.execute_script("arguments[0].click();", btn)

                file_path = wait_download(before)

                if file_path:
                    text = read_pdf(file_path)
                    
                    folder = classify(text)
                    tp     = topic(text)
                    yr     = year(text)
                    citation = extract_citation_from_row(row)  # Get unique case number from table
                    
                    filename = f"{folder}_{tp}_shc_{yr}_{citation}.pdf"
                    final    = save_file(file_path, folder, filename)
                    
                    print(f"✅ Row {i+1} ({citation}) DONE")
                else:
                    print(f"⚠️ Row {i+1} SKIPPED (no download)")

            except Exception as e:
                print(f"⚠️ Row {i+1} ERROR → SKIPPED")

        # ================= NEXT PAGE =================
        try:
            nxt = driver.find_element(By.XPATH, "//a[contains(.,'Next')]")
            driver.execute_script("arguments[0].click();", nxt)
            time.sleep(3)
            page += 1

        except:
            print("🛑 NO MORE PAGES")
            break

    print(f"📈 Total rows across all pages: {total_rows}")
    driver.quit()
    print("🎯 DONE")

# ================= RUN =================
scrape()