from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
import time, os, shutil

# ================= PATHS =================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DOWNLOAD_DIR = os.path.join(BASE_DIR, "downloads_temp")
PDF_DIR = os.path.join(BASE_DIR, "pdfs")

os.makedirs(DOWNLOAD_DIR, exist_ok=True)

FOLDERS = ["civil","criminal","bail","contempt","service","tax","uncategorized"]
for f in FOLDERS:
    os.makedirs(os.path.join(PDF_DIR, f), exist_ok=True)

# ================= DRIVER =================
options = webdriver.ChromeOptions()
options.add_experimental_option("prefs", {
    "download.default_directory": DOWNLOAD_DIR,
    "plugins.always_open_pdf_externally": True
})

driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
wait = WebDriverWait(driver, 10)

# ================= FAST DOWNLOAD =================
def wait_download(before, timeout=30):  # 🔥 FAST
    for _ in range(timeout):
        now = set(os.listdir(DOWNLOAD_DIR))
        new = now - before

        for f in new:
            if f.endswith(".pdf") and not f.endswith(".crdownload"):
                return os.path.join(DOWNLOAD_DIR, f)

        time.sleep(1)
    return ""

# ================= MOVE + SHOW =================
def move_file(path, folder, idx):
    target_folder = os.path.join(PDF_DIR, folder)
    os.makedirs(target_folder, exist_ok=True)

    target = os.path.join(target_folder, f"case_{idx}.pdf")

    i = 1
    while os.path.exists(target):
        target = os.path.join(target_folder, f"case_{idx}_{i}.pdf")
        i += 1

    shutil.move(path, target)

    # ✅ SHOW OUTPUT
    print(f"📁 Saved in: {folder} → {os.path.basename(target)}")

# ================= CLASSIFY (FAST) =================
def classify(text):
    t = (text or "").lower()
    if "criminal" in t: return "criminal"
    if "bail" in t: return "bail"
    if "tax" in t: return "tax"
    if "service" in t: return "service"
    if "contempt" in t: return "contempt"
    return "civil"

# ================= SCRAPER =================
def scrape():

    driver.get("https://caselaw.shc.gov.pk/caselaw/search-all/search")
    input("👉 ENTER after filter")

    wait.until(EC.presence_of_element_located((By.ID, "tblExport")))

    page = 1

    while True:

        print(f"\n📄 Page {page}")

        rows = driver.find_elements(By.XPATH, "//table[@id='tblExport']//tbody//tr")

        for i in range(len(rows)):

            try:
                rows = driver.find_elements(By.XPATH, "//table[@id='tblExport']//tbody//tr")
                row = rows[i]

                # ================= CLICK BUTTON =================
                btn = row.find_element(By.XPATH, ".//td[16]//a//button")

                before = set(os.listdir(DOWNLOAD_DIR))

                driver.execute_script("arguments[0].click();", btn)

                file_path = wait_download(before)

                if file_path:

                    # (optional) text skip → FAST MODE
                    folder = "civil"

                    move_file(file_path, folder, i+1)

                    print(f"✅ Row {i+1} DONE")

                else:
                    print(f"⚠️ Row {i+1} download fail")

            except Exception as e:
                print(f"⚠️ Row {i+1} error:", e)

        # ================= NEXT PAGE =================
        try:
            nxt = driver.find_element(By.XPATH, "//a[contains(.,'Next')]")
            driver.execute_script("arguments[0].click();", nxt)
            time.sleep(2)  # 🔥 reduced
            page += 1
        except:
            print("🛑 No Next Page")
            break

    driver.quit()
    print("🎯 DONE")

# ================= RUN =================
scrape()