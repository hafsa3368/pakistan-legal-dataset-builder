"""
Legal PDF Extractor — Paragraph-Aware Chunking for Pakistani Court Documents
Handles 1000+ PDFs with crash-safe resume, OCR fallback, and rich metadata per chunk.

v9 - Immediate Ctrl+C stop + progress counter:
  - Ctrl+C ab turant rok deta hai (thread pool bhi cancel_futures se turant band hota hai)
  - Har JSON banne ke baad "X/Y done | Z remaining" print hota hai
  - Checkpoint hamesha save hota hai (try/finally) — agli run wahi se resume karti hai

v8 - EasyOCR + ThreadPoolExecutor:
  - EasyOCR wapas (better accuracy, Urdu support)
  - Singleton model — ek baar load, saare threads share karte hain
  - Two-pass: Pass 1 normal extraction (fast), Pass 2 parallel OCR (ThreadPoolExecutor)
  - ThreadPoolExecutor use kiya ProcessPool ki jagah — model memory mein ek baar rehta hai
  - OCR_WORKERS = 2 default (EasyOCR internally bhi multi-threaded hai)

Install:
  pip install easyocr pymupdf pandas openpyxl pillow numpy beautifulsoup4 --break-system-packages
"""

import os
import sys
import json
import re
import gc
import html
import unicodedata
import traceback
import datetime
import signal
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed

# Force line-buffered stdout so progress prints show up immediately
# in the terminal instead of sitting in a buffer.
try:
    sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass

import pandas as pd
import fitz  # PyMuPDF
from PIL import Image

try:
    import numpy as np
    import easyocr
    OCR_SUPPORTED = True
except ImportError:
    OCR_SUPPORTED = False
    print("⚠  EasyOCR not found. Install: pip install easyocr pillow numpy --break-system-packages", flush=True)

try:
    from bs4 import BeautifulSoup
    BS4_SUPPORTED = True
except ImportError:
    BS4_SUPPORTED = False


# ==========================
# CONFIG
# ==========================
EXCEL_FILE          = "pdf_name_metadata.xlsx"
OUTPUT_DIR          = "extracted_text_clean"
CHECKPOINT_FILE     = "extractor_checkpoint.json"
STOP_FILE           = "extractor.stop"
LOG_FILE            = "extractor_errors.log"

TARGET_CHUNK_TOKENS = 400
MAX_CHUNK_TOKENS    = 600
OVERLAP_SENTENCES   = 2
OCR_MAX_PAGES       = 10      # max pages to OCR per PDF
OCR_WORKERS         = 2       # parallel OCR threads
                               # (2-3 recommended — EasyOCR is already multi-threaded internally)
                               # increase only if you have many CPU cores and enough RAM

# EasyOCR languages
# ['en'] = English only (faster)
# ['en', 'ur'] = English + Urdu (slower but handles Urdu text)
OCR_LANGUAGES       = ['en', 'ur']

FIX_MOJIBAKE_APOSTROPHE = False

# Ollama fallback — only called when regex could not find a field, so it
# does NOT slow down the fast path. Set OLLAMA_ENABLED = False to disable.
OLLAMA_ENABLED   = True
OLLAMA_URL       = "http://localhost:11434/api/generate"
OLLAMA_TAGS_URL  = "http://localhost:11434/api/tags"   # lightweight alive-check endpoint
OLLAMA_MODEL     = "llama3.2:3b"
OLLAMA_TIMEOUT   = 60          # seconds
OLLAMA_MAX_CHARS = 4000        # only send the first N chars of the doc (judge/case-no/date usually appear early)

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ==========================
# SINGLETON EasyOCR READER
# (loaded once, shared across all threads)
# ==========================
_OCR_READER = None

def get_ocr_reader():
    global _OCR_READER
    if _OCR_READER is None:
        print(f"🧠 Loading EasyOCR model (one-time) — languages: {OCR_LANGUAGES} ...", flush=True)
        _OCR_READER = easyocr.Reader(OCR_LANGUAGES, gpu=False)
        print("✅ EasyOCR model ready.\n", flush=True)
    return _OCR_READER


# ==========================
# OLLAMA METADATA FALLBACK
# (only called for fields regex couldn't find — keeps the fast path fast)
# ==========================
_OLLAMA_ALIVE = None   # cached after first check, so we don't re-check every PDF

def is_ollama_alive() -> bool:
    """
    Quick check using the lightweight /api/tags endpoint instead of hitting
    /api/generate — avoids a slow timeout on every single PDF if Ollama
    isn't running. Result is cached for the rest of the run.
    """
    global _OLLAMA_ALIVE
    if _OLLAMA_ALIVE is not None:
        return _OLLAMA_ALIVE
    try:
        resp = requests.get(OLLAMA_TAGS_URL, timeout=3)
        _OLLAMA_ALIVE = resp.status_code == 200
    except Exception:
        _OLLAMA_ALIVE = False
    if not _OLLAMA_ALIVE:
        print("⚠  Ollama not reachable — metadata fallback will be skipped for this run.", flush=True)
    return _OLLAMA_ALIVE


def ollama_fill_missing_metadata(full_text: str, missing_fields: list) -> dict:
    """
    Asks the local Ollama model to extract ONLY the fields regex missed.
    Returns a dict with just those fields (empty dict on any failure —
    caller should keep whatever regex already had in that case).
    """
    if not OLLAMA_ENABLED or not missing_fields:
        return {}

    if not is_ollama_alive():
        return {}

    snippet = full_text[:OLLAMA_MAX_CHARS]

    field_descriptions = {
        "judge":         "the name of the judge who authored/signed the order (not the typist's initials)",
        "case_number":   "the case number (e.g. 'Cr.B.A.No.S-994 of 2019')",
        "date_of_order": "the date the order/judgment was passed (DD.MM.YYYY if possible)",
        "sections_cited": "a list of legal sections/acts cited (e.g. 'Section 497 CrPC', 'u/s 498 Cr.P.C.')",
        "parties":       "a list of party names (petitioner/applicant/appellant)",
    }
    wanted = {k: field_descriptions[k] for k in missing_fields if k in field_descriptions}
    if not wanted:
        return {}

    prompt = (
        "You are extracting structured metadata from a Pakistani court judgment. "
        "Return ONLY a valid JSON object, no preamble, no markdown fences. "
        "Extract these fields:\n"
        + "\n".join(f'- "{k}": {v}' for k, v in wanted.items())
        + "\nIf a field cannot be found, use an empty string (or empty list for list fields).\n\n"
        + f"Document text:\n{snippet}"
    )

    try:
        resp = requests.post(
            OLLAMA_URL,
            json={
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False,
                "format": "json",
            },
            timeout=OLLAMA_TIMEOUT,
        )
        resp.raise_for_status()
        raw = resp.json().get("response", "").strip()
        raw = re.sub(r"^```json\s*|\s*```$", "", raw.strip())
        parsed = json.loads(raw)
        return {k: parsed[k] for k in wanted if k in parsed}
    except Exception as e:
        log_error("OLLAMA_FALLBACK", e)
        return {}


# ==========================
# CHECKPOINT HELPERS
# ==========================
def load_checkpoint():
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, "r", encoding="utf-8") as f:
            return set(json.load(f))
    return set()


def save_checkpoint(done_set):
    with open(CHECKPOINT_FILE, "w", encoding="utf-8") as f:
        json.dump(sorted(done_set), f, ensure_ascii=False, indent=2)


def log_error(generated_name, error):
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"[{datetime.datetime.now()}] {generated_name} -> {error}\n")
        f.write(traceback.format_exc() + "\n")


def stop_requested() -> bool:
    return os.path.exists(STOP_FILE)


def clear_stop_flag():
    if os.path.exists(STOP_FILE):
        try:
            os.remove(STOP_FILE)
        except Exception:
            pass


def log_page_error(pdf_path: str, page_num: int, error):
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"[{datetime.datetime.now()}] PAGE_ERROR "
                f"{os.path.basename(pdf_path)}:{page_num} -> {error}\n")
        f.write(traceback.format_exc() + "\n")


def print_progress(done_count: int, total: int):
    """Har PDF process hone ke baad ye line print hoti hai."""
    remaining = max(total - done_count, 0)
    print(f"📊 Progress: {done_count}/{total} done | {remaining} remaining\n", flush=True)


# ==========================
# CTRL+C HANDLING
# ==========================
# NOTE: EasyOCR/PyTorch calls run in native (C) code, which blocks Python
# from noticing a Ctrl+C signal until that native call returns — that's
# why Ctrl+C used to feel like it "did nothing" during OCR. A signal
# handler itself is NOT blocked by that (the OS delivers it to the main
# thread immediately); it just sets a flag here. The loops below check
# this flag after every PDF/page — so the script stops as soon as the
# current unit of work finishes, instead of hanging indefinitely.
_INTERRUPTED = False

def _handle_sigint(signum, frame):
    global _INTERRUPTED
    if not _INTERRUPTED:
        print("\n⏸ Ctrl+C received — current file/page finish hote hi ruk jayega "
              "(checkpoint save ho jayega, wait karein)...", flush=True)
    _INTERRUPTED = True

signal.signal(signal.SIGINT, _handle_sigint)


# ==========================
# TEXT CLEANING
# ==========================
def clean_text(text: str) -> str:
    if not text:
        return ""
    text = html.unescape(text)
    if BS4_SUPPORTED and "<" in text and ">" in text:
        soup = BeautifulSoup(text, "html.parser")
        for a_tag in soup.find_all("a", href=re.compile(r"#_ftn|#_edn|#_msocom", re.I)):
            a_tag.decompose()
        text = soup.get_text(separator=" ")
    else:
        text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\bMso\w+\b", " ", text)
    text = re.sub(r"\bo:p\b", " ", text)
    text = re.sub(r"\bst1:\w+\b", " ", text)
    text = re.sub(r"\[\d+\]", "", text)
    text = re.sub(r"\(\d+\)", "", text)
    text = text.encode("utf-8", "ignore").decode("utf-8", "ignore")
    if FIX_MOJIBAKE_APOSTROPHE:
        text = re.sub(r"(?<=\w)\ufffd(?=\w)", "'", text)
    text = text.replace("\ufffd", " ")
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", " ", text)
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s+([.,;:])", r"\1", text)
    return text.strip()


# ==========================
# PAGE-BY-PAGE PDF READER
# ==========================
def read_pdf_pages(pdf_path: str):
    """
    Yields (page_num, page_text) for every page with extractable text.
    Falls back to single-page EasyOCR if one page fails during normal extraction.
    """
    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        log_page_error(pdf_path, 0, f"failed to open PDF: {e}")
        return

    try:
        for page_num in range(len(doc)):
            page = None
            page_text = ""
            try:
                page = doc.load_page(page_num)
                page_text = page.get_text()
            except Exception as e:
                log_page_error(pdf_path, page_num + 1, f"page extract failed: {e}")
                if OCR_SUPPORTED:
                    page_text = _ocr_single_page_text(pdf_path, page_num + 1)
                else:
                    page_text = ""
            finally:
                if page is not None:
                    del page

            if page_text and page_text.strip():
                yield page_num + 1, page_text
    finally:
        try:
            doc.close()
        except Exception:
            pass
        gc.collect()


# ==========================
# OCR — PAGE-WISE (EasyOCR, shared singleton)
# ==========================
def ocr_text_pages(pdf_path: str, max_pages: int = OCR_MAX_PAGES) -> list[tuple[int, str]]:
    """
    Returns [(page_num, text), ...] using shared EasyOCR singleton.
    Page-wise output ensures proper downstream chunking (not 1 chunk).
    Called from threads — EasyOCR singleton is thread-safe for inference.
    """
    if not OCR_SUPPORTED:
        return []

    doc = None
    try:
        doc = fitz.open(pdf_path)
        reader = get_ocr_reader()   # shared singleton, loaded once
        results = []
        total_pages = min(max_pages, len(doc))

        for page_num in range(1, total_pages + 1):
            if _INTERRUPTED:
                # stop OCR-ing further pages of THIS pdf as soon as possible;
                # whatever pages we already OCR'd are still returned/used
                break
            try:
                page = doc.load_page(page_num - 1)
                pix = page.get_pixmap()
                img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
                arr = np.array(img)
                ocr_result = reader.readtext(arr)
                page_text = "\n".join([r[1] for r in ocr_result if r[1].strip()])
                if page_text.strip():
                    results.append((page_num, page_text))
                print(f"    [{os.path.basename(pdf_path)}] OCR page {page_num}/{total_pages} done", flush=True)
            except Exception as e:
                log_page_error(pdf_path, page_num, f"OCR page failed: {e}")
                continue

        return results

    except Exception as e:
        log_page_error(pdf_path, 0, f"OCR init failed: {e}")
        return []
    finally:
        if doc is not None:
            try:
                doc.close()
            except Exception:
                pass


def _ocr_single_page_text(pdf_path: str, page_num: int) -> str:
    """Single-page OCR — used as fallback inside read_pdf_pages for a broken page."""
    if not OCR_SUPPORTED:
        return ""
    doc = None
    try:
        doc = fitz.open(pdf_path)
        if page_num < 1 or page_num > len(doc):
            return ""
        reader = get_ocr_reader()
        page = doc.load_page(page_num - 1)
        pix = page.get_pixmap()
        img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
        arr = np.array(img)
        result = reader.readtext(arr)
        return "\n".join([r[1] for r in result if r[1].strip()])
    except Exception as e:
        log_page_error(pdf_path, page_num, f"Single-page OCR failed: {e}")
        return ""
    finally:
        if doc is not None:
            try:
                doc.close()
            except Exception:
                pass


# ==========================
# METADATA EXTRACTION
# ==========================
def extract_document_metadata(full_text: str, row: dict) -> dict:
    meta = {
        "court":     str(row.get("court", "")),
        "case_type": str(row.get("case_type", "")),
        "year":      str(row.get("year", "")),
    }

    # ── Case number ────────────────────────────────────────────────────
    case_no = re.search(
        r"("
        r"(?:Crl\.?\s*|CRL\.?\s*|Cr\.B\.A\.?\s*|Cr\.R\.A\.?\s*|Cr\.A\.?\s*|"
        r"Civil\s*|W\.P\.?\s*|Const\.?\s*|Criminal\s*|Misc\.?\s*)"
        r"(?:[\w\s\.\-]{0,40}?)"
        r"No\.?\s*[\w\-]+(?:/[\w]+)*"
        r"(?:\s+of\s+\d{4})?"
        r")",
        full_text, re.I
    )
    meta["case_number"] = case_no.group(1).strip() if case_no else ""

    # ── Date of order ──────────────────────────────────────────────────
    date_order = re.search(
        r"Date\s+of\s+(?:Hearing|Order)\s*[:\-]?\s*(\d{1,2}[.\-/]\d{1,2}[.\-/]\d{2,4})",
        full_text, re.I
    )
    if not date_order:
        date_order = re.search(
            r"(\d{2}\.\d{2}\.\d{4})\s*[.\s]+[A-Z][a-z]+.*?"
            r"(?:Advocate|ASC|DPG|APG|Counsel|A\.P\.G|D\.P\.G)",
            full_text
        )
    if not date_order:
        date_order = re.search(
            r"(?:^|\s)(\d{2}\.\d{2}\.\d{4})[\.\s]",
            full_text
        )
    if not date_order:
        date_order = re.search(
            r"(\d{1,2}(?:st|nd|rd|th)?\s+"
            r"(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|"
            r"Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)"
            r",?\s+\d{4})",
            full_text, re.I
        )
    meta["date_of_order"] = date_order.group(1).strip() if date_order else ""

    # ── Judge name ─────────────────────────────────────────────────────
    judge = re.search(
        r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,4})\s+J\s*[;:\-]+",
        full_text
    )
    if not judge:
        judge = re.search(
            r"\(([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,4}|"
            r"[A-Z]{2,}(?:\s+[A-Z]{2,}){1,4})\)\s*\n?\s*(?:JUDGE|Judge)",
            full_text
        )
    if not judge:
        judge = re.search(
            r"\*([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,4})\*",
            full_text
        )
    if not judge:
        present_block = re.search(
            r"PRESENT\s+([\s\S]{0,300}?)(?:CRL\.|Crl\.|ORDER|Date)",
            full_text, re.I
        )
        if present_block:
            judge = re.search(
                r"Justice\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,4})",
                present_block.group(1)
            )
    if not judge:
        judge = re.search(
            r"Justice\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,4})"
            r"(?!\s+(?:Act|Ordinance)\b)",
            full_text
        )
    if not judge:
        judge = re.search(
            r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})\s*\n?\s*JUDGE",
            full_text
        )
    if not judge:
        judge = re.search(
            r"JUDGE[.:]?\s*([A-Z][A-Za-z\/\s]{2,100}?)\s*(?:\n|$)",
            full_text
        )
        if judge:
            candidate = judge.group(1).strip()
            candidate = re.sub(
                r"\s+(?:ORDER|DATE|Advocate|APG|DPG|Counsel|A\.P\.G|D\.P\.G).*",
                "", candidate, flags=re.I
            ).strip()
            judge = re.match(r"([A-Z][A-Za-z\/\s]{2,100})$", candidate)
    meta["judge"] = judge.group(1).strip() if judge else ""

    # ── Sections cited ─────────────────────────────────────────────────
    raw_sections = re.findall(
        r"(?:[Ss]ections?|u/s|U/S|U/s)\s+"
        r"\d+[\w\-/]*"
        r"(?:\s*[,&]\s*\d+[\w\-/]*)*"
        r"(?:\s+(?:and|or)\s+\d+[\w\-/]*)?"
        r"(?:\s+(?:of\s+)?(?:the\s+)?"
        r"(?:[A-Z][A-Za-z]+\s*){0,5}"
        r"(?:Act|Code|Ordinance|Rules|Order|PPC|CPC|Cr\.?P\.?C\.?))?",
        full_text
    )
    clean_sections = []
    seen_s = set()
    for s in raw_sections:
        s = re.sub(r"\s+", " ", s).strip().rstrip(".,;:")
        if 3 < len(s) < 80 and s not in seen_s:
            seen_s.add(s)
            clean_sections.append(s)
    meta["sections_cited"] = clean_sections[:10]

    # ── Citations ──────────────────────────────────────────────────────
    raw_citations = []
    raw_citations += re.findall(
        r"\d{4}\s+"
        r"(?:SCMR|MLD|YLR|CLC|NLR|CLCN|PCrLJ|"
        r"P\.?\s*Cr\.?\s*L\.?\s*J\.?|P\s+Cr\.L\s+J)"
        r"[\s\-]*(?:Note\s+)?\d+",
        full_text
    )
    raw_citations += re.findall(
        r"PLD\s+\d{4}\s+[A-Z][A-Za-z]+\s+\d+",
        full_text
    )
    raw_citations += re.findall(
        r"(?:SCMR|MLD|YLR|CLC)\s+\d{4}\s+[A-Z][A-Za-z]+\s+\d+",
        full_text
    )
    seen_c = set()
    clean_citations = []
    for c in raw_citations:
        c = re.sub(r"\s+", " ", c).strip()
        if c not in seen_c:
            seen_c.add(c)
            clean_citations.append(c)
    meta["citations"] = clean_citations[:15]

    # ── Parties ────────────────────────────────────────────────────────
    parties = []
    versus = re.search(
        r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,4})\s*\.\.\.\s*"
        r"(?:Petitioner|Applicant|Appellant)",
        full_text
    )
    if versus:
        parties.append(versus.group(1).strip())
    else:
        applicants = re.findall(
            r"(?:Applicants?|Petitioner|Accused)\s*[:\-]?\s*((?:[A-Z][a-z]+\s*){1,4})",
            full_text
        )
        parties = list(dict.fromkeys(applicants))[:5]
    meta["parties"] = parties

    # ── Ollama fallback for whatever regex couldn't find ────────────────
    missing = [
        f for f in ("judge", "case_number", "date_of_order", "sections_cited", "parties")
        if not meta.get(f)
    ]
    if missing:
        filled = ollama_fill_missing_metadata(full_text, missing)
        for k, v in filled.items():
            if v:   # only overwrite if Ollama actually returned something
                meta[k] = v

    return meta


# ==========================
# PARAGRAPH SPLITTER
# ==========================
PARA_SPLIT_RE = re.compile(
    r'(?<=[.!?])\s{2,}'
    r'|(?=\n\s*\d+\.\s+[A-Z])'
    r'|(?=\n\s*\(\d+\)\s+[A-Z])'
    r'|\n{2,}',
    re.MULTILINE
)


def split_into_paragraphs(text: str) -> list:
    text = re.sub(r'(?<!\n)\n(?!\n)', ' ', text)
    parts = PARA_SPLIT_RE.split(text)
    return [p.strip() for p in parts if p.strip()]


ORDER_BOUNDARY_RE = re.compile(
    r'^(?:ORDER\s*SHEET|ORDER(?:\s+SHEET)?|O\s*R\s*D\s*E\s*R)(?:\s|$)',
    re.I
)


def is_order_boundary(paragraph_text: str) -> bool:
    return bool(ORDER_BOUNDARY_RE.match(paragraph_text.strip()))


# ==========================
# SENTENCE SPLITTER
# ==========================
SENT_END_RE = re.compile(r'(?<=[.!?])\s+(?=[A-Z])')


def split_sentences(text: str) -> list:
    return [s.strip() for s in SENT_END_RE.split(text) if s.strip()]


def ends_with_sentence_terminator(text: str) -> bool:
    text = text.strip()
    return bool(re.search(r'[.!?]["\']?$|[:;]$', text))


# ==========================
# SMART CHUNKER
# ==========================
def approx_tokens(text: str) -> int:
    return len(text) // 4


def merge_continuation_paragraphs(paragraphs: list[dict]) -> list[dict]:
    merged = []
    for para in paragraphs:
        if (merged
                and not merged[-1]["boundary"]
                and not ends_with_sentence_terminator(merged[-1]["text"])):
            first_char = para["text"].lstrip()[:1]
            if first_char.islower() and not para["boundary"]:
                merged[-1]["text"] += " " + para["text"]
                merged[-1]["page_nums"].extend(para["page_nums"])
                continue
        merged.append(para)
    return merged


def split_page_paragraphs(page_texts: list[tuple[int, str]]) -> list:
    paragraphs = []
    for page_num, page_text in page_texts:
        for part in split_into_paragraphs(page_text):
            paragraphs.append({
                "text":      part,
                "page_nums": [page_num],
                "boundary":  is_order_boundary(part),
            })
    return merge_continuation_paragraphs(paragraphs)


def build_chunks(paragraphs: list) -> list:
    chunks        = []
    current_units = []
    current_tokens = 0
    overlap_tail  = []

    def flush(units, tail_sentences):
        body = " ".join([u["text"] for u in units])
        if tail_sentences:
            body = " ".join(tail_sentences) + " " + body
        chunk_pages = sorted({page for u in units for page in u["page_nums"]})
        chunks.append({
            "chunk_index":    len(chunks),
            "token_estimate": approx_tokens(body),
            "source_pages":   chunk_pages,
            "text":           body,
        })
        all_sentences = split_sentences(body)
        return all_sentences[-OVERLAP_SENTENCES:] if len(all_sentences) >= OVERLAP_SENTENCES else all_sentences

    def split_super_long_sentence(unit):
        words = unit["text"].split()
        parts, current_words, current_chars = [], [], 0
        for word in words:
            current_words.append(word)
            current_chars += len(word) + 1
            if current_chars // 4 >= MAX_CHUNK_TOKENS:
                parts.append({"text": " ".join(current_words), "page_nums": unit["page_nums"], "boundary": False})
                current_words, current_chars = [], 0
        if current_words:
            parts.append({"text": " ".join(current_words), "page_nums": unit["page_nums"], "boundary": False})
        return parts

    sentence_units = []
    for para in paragraphs:
        sentences = split_sentences(para["text"]) or [para["text"]]
        for idx, sent in enumerate(sentences):
            sentence_units.append({
                "text":      sent,
                "page_nums": para["page_nums"],
                "boundary":  para["boundary"] if idx == 0 else False,
            })

    for unit in sentence_units:
        if unit["boundary"] and current_units:
            overlap_tail  = flush(current_units, overlap_tail)
            current_units = []
            current_tokens = 0

        unit_tokens = approx_tokens(unit["text"])

        if unit_tokens > MAX_CHUNK_TOKENS:
            for part in split_super_long_sentence(unit):
                part_tokens = approx_tokens(part["text"])
                if current_tokens + part_tokens > MAX_CHUNK_TOKENS and current_units:
                    overlap_tail  = flush(current_units, overlap_tail)
                    current_units = []
                    current_tokens = 0
                current_units.append(part)
                current_tokens += part_tokens
                if current_tokens >= TARGET_CHUNK_TOKENS:
                    overlap_tail  = flush(current_units, overlap_tail)
                    current_units = []
                    current_tokens = 0
            continue

        if current_tokens + unit_tokens > MAX_CHUNK_TOKENS and current_units:
            overlap_tail  = flush(current_units, overlap_tail)
            current_units = []
            current_tokens = 0

        current_units.append(unit)
        current_tokens += unit_tokens

        if current_tokens >= TARGET_CHUNK_TOKENS:
            overlap_tail  = flush(current_units, overlap_tail)
            current_units = []
            current_tokens = 0

    if current_units:
        flush(current_units, overlap_tail)

    return chunks


# ==========================
# PROCESS NORMAL PDF (no OCR)
# ==========================
def process_normal_pdf(generated_name, pdf_path, row, output_file):
    """
    Normal text extraction (fast). Returns chunk count, or None if PDF is scanned.
    """
    text_parts, page_nums = [], []

    for page_num, page_text in read_pdf_pages(pdf_path):
        clean_page = clean_text(page_text)
        if clean_page:
            text_parts.append(clean_page)
            page_nums.append(page_num)

    full_text = " ".join(text_parts)
    if not full_text.strip():
        return None   # scanned PDF — needs OCR

    doc_meta   = extract_document_metadata(full_text, row)
    page_paras = split_page_paragraphs(list(zip(page_nums, text_parts)))
    chunks     = build_chunks(page_paras)

    output_data = {
        "generated_name":  generated_name,
        "actual_filename": row["actual_filename"],
        "court":           row["court"],
        "case_type":       row["case_type"],
        "year":            row["year"],
        "used_ocr":        False,
        "num_chunks":      len(chunks),
        "case_number":     doc_meta["case_number"],
        "date_of_order":   doc_meta["date_of_order"],
        "judge":           doc_meta["judge"],
        "sections_cited":  doc_meta["sections_cited"],
        "citations":       doc_meta["citations"],
        "parties":         doc_meta["parties"],
        "chunks":          chunks,
    }

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    return len(chunks)


# ==========================
# PROCESS ONE OCR PDF (thread worker)
# ==========================
def _process_ocr_pdf(args: tuple) -> tuple[str, int | None]:
    """
    Thread worker: runs full OCR pipeline on one scanned PDF.
    Returns (generated_name, chunk_count) or (generated_name, None) on failure.
    """
    generated_name, pdf_path, row_dict, output_file = args

    try:
        ocr_pages = ocr_text_pages(pdf_path, max_pages=OCR_MAX_PAGES)
        if not ocr_pages:
            return generated_name, None

        text_parts, page_nums = [], []
        for page_num, page_text in ocr_pages:
            clean_page = clean_text(page_text)
            if clean_page:
                text_parts.append(clean_page)
                page_nums.append(page_num)

        full_text = " ".join(text_parts)
        if not full_text.strip():
            return generated_name, None

        doc_meta   = extract_document_metadata(full_text, row_dict)
        page_paras = split_page_paragraphs(list(zip(page_nums, text_parts)))
        chunks     = build_chunks(page_paras)

        output_data = {
            "generated_name":  generated_name,
            "actual_filename": row_dict["actual_filename"],
            "court":           row_dict["court"],
            "case_type":       row_dict["case_type"],
            "year":            row_dict["year"],
            "used_ocr":        True,
            "num_chunks":      len(chunks),
            "case_number":     doc_meta["case_number"],
            "date_of_order":   doc_meta["date_of_order"],
            "judge":           doc_meta["judge"],
            "sections_cited":  doc_meta["sections_cited"],
            "citations":       doc_meta["citations"],
            "parties":         doc_meta["parties"],
            "chunks":          chunks,
        }

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)

        return generated_name, len(chunks)

    except Exception as e:
        log_error(generated_name, e)
        return generated_name, None


# ==========================
# MAIN PIPELINE
# ==========================
def run():
    df    = pd.read_excel(EXCEL_FILE)
    total = len(df)
    done  = load_checkpoint()

    print(f"\n📂 Total PDFs   : {total}", flush=True)
    print(f"✅ Already done : {len(done)}", flush=True)
    print(f"🔧 OCR engine   : EasyOCR {OCR_LANGUAGES}", flush=True)
    print(f"⚡ OCR threads  : {OCR_WORKERS}", flush=True)
    if not BS4_SUPPORTED:
        print("⚠  bs4 missing: pip install beautifulsoup4 --break-system-packages", flush=True)
    print("🔥 STARTED — Ctrl+C kabhi bhi safely rok sakte hain, agli baar wahi se resume hoga\n", flush=True)

    # Warm up EasyOCR model before threads start
    if OCR_SUPPORTED:
        get_ocr_reader()

    ocr_queue = []   # [(generated_name, pdf_path, row_dict, output_file)]

    # try/finally guarantees checkpoint save no matter WHERE Ctrl+C hits
    try:
        # ── Pass 1: Fast normal extraction ────────────────────────────
        for idx, row in df.iterrows():
            generated_name = str(row["generated_name"])
            pdf_path       = str(row["actual_path"])
            safe_name      = generated_name.replace(".pdf", "")
            output_file    = os.path.join(OUTPUT_DIR, f"{safe_name}.json")

            if generated_name in done:
                continue

            if os.path.exists(output_file):
                print(f"  ⚠ Exists — skip: {os.path.basename(output_file)}", flush=True)
                done.add(generated_name)
                save_checkpoint(done)
                print_progress(len(done), total)
                continue

            if not os.path.exists(pdf_path):
                print(f"[{idx+1}/{total}] ⚠  File missing: {generated_name}", flush=True)
                done.add(generated_name)
                save_checkpoint(done)
                print_progress(len(done), total)
                continue

            try:
                result = process_normal_pdf(
                    generated_name, pdf_path, row.to_dict(), output_file
                )

                if result is None:
                    print(f"[{idx+1}/{total}] 🔄 Queued for OCR : {generated_name}", flush=True)
                    ocr_queue.append((
                        generated_name,
                        pdf_path,
                        row.to_dict(),
                        output_file,
                    ))
                else:
                    print(f"[{idx+1}/{total}] ✅ {result} chunks → {os.path.basename(output_file)}", flush=True)
                    done.add(generated_name)
                    save_checkpoint(done)
                    print_progress(len(done), total)

            except Exception as e:
                print(f"[{idx+1}/{total}] ❌ Error: {e}", flush=True)
                log_error(generated_name, e)

            finally:
                gc.collect()

            if stop_requested():
                print("\n⏸ Stop file detected.", flush=True)
                clear_stop_flag()
                return

            if _INTERRUPTED:
                print(f"\n⏸ Ctrl+C confirmed — Pass 1 rok raha hoon "
                      f"({len(done)}/{total} done, {total - len(done)} remaining).", flush=True)
                return

        # ── Pass 2: Parallel OCR (ThreadPoolExecutor, shared EasyOCR model) ──
        if ocr_queue:
            print(f"\n🧠 OCR pass: {len(ocr_queue)} PDFs | "
                  f"{OCR_WORKERS} threads | EasyOCR {OCR_LANGUAGES}\n", flush=True)

            executor = ThreadPoolExecutor(max_workers=OCR_WORKERS)
            try:
                future_to_name = {
                    executor.submit(_process_ocr_pdf, args): args[0]
                    for args in ocr_queue
                }

                for future in as_completed(future_to_name):
                    name = future_to_name[future]
                    try:
                        gen_name, n_chunks = future.result()
                        if n_chunks is not None:
                            print(f"  ✅ OCR done : {gen_name} → {n_chunks} chunks", flush=True)
                        else:
                            print(f"  ❌ OCR failed (no text): {gen_name}", flush=True)
                        done.add(gen_name)
                        save_checkpoint(done)
                        print_progress(len(done), total)

                    except Exception as e:
                        print(f"  ❌ OCR error: {name} → {e}", flush=True)
                        log_error(name, e)

                    if stop_requested():
                        print("\n⏸ Stop file detected during OCR pass.", flush=True)
                        clear_stop_flag()
                        # cancel any not-yet-started OCR jobs and exit fast
                        executor.shutdown(wait=False, cancel_futures=True)
                        return

                    if _INTERRUPTED:
                        print(f"\n⏸ Ctrl+C confirmed — abhi chal rahi OCR file(s) "
                              f"khatam hote hi rukega ({len(done)}/{total} done, "
                              f"{total - len(done)} remaining). Baaki queued files "
                              f"cancel ho rahi hain...", flush=True)
                        # cancel jobs that haven't started yet; jobs already
                        # running in a thread will finish (native OCR calls
                        # can't be killed mid-flight) but nothing NEW starts
                        executor.shutdown(wait=False, cancel_futures=True)
                        return
            finally:
                # normal completion path — just wait for the pool to close
                executor.shutdown(wait=True)

        print("\n🎉 Done. Re-run anytime to resume from checkpoint.", flush=True)

    except KeyboardInterrupt:
        # Ctrl+C — stop IMMEDIATELY, don't wait for the rest of the queue
        print("\n⏸ Ctrl+C dabaya gaya — turant ruk raha hai.", flush=True)
        try:
            # if we were inside the OCR pass, kill pending threads right away
            executor.shutdown(wait=False, cancel_futures=True)
        except NameError:
            pass  # Pass 1 tha, koi executor nahi bana

    finally:
        save_checkpoint(done)
        print(f"💾 Checkpoint saved: {len(done)}/{total} done. "
              f"Script dobara chalao to isi jagah se resume hoga.", flush=True)


if __name__ == "__main__":
    run()
    