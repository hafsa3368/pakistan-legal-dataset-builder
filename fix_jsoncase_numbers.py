"""
fix_case_numbers.py
====================
One-off targeted fix for the case_number bug in extract_document_metadata():
it used to re.search() the FULL document body, so on some files it matched a
citation to a DIFFERENT (precedent/co-accused) case mentioned in the body
text instead of the document's own case number in the header. That caused
many unrelated documents to end up with the exact same case_number.

REVISION 15 (this revision) -- LIKELY FINAL PRACTICAL ROUND
-----------------------------------------------------------------------------
unresolved_clean_header.json is now down to 136 files (from 4330 files back
at Revision 6). Reviewing this batch, ~90% are genuinely garbled OCR text
that no regex can recover ("IN THI HICH COURTOI SINDH", "ORUER SHFFT",
Arabic-comma-for-period substitutions like "C،P No.D-1275", etc.) -- these
belong with the OCR-garbled bucket in spirit, they simply didn't trip the
garble heuristic. Only two genuinely fixable items were found:

    'I.T.R. A. No.327 of 2019'              <- Income Tax Reference Appeal,
                                                 never covered
    'Cr. Bail Appln: No.S-1249 of 2022'     <- colon instead of period after
                                                 "Appln" -- the existing
                                                 "Cr. Bail App*" pattern only
                                                 accepted a literal "." here,
                                                 not ":"

Fix: added I.T.R.A. as a new prefix, and widened the punctuation accepted
after "Cr. Bail App[a-z]*" from "\\.?" to "[.:]?" so both period and colon
work.

RECOMMENDATION: after this revision, further regex iteration on the
remaining clean-header bucket has sharply diminishing returns -- most of
what's left is OCR noise, not missing patterns. Treat any files still in
unresolved_clean_header.json / unresolved_ocr_garbled.json /
unresolved_no_content_extracted.json after this round as needing either
manual review or a fresh OCR pass on the source PDFs, not more regex work.

See prior revisions (kept in file history) for the full list of previously
added prefixes and fixes.

REQUIRES: nothing from extractor.py (fully self-contained), except
          OUTPUT_DIR path convention, which is kept identical below.

USAGE:
    python fix_jsoncase_numbers.py                    # dry run, regex only (fast)
    python fix_jsoncase_numbers.py --use-ollama        # dry run, with Ollama fallback for header-miss cases
    python fix_jsoncase_numbers.py --apply             # actually write changes
    python fix_jsoncase_numbers.py --apply --use-ollama
"""

import os
import re
import sys
import json
import glob
import time
import signal
import argparse

OUTPUT_DIR              = "extracted_text_clean"
CHECKPOINT_FILE         = "fix_case_numbers_checkpoint.json"
REPAIR_CHECKPOINT_FILE  = "repair_checkpoint.json"
REVIEW_FILE             = "fix_case_numbers_needs_review.json"
CHANGED_AUDIT_FILE      = "fix_case_numbers_changed_audit.json"

SLOW_FILE_WARN_SECONDS  = 2.0   # print a warning if a single file takes longer than this

# Ollama config (only used if --use-ollama is passed)
OLLAMA_ENABLED_DEFAULT = False
OLLAMA_URL             = "http://localhost:11434/api/generate"
OLLAMA_TAGS_URL        = "http://localhost:11434/api/tags"
OLLAMA_MODEL           = "llama3.2:3b"
OLLAMA_TIMEOUT         = 15     # seconds -- kept short on purpose; this is a bulk audit pass,
                                 # not a one-off extraction, so we don't want 60s stalls x1000s of files


# ==========================
# CTRL+C HANDLING
# ==========================
_INTERRUPTED = False

def _handle_sigint(signum, frame):
    global _INTERRUPTED
    if not _INTERRUPTED:
        print("\n⏸ Ctrl+C received — stopping after current file, saving checkpoint...", flush=True)
    _INTERRUPTED = True

signal.signal(signal.SIGINT, _handle_sigint)


# ==========================
# NORMALIZED VERBATIM CHECK
# ==========================
def _normalize_for_compare(s: str) -> str:
    """Lowercase, normalize dash characters to a plain hyphen, and strip
    ALL whitespace. Used ONLY for the "is this value present in the header"
    comparison -- never used to overwrite/store a value."""
    s = str(s or "")
    s = s.lower()
    s = s.replace("\u2013", "-").replace("\u2014", "-")  # en-dash, em-dash -> hyphen
    s = re.sub(r"\s+", "", s)
    return s


def is_verbatim_in_header(value: str, header_text: str) -> bool:
    """True if `value` occurs in `header_text` once both are normalized
    (see _normalize_for_compare)."""
    if not value:
        return False
    return _normalize_for_compare(value) in _normalize_for_compare(header_text)


# ==========================
# CASE NUMBER EXTRACTION ONLY
# (header-only regex)
# ==========================
# Number-token tail: tolerates a letter+dash prefix with spaces around the
# dash (e.g. "D – 1115"), not just contiguous "D-1115".
_NUMBER_TAIL = r"(?:[A-Za-z]{1,3}\s*[-\u2013]\s*)?\d[\w\-/]*(?:\s+of\s+\d{4})?"

# Middle-gap character class includes parentheses, so captions with a
# parenthetical between the prefix and "No." (e.g. "C.O.S (Insurance) No.")
# can match.
_MIDDLE_GAP = r"(?:[\w\s\.\-\(\)]{0,40}?)"

# REVISION 15 -- added I.T.R.A. (Income Tax Reference Appeal). Widened the
# punctuation after "Cr. Bail App[a-z]*" from period-only to period-or-colon
# ("Cr. Bail Appln: No." was previously missed).
_CASE_NO_STANDARD_RE = re.compile(
    r"("
    r"(?:Crl\.?\s*|CRL\.?\s*|Cr\.B\.A\.?\s*|Cr\.R\.A\.?\s*|Cr\.A\.?\s*|"
    r"Cr\.?\s*Appeal\s*|"
    r"Cr\.?\s*Bail\s*App[a-z]*[.:]?\s*|"
    r"Cr\.?\s*Transfer\s*App[a-z]*\.?\s*|"
    r"Civil\s*|W\.?\s*P\.?\s*|Writ\s+Petition\s*|Const\.?\s*|Criminal\s*|Misc\.?\s*|"
    r"Election\s+Appeal\s*|"
    r"P\.?\s*S\.?\s*L\.?\s*A\.?\s*|"
    r"C\.?\s*P\.?\s*|"
    r"Constt:?\s*Petition\.?\s*|"
    r"R\.?\s*F\.?\s*A\.?s?\.?\s*|"
    r"C\.?\s*R\.?\s*|"
    r"I\.?\s*C\.?\s*A\.?\s*|"
    r"Service\s+Appeal\s*|"
    r"Regular\s+First\s+Appeal\s*|"
    r"Election\s+Petition\s*|"
    r"Suo\s+Motu\s+Case\s*|"
    r"Review\s+Application\s*|"
    r"Execution\s+First\s+Appeal\s*|"
    r"Insurance\s+Appeal\s*|"
    r"Diary\.?\s*|"
    r"C\.?\s*O\.?\s*S\.?\s*|"
    r"P\.?\s*L\.?\s*A\.?\s*|"
    r"I-?\s*Appeal\s*|"
    r"H\.?\s*C\.?\s*A\.?\s*|"
    r"High\s+Court\s+Appeal\s*|"
    r"M\.?\s*A\.?\s*|"
    r"T\.?\s*A\.?\s*|"
    r"E\.?\s*A\.?\s*|"
    r"I\.?\s*T\.?\s*R\.?\s*A\.?\s*"
    r")"
    + _MIDDLE_GAP +
    r"Nos?\.?\s*" + _NUMBER_TAIL +
    r")",
    re.I
)

# Short abbreviations: NO middle-gap wildcard exists in this branch, so the
# number (or "No." + number) must appear immediately after the abbreviation
# (only whitespace/dots allowed in between). That adjacency requirement
# alone prevents false matches inside unrelated words like "COURT" -- no
# lookahead guard needed.
_CASE_NO_SHORT_RE = re.compile(
    r"("
    r"(?:"
    r"\bE\.?\s*F\.?\s*A\.?\s*|"
    r"\bR\.?\s*S\.?\s*A\.?\s*|"
    r"\bC\.?\s*O\.?\s*|"
    r"\bF\.?\s*A\.?\s*O\.?\s*"
    r")"
    r"(?:Nos?\.?\s*)?" + _NUMBER_TAIL +
    r")",
    re.I
)

HEADER_WINDOW = 3000

_LOWCONF_WORDS = re.compile(
    r"\b(?:cannot|could|does|was|not|under|dispute|consent|forums|"
    r"notification|jurisdiction|matters|private|respondents|crime|panoply|"
    r"revenue|another|laws|along|bearing\s+private|does\s+not)\b",
    re.I
)

def match_confidence(matched_text: str) -> str:
    """Returns 'high' or 'low'. 'low' means this looks like it may have
    drifted into unrelated prose rather than a genuine case-number caption."""
    if len(matched_text) > 55:
        return "low"
    if _LOWCONF_WORDS.search(matched_text):
        return "low"
    return "high"


def _is_well_formed_case_number(candidate: str) -> bool:
    """True if `candidate`, taken on its own, is a complete supported case
    number, including short abbreviations where "No." is optional."""
    if not candidate:
        return False
    candidate = candidate.strip()
    return bool(
        _CASE_NO_STANDARD_RE.fullmatch(candidate)
        or _CASE_NO_SHORT_RE.fullmatch(candidate)
    )


_OLLAMA_ALIVE = None  # cached after first check

def _is_ollama_alive() -> bool:
    global _OLLAMA_ALIVE
    if _OLLAMA_ALIVE is not None:
        return _OLLAMA_ALIVE
    try:
        import requests
        resp = requests.get(OLLAMA_TAGS_URL, timeout=3)
        _OLLAMA_ALIVE = resp.status_code == 200
    except Exception:
        _OLLAMA_ALIVE = False
    if not _OLLAMA_ALIVE:
        print("⚠  Ollama not reachable — case_number fallback will be skipped for this run.", flush=True)
    return _OLLAMA_ALIVE


def _ollama_case_number(header_text: str) -> str:
    """Ask Ollama for ONLY case_number, scoped to header_text only."""
    if not _is_ollama_alive():
        return ""
    try:
        import requests
        prompt = (
            "You are extracting structured metadata from a Pakistani court judgment. "
            "Return ONLY a valid JSON object, no preamble, no markdown fences. "
            "Extract this field:\n"
            '- "case_number": the case number exactly as it appears in the document '
            "header (format varies -- e.g. prefix + 'No.' + digits, optionally "
            "'of' + a year, such as [TYPE].No.[NUMBER] of [YEAR]).\n"
            "Do NOT invent, guess, or reuse any example format shown here -- only "
            "return a case number if it is literally present in the text below. "
            "Return the COMPLETE case number including its prefix (e.g. 'Cr.B.A.' "
            "or 'W.P.') and the 'No.' token -- never just the trailing digits/year. "
            "If no case number can be found in the text, return an empty string for "
            "this field. Never output a value that does not appear verbatim in the "
            "document text.\n\n"
            f"Document text:\n{header_text}"
        )
        resp = requests.post(
            OLLAMA_URL,
            json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False, "format": "json"},
            timeout=OLLAMA_TIMEOUT,
        )
        resp.raise_for_status()
        raw = resp.json().get("response", "").strip()
        raw = re.sub(r"^```json\s*|\s*```$", "", raw.strip())
        parsed = json.loads(raw)
        candidate = (parsed.get("case_number") or "").strip()
        if not candidate or not is_verbatim_in_header(candidate, header_text):
            return ""
        if not _is_well_formed_case_number(candidate):
            return ""
        return candidate
    except Exception:
        pass
    return ""


def extract_case_number(header_text: str, use_ollama: bool) -> str:
    m = _CASE_NO_STANDARD_RE.search(header_text)
    if m:
        return m.group(1).strip()

    m = _CASE_NO_SHORT_RE.search(header_text)
    if m:
        return m.group(1).strip()

    if use_ollama:
        return _ollama_case_number(header_text)
    return ""


# ==========================
# CHECKPOINT HELPERS
# ==========================
def load_repaired_filenames():
    if not os.path.exists(REPAIR_CHECKPOINT_FILE):
        raise FileNotFoundError(
            f"{REPAIR_CHECKPOINT_FILE} not found in this folder. "
            f"Run repair_metadata.py first, or update REPAIR_CHECKPOINT_FILE path."
        )
    with open(REPAIR_CHECKPOINT_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    names = set()
    if isinstance(data, list):
        names.update(data)
    elif isinstance(data, dict):
        for key in ("completed", "fixed", "processed", "done", "repaired"):
            value = data.get(key)
            if isinstance(value, list):
                names.update(value)
            elif isinstance(value, dict):
                names.update(value.keys())

    if not names:
        raise ValueError(f"{REPAIR_CHECKPOINT_FILE} loaded but no filenames found in it.")

    return {os.path.basename(n) for n in names}


def load_checkpoint() -> set:
    if not os.path.exists(CHECKPOINT_FILE):
        return set()
    try:
        with open(CHECKPOINT_FILE, "r", encoding="utf-8") as f:
            content = f.read()
        if not content.strip():
            print(f"⚠  {CHECKPOINT_FILE} exists but is empty — treating as no "
                  f"progress yet (this run will re-scan all files). The empty "
                  f"file has been preserved as {CHECKPOINT_FILE}.corrupt for "
                  f"reference.", flush=True)
            try:
                os.replace(CHECKPOINT_FILE, CHECKPOINT_FILE + ".corrupt")
            except Exception:
                pass
            return set()
        return set(json.loads(content))
    except json.JSONDecodeError as e:
        print(f"⚠  {CHECKPOINT_FILE} is corrupted ({e}) — backing it up as "
              f"{CHECKPOINT_FILE}.corrupt and starting with no progress "
              f"recorded (this run will re-scan all files; already-fixed "
              f"files will simply be re-verified, which is safe since this "
              f"script recomputes case_number deterministically).", flush=True)
        try:
            os.replace(CHECKPOINT_FILE, CHECKPOINT_FILE + ".corrupt")
        except Exception:
            pass
        return set()


def save_checkpoint(done: set):
    tmp_path = CHECKPOINT_FILE + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(sorted(done), f, ensure_ascii=False, indent=2)
        f.flush()
        os.fsync(f.fileno())

    max_attempts = 5
    for attempt in range(1, max_attempts + 1):
        try:
            os.replace(tmp_path, CHECKPOINT_FILE)
            return
        except PermissionError:
            if attempt == max_attempts:
                print(f"⚠  Could not save checkpoint after {max_attempts} attempts "
                      f"(file locked, likely by OneDrive sync or antivirus). "
                      f"Continuing run without saving this checkpoint update -- "
                      f"the next successful save will catch up. If this file lives "
                      f"in a OneDrive/cloud-synced folder, consider pausing sync "
                      f"while this script runs.", flush=True)
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass
                return
            time.sleep(0.5 * attempt)  # 0.5s, 1s, 1.5s, 2s backoff


def reconstruct_full_text(data: dict) -> str:
    chunks = sorted(data.get("chunks", []), key=lambda c: c.get("chunk_index", 0))
    return " ".join(c.get("text", "") for c in chunks)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true",
                         help="Actually write changes. Without this, dry-run preview only.")
    parser.add_argument("--use-ollama", action="store_true", default=OLLAMA_ENABLED_DEFAULT,
                         help="Fall back to Ollama (scoped to header only) when regex finds nothing. "
                              "OFF by default — this pass is meant to be fast + deterministic.")
    args = parser.parse_args()

    repaired_names = load_repaired_filenames()
    all_files = [f for f in glob.glob(os.path.join(OUTPUT_DIR, "*.json")) if os.path.basename(f) in repaired_names]

    done = load_checkpoint() if args.apply else set()
    pending = [f for f in all_files if os.path.basename(f) not in done]

    print(f"📋 Repaired filenames in checkpoint: {len(repaired_names)}")
    print(f"📂 Matched on disk: {len(all_files)} | pending: {len(pending)}")
    print(f"Mode: {'APPLY (writing changes)' if args.apply else 'DRY RUN (preview only)'}")
    print(f"Ollama fallback: {'ON' if args.use_ollama else 'OFF (regex-only, fast)'}\n")

    changed = 0
    flagged = 0
    needs_review = []
    changed_audit = []
    t_start = time.time()
    i = 0

    for i, path in enumerate(pending, 1):
        if _INTERRUPTED:
            break

        t0 = time.time()

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        full_text = reconstruct_full_text(data)
        if not full_text.strip():
            if args.apply:
                done.add(os.path.basename(path))
                save_checkpoint(done)
            continue

        header_text = full_text[:HEADER_WINDOW]
        old_case_no_raw = data.get("case_number", "")

        if isinstance(old_case_no_raw, str):
            old_case_no = old_case_no_raw.strip()
            stored_is_in_own_header = is_verbatim_in_header(old_case_no, header_text)
        else:
            old_case_no = old_case_no_raw
            stored_is_in_own_header = False

        if stored_is_in_own_header:
            if args.apply:
                done.add(os.path.basename(path))
                save_checkpoint(done)
            elapsed = time.time() - t0
            if elapsed > SLOW_FILE_WARN_SECONDS:
                print(f"    ⏱ Slow file ({elapsed:.1f}s): {os.path.basename(path)}", flush=True)
            continue

        new_case_no = extract_case_number(header_text, use_ollama=args.use_ollama)

        if new_case_no:
            confidence = match_confidence(new_case_no)
            candidate_is_in_header = is_verbatim_in_header(new_case_no, header_text)

            if confidence == "high" and candidate_is_in_header:
                changed += 1
                print(
                    f"[{i}/{len(pending)}] {os.path.basename(path)}: "
                    f"{old_case_no!r} -> {new_case_no!r} "
                    f"(stored value not verified in own header)",
                    flush=True,
                )
                changed_audit.append({
                    "file": os.path.basename(path),
                    "old_case_number": old_case_no,
                    "new_case_number": new_case_no,
                })
                if args.apply:
                    data["case_number"] = new_case_no
                    with open(path, "w", encoding="utf-8") as f:
                        json.dump(data, f, ensure_ascii=False, indent=2)
            else:
                flagged += 1
                needs_review.append({
                    "file": os.path.basename(path),
                    "old_case_number": old_case_no,
                    "candidate_case_number": new_case_no,
                    "reason": (
                        "contaminated_stored_value_low_confidence_candidate"
                        if confidence != "high"
                        else "candidate_not_verbatim_in_header"
                    ),
                })
                print(
                    f"[{i}/{len(pending)}] ⚠ NEEDS REVIEW {os.path.basename(path)}: "
                    f"stored {old_case_no!r} not verified in own header; "
                    f"candidate {new_case_no!r} not applied",
                    flush=True,
                )
        else:
            flagged += 1
            needs_review.append({
                "file": os.path.basename(path),
                "old_case_number": old_case_no,
                "candidate_case_number": "",
                "reason": "contaminated_stored_value_not_found_in_own_header_no_replacement",
            })
            print(
                f"[{i}/{len(pending)}] 🛑 CONTAMINATED {os.path.basename(path)}: "
                f"stored {old_case_no!r} not verified in own header; "
                f"no replacement found",
                flush=True,
            )

        if args.apply:
            done.add(os.path.basename(path))
            save_checkpoint(done)

        elapsed = time.time() - t0
        if elapsed > SLOW_FILE_WARN_SECONDS:
            print(f"    ⏱ Slow file ({elapsed:.1f}s): {os.path.basename(path)}", flush=True)

        if i % 500 == 0:
            rate = i / (time.time() - t_start)
            print(f"    📊 Progress: {i}/{len(pending)} scanned | {changed} changed | "
                  f"{flagged} flagged | {rate:.1f} files/sec\n", flush=True)

    if needs_review:
        with open(REVIEW_FILE, "w", encoding="utf-8") as f:
            json.dump(needs_review, f, ensure_ascii=False, indent=2)

    if changed_audit:
        with open(CHANGED_AUDIT_FILE, "w", encoding="utf-8") as f:
            json.dump(changed_audit, f, ensure_ascii=False, indent=2)

    if _INTERRUPTED:
        print(f"\n⏸ Stopped early by Ctrl+C. Scanned {i}/{len(pending)} in this run.")
        if args.apply:
            print(f"💾 Checkpoint saved: {len(done)} files done total. Re-run to resume.")
        else:
            print("👉 Dry run does not checkpoint progress — re-run will rescan from the start.")
    else:
        print(f"\n✅ Done. {changed} file(s) had case_number corrected.")
        if flagged:
            print(f"⚠  {flagged} file(s) were flagged because the stored case_number "
                  f"could not be verified against the file header, or no safe replacement "
                  f"was found — see {REVIEW_FILE} to review manually.")
        if not args.apply:
            print("👉 This was a DRY RUN. Re-run with --apply to write changes.")


if __name__ == "__main__":
    main()