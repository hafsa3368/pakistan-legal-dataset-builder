"""
fix_case_numbers.py
====================
One-off targeted fix for the case_number bug in extract_document_metadata():
it used to re.search() the FULL document body, so on some files it matched a
citation to a DIFFERENT (precedent/co-accused) case mentioned in the body
text instead of the document's own case number in the header. That caused
many unrelated documents to end up with the exact same case_number.

This script:
  - Does NOT touch chunks, judge, parties, sections_cited, citations, dates
  - Reconstructs full_text from each file's own stored chunks (no PDF/OCR)
  - Recomputes ONLY case_number -- using a lean, header-only regex (mirrors
    the v9.1 fix in extractor.py), with an OPTIONAL Ollama fallback scoped
    ONLY to case_number (mirrors the v9.2 fix), same safety guard: Ollama's
    answer is only accepted if it's a verbatim substring of header_text.
  - Overwrites case_number UNCONDITIONALLY when it differs from the fresh
    value -- unlike repair_metadata.py, this does NOT check "is it already
    valid" first, because the old ghost values are structurally well-formed
    (they pass a format check) even though they're factually wrong.
  - Dry run by default; nothing is written unless --apply is passed.

WHY THIS VERSION IS DIFFERENT FROM THE ORIGINAL
-------------------------------------------------
The original script called the FULL extract_document_metadata(), which also
recomputes judge / date_of_order / sections_cited / parties -- fields this
script has no business touching. Whenever regex missed any of THOSE fields
(very common), the old code fired an Ollama network call (up to 60s timeout
EACH) purely to fill fields that were going to be thrown away anyway. Across
10,000+ files that's what made the run crawl and made Ctrl+C feel dead (it
was stuck inside a blocking network call with no interrupt handling at all).

This version:
  - Only ever computes case_number. Nothing else.
  - Ollama is OFF by default here (--use-ollama to turn it on) because for
    a pure "fix the regex bug" pass, the deterministic header regex is
    exactly what you want to audit -- Ollama adds latency and (small but
    nonzero) nondeterminism on top of a fix that's supposed to be exact.
  - Has a real Ctrl+C handler: sets a flag checked every iteration, saves
    checkpoint immediately, and exits -- no waiting on network timeouts.
  - Prints elapsed time per file over a threshold so slow files are visible
    instead of the run just silently sitting there.

REQUIRES: nothing from extractor.py anymore (fully self-contained), except
          OUTPUT_DIR path convention, which is kept identical below.

USAGE:
    python fix_case_numbers.py                    # dry run, regex only (fast)
    python fix_case_numbers.py --use-ollama        # dry run, with Ollama fallback for header-miss cases
    python fix_case_numbers.py --apply             # actually write changes
    python fix_case_numbers.py --apply --use-ollama
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

SLOW_FILE_WARN_SECONDS  = 2.0   # print a warning if a single file takes longer than this

# Ollama config (only used if --use-ollama is passed)
OLLAMA_ENABLED_DEFAULT = False
OLLAMA_URL             = "http://localhost:11434/api/generate"
OLLAMA_TAGS_URL        = "http://localhost:11434/api/tags"
OLLAMA_MODEL           = "llama3.2:3b"
OLLAMA_TIMEOUT         = 15     # seconds -- kept short on purpose; this is a bulk audit pass,
                                 # not a one-off extraction, so we don't want 60s stalls x1000s of files


# ==========================
# CTRL+C HANDLING (was completely missing before)
# ==========================
_INTERRUPTED = False

def _handle_sigint(signum, frame):
    global _INTERRUPTED
    if not _INTERRUPTED:
        print("\n⏸ Ctrl+C received — stopping after current file, saving checkpoint...", flush=True)
    _INTERRUPTED = True

signal.signal(signal.SIGINT, _handle_sigint)


# ==========================
# CASE NUMBER EXTRACTION ONLY
# (header-only regex, same pattern as extractor.py v9.1 fix)
# ==========================
CASE_NO_RE = re.compile(
    r"("
    r"(?:Crl\.?\s*|CRL\.?\s*|Cr\.B\.A\.?\s*|Cr\.R\.A\.?\s*|Cr\.A\.?\s*|"
    r"Civil\s*|W\.P\.?\s*|Const\.?\s*|Criminal\s*|Misc\.?\s*)"
    r"(?:[\w\s\.\-]{0,40}?)"
    # BUG FIX: the token after "No." must actually contain a digit somewhere.
    # Without this, a blank/illegible case-number field on the source page
    # (common in scanned orders) let this regex grab the next plain word
    # instead (e.g. "No. IN" from "...No. ___ IN THE COURT OF...").
    r"No\.?\s*(?=[\w\-]*\d)[\w\-]+(?:/[\w\-]+)*"
    r"(?:\s+of\s+\d{4})?"
    r")",
    re.I
)

# BUG FIX: a hard 1200-char cutoff could slice a real case number in half
# right at the boundary, or (subtler) simply not include the optional
# " of 2023" suffix because that text happened to fall just past the cut --
# the regex still matched successfully without it, so no truncation was
# ever detected. Using one generously sized window instead of a tight one
# avoids this while still staying scoped to the document header (captions,
# parties, court name) rather than the full body where precedent/co-accused
# case numbers get cited -- that's what the original bug was about.
HEADER_WINDOW = 3000

# ── Confidence check ──────────────────────────────────────────────────
# The regex's middle gap (up to 40 chars between the prefix and "No.") is
# loose enough to sometimes drift across a whole sentence and latch onto an
# unrelated "No." mention -- a notification number, a "Crime No.", a suit
# number inside a narrative sentence, etc. Genuine captions are short and
# mostly capitalized abbreviations ("W.P. No.24269 of 2019"); prose drift
# tends to be longer and full of ordinary lowercase sentence words. Rather
# than trust every regex match equally, flag the prose-like ones as low
# confidence so they can be reviewed by a person instead of silently
# overwriting a (possibly correct) old value with a wrong one.
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
    """Ask Ollama for ONLY case_number, scoped to header_text only. Verified via substring check."""
    if not _is_ollama_alive():
        return ""
    try:
        import requests
        prompt = (
            "You are extracting structured metadata from a Pakistani court judgment. "
            "Return ONLY a valid JSON object, no preamble, no markdown fences. "
            "Extract this field:\n"
            '- "case_number": the case number (e.g. \'Cr.B.A.No.S-994 of 2019\')\n'
            "If it cannot be found, use an empty string.\n\n"
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
        # Safety guard: only accept if it's actually present in the header text.
        if candidate and candidate.lower() in header_text.lower():
            return candidate
    except Exception:
        pass
    return ""


def extract_case_number(full_text: str, use_ollama: bool) -> str:
    header_text = full_text[:HEADER_WINDOW]
    m = CASE_NO_RE.search(header_text)
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
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, "r", encoding="utf-8") as f:
            return set(json.load(f))
    return set()


def save_checkpoint(done: set):
    with open(CHECKPOINT_FILE, "w", encoding="utf-8") as f:
        json.dump(sorted(done), f, ensure_ascii=False, indent=2)


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
    t_start = time.time()

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

        new_case_no = extract_case_number(full_text, use_ollama=args.use_ollama)
        old_case_no = data.get("case_number", "")

        if new_case_no and new_case_no != old_case_no:
            confidence = match_confidence(new_case_no)

            if confidence == "high":
                changed += 1
                print(f"[{i}/{len(pending)}] {os.path.basename(path)}: {old_case_no!r} -> {new_case_no!r}", flush=True)
                if args.apply:
                    data["case_number"] = new_case_no
                    with open(path, "w", encoding="utf-8") as f:
                        json.dump(data, f, ensure_ascii=False, indent=2)
            else:
                # LOW CONFIDENCE: looks like it may have drifted into prose
                # rather than a real caption. Never auto-applied, even with
                # --apply -- logged for manual review instead.
                flagged += 1
                needs_review.append({
                    "file": os.path.basename(path),
                    "old_case_number": old_case_no,
                    "candidate_case_number": new_case_no,
                })
                print(f"[{i}/{len(pending)}] ⚠ NEEDS REVIEW {os.path.basename(path)}: "
                      f"{old_case_no!r} -> {new_case_no!r} (not applied)", flush=True)

        if args.apply:
            done.add(os.path.basename(path))
            save_checkpoint(done)

        elapsed = time.time() - t0
        if elapsed > SLOW_FILE_WARN_SECONDS:
            print(f"    ⏱ Slow file ({elapsed:.1f}s): {os.path.basename(path)}", flush=True)

        if i % 500 == 0:
            rate = i / (time.time() - t_start)
            print(f"    📊 Progress: {i}/{len(pending)} scanned | {changed} changed | "
                  f"{flagged} flagged for review | {rate:.1f} files/sec\n", flush=True)

    if needs_review:
        with open(REVIEW_FILE, "w", encoding="utf-8") as f:
            json.dump(needs_review, f, ensure_ascii=False, indent=2)

    if _INTERRUPTED:
        print(f"\n⏸ Stopped early by Ctrl+C. Scanned {i}/{len(pending)} in this run.")
        if args.apply:
            print(f"💾 Checkpoint saved: {len(done)} files done total. Re-run to resume.")
        else:
            print("👉 Dry run does not checkpoint progress — re-run will rescan from the start.")
    else:
        print(f"\n✅ Done. {changed} file(s) had case_number corrected.")
        if flagged:
            print(f"⚠  {flagged} file(s) had a candidate case_number that looked prose-like "
                  f"and were NOT applied — see {REVIEW_FILE} to review manually.")
        if not args.apply:
            print("👉 This was a DRY RUN. Re-run with --apply to write changes.")


if __name__ == "__main__":
    main()