"""
fix_judge_names_json.py
=========================
Fast, regex-only repair of the "judge" field in the already-extracted JSON
files in extracted_text_clean/ -- same speed philosophy as
fix_jsoncase_numbers.py (Ollama OFF by default, so this is a deterministic,
fast pass, not the slow Ollama-per-file path repair_metadata.py takes).

Does NOT open any PDF, does NOT run OCR, does NOT call Ollama unless you
pass --use-ollama. Reconstructs each file's full_text from its own
"chunks" (already extracted) and re-runs the SAME judge-name regex cascade
extractor.py uses, self-contained here so this script has no import
dependency on extractor.py.

Only replaces "judge" when the CURRENT value fails validation AND a valid
candidate can be found in the reconstructed text. Every other field is
left untouched. Scoped to the ~10.5k files listed in repair_checkpoint.json
(the same set neo4j_import.py trusts) -- not the ~58k raw files sitting in
extracted_text_clean/, most of which are leftover/duplicate extraction runs.

USAGE:
    python fix_judge_names_json.py                 # dry run, regex only (fast)
    python fix_judge_names_json.py --use-ollama     # dry run, with Ollama fallback for regex-miss cases
    python fix_judge_names_json.py --apply          # actually write changes
    python fix_judge_names_json.py --apply --use-ollama
"""

import os
import re
import sys
import json
import glob
import time
import signal
import argparse

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

OUTPUT_DIR = "extracted_text_clean"
REPAIR_CHECKPOINT_FILE = "repair_checkpoint.json"   # read-only scope source
CHECKPOINT_FILE = "fix_judge_names_json_checkpoint.json"
REVIEW_FILE = "fix_judge_names_json_needs_review.json"
CHANGED_AUDIT_FILE = "fix_judge_names_json_changed_audit.json"

SLOW_FILE_WARN_SECONDS = 2.0

OLLAMA_ENABLED_DEFAULT = False
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_TAGS_URL = "http://localhost:11434/api/tags"
OLLAMA_MODEL = "llama3.2:3b"
OLLAMA_TIMEOUT = 15

MAX_TEXT_CHARS = 200_000


# ==========================================================
# CTRL+C HANDLING
# ==========================================================
_INTERRUPTED = False


def _handle_sigint(signum, frame):
    global _INTERRUPTED
    if not _INTERRUPTED:
        print("\n⏸ Ctrl+C received -- stopping after current file, saving checkpoint...", flush=True)
    _INTERRUPTED = True


signal.signal(signal.SIGINT, _handle_sigint)


# ==========================================================
# JUDGE-NAME EXTRACTION (regex cascade, copied from extractor.py so this
# script is self-contained -- see extractor.py's extract_document_metadata()
# "Judge name" section for the source of truth these mirror)
# ==========================================================
def extract_judge_from_text(full_text: str) -> str:
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
    return judge.group(1).strip() if judge else ""


# ==========================================================
# VALIDATION -- same rules as repair_metadata.py's is_valid_judge(), kept
# self-contained here rather than imported.
# ==========================================================
BAD_JUDGE_TOKENS = {
    "PA", "PS", "APG", "DPG", "ASC", "ORDER", "DATE", "JUDGE", "PRESENT",
    "COURT", "SHEET", "HEARING", "COUNSEL"
}


def _coerce_scalar(val) -> str:
    if val is None:
        return ""
    if isinstance(val, str):
        return val.strip()
    if isinstance(val, list):
        for item in val:
            if isinstance(item, str) and item.strip():
                return item.strip()
        return ""
    return str(val).strip()


# Same denylist family as fix_judge_names.py's is_garbage_judge_name() --
# reused here because live testing showed the weaker original check (no
# word-count floor, no role/title check) was accepting real garbage from
# Ollama: "Shahdadkot" (a place name, not a judge) and "Justice of Peace,
# Larkana" (a role+place, not a person) both passed the original version
# of this function.
_PROSE_FUNCTION_WORDS = re.compile(
    r"\b(?:the|and|was|is|are|were|not|that|which|has|have|had|been|this|"
    r"these|those|from|with|for|will|shall|would|could|should|against|"
    r"them|others|by|preferred|reporting|approved)\b",
    re.IGNORECASE,
)
_JUSTICE_OF_PEACE_RE = re.compile(r"justice\s+of\s+peace", re.IGNORECASE)
# "K.N.Shah", "M.A.Khan" -- initials-dot-surname with no spaces is a real,
# common judicial naming style here; it would otherwise fail the 2-word
# floor below since there's no whitespace at all.
_INITIALS_SURNAME_RE = re.compile(r"^(?:[A-Z]\.){1,3}[A-Z][a-z]+$")


def is_valid_judge(name) -> bool:
    name = _coerce_scalar(name)
    if not name:
        return False
    if len(name) < 4 or len(name) > 60:
        return False
    if any(ch.isdigit() for ch in name):
        return False
    if "/" in name or "\\" in name:
        return False
    if _JUSTICE_OF_PEACE_RE.search(name):
        return False
    if len(_PROSE_FUNCTION_WORDS.findall(name)) >= 2:
        return False
    words = [w for w in re.split(r"\s+", name) if w]
    if len(words) < 2 and not _INITIALS_SURNAME_RE.match(name):
        # A judge's name is virtually always at least a first + last name.
        # A single token here (e.g. "Shahdadkot", a place name; "Hafiz", a
        # bare fragment) is far more likely something other than a real
        # judge's identity -- unlike fix_judge_names.py's cleanup of the
        # NEO4J graph, this per-file check has no cross-file case_count to
        # weigh against, so there's no safe way to make an exception here
        # (except the initials-surname pattern above, which is unambiguous).
        return False
    for w in words:
        letters_only = re.sub(r"[^A-Za-z]", "", w)
        if len(letters_only) < 2:
            return False
        if letters_only.upper() in BAD_JUDGE_TOKENS:
            return False
    return True


# ==========================================================
# OLLAMA FALLBACK (off by default -- only used with --use-ollama)
# ==========================================================
_OLLAMA_ALIVE = None


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
        print("⚠  Ollama not reachable -- judge fallback will be skipped for this run.", flush=True)
    return _OLLAMA_ALIVE


def _ollama_judge(full_text: str) -> str:
    if not _is_ollama_alive():
        return ""
    try:
        import requests
        snippet = full_text[:4000]
        prompt = (
            "You are extracting structured metadata from a Pakistani court judgment. "
            "Return ONLY a valid JSON object, no preamble, no markdown fences. "
            "Extract this field:\n"
            '- "judge": the name of the judge who authored/signed the order '
            "(not the typist's initials, not the word JUDGE itself). "
            "If no judge name can be found, return an empty string.\n\n"
            f"Document text:\n{snippet}"
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
        candidate = (parsed.get("judge") or "").strip()
        return candidate if is_valid_judge(candidate) else ""
    except Exception:
        return ""


# ==========================================================
# SCOPE + CHECKPOINT HELPERS (same convention as fix_jsoncase_numbers.py)
# ==========================================================
def load_repaired_filenames() -> set:
    if not os.path.exists(REPAIR_CHECKPOINT_FILE):
        raise FileNotFoundError(f"{REPAIR_CHECKPOINT_FILE} not found in this folder.")
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
            return set(json.load(f))
    except Exception:
        return set()


def save_checkpoint(done: set):
    tmp_path = CHECKPOINT_FILE + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(sorted(done), f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, CHECKPOINT_FILE)


def reconstruct_full_text(data: dict) -> str:
    chunks = sorted(data.get("chunks", []), key=lambda c: c.get("chunk_index", 0))
    text = " ".join(c.get("text", "") for c in chunks)
    return text[:MAX_TEXT_CHARS]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true",
                         help="Actually write changes. Without this, dry-run preview only.")
    parser.add_argument("--use-ollama", action="store_true", default=OLLAMA_ENABLED_DEFAULT,
                         help="Fall back to Ollama (off by default) when regex finds nothing.")
    parser.add_argument("--stuck-only", action="store_true",
                         help=f"Skip the full scan -- go straight to the file list already saved "
                              f"in {REVIEW_FILE} by a prior regex-only run (files a plain regex pass "
                              f"could not resolve). Much faster to start since it never re-reads/"
                              f"re-validates the files that were already fine.")
    args = parser.parse_args()

    if args.stuck_only:
        if not os.path.exists(REVIEW_FILE):
            raise SystemExit(f"{REVIEW_FILE} not found -- run without --stuck-only first to generate it.")
        with open(REVIEW_FILE, "r", encoding="utf-8") as f:
            stuck_names = {row["file"] for row in json.load(f)}
        all_files = [f for f in glob.glob(os.path.join(OUTPUT_DIR, "*.json")) if os.path.basename(f) in stuck_names]
        print(f"📋 Stuck-only mode: {len(stuck_names)} files from {REVIEW_FILE}")
        print(f"📂 Matched on disk: {len(all_files)}")
    else:
        repaired_names = load_repaired_filenames()
        all_files = [f for f in glob.glob(os.path.join(OUTPUT_DIR, "*.json")) if os.path.basename(f) in repaired_names]
        print(f"📋 In-scope files (from {REPAIR_CHECKPOINT_FILE}): {len(repaired_names)}")
        print(f"📂 Matched on disk: {len(all_files)}")

    done = load_checkpoint() if args.apply else set()
    pending = [f for f in all_files if os.path.basename(f) not in done]

    print(f"pending: {len(pending)}")
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

        old_judge = data.get("judge", "")
        if is_valid_judge(old_judge):
            if args.apply:
                done.add(os.path.basename(path))
                save_checkpoint(done)
            continue

        full_text = reconstruct_full_text(data)
        if not full_text.strip():
            if args.apply:
                done.add(os.path.basename(path))
                save_checkpoint(done)
            continue

        new_judge = extract_judge_from_text(full_text)
        if not new_judge and args.use_ollama:
            new_judge = _ollama_judge(full_text)

        if new_judge and is_valid_judge(new_judge):
            changed += 1
            print(f"[{i}/{len(pending)}] {os.path.basename(path)}: {old_judge!r} -> {new_judge!r}", flush=True)
            changed_audit.append({
                "file": os.path.basename(path),
                "old_judge": old_judge,
                "new_judge": new_judge,
            })
            if args.apply:
                data["judge"] = new_judge
                with open(path, "w", encoding="utf-8") as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
        else:
            flagged += 1
            needs_review.append({
                "file": os.path.basename(path),
                "old_judge": old_judge,
                "reason": "no_valid_candidate_found_in_text",
            })

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
            print("👉 Dry run does not checkpoint progress -- re-run will rescan from the start.")
    else:
        print(f"\n✅ Done. {changed} file(s) had judge corrected.")
        if flagged:
            print(f"⚠  {flagged} file(s) had no valid judge candidate found -- see {REVIEW_FILE}.")
        if not args.apply:
            print("👉 This was a DRY RUN. Re-run with --apply to write changes.")


if __name__ == "__main__":
    main()
