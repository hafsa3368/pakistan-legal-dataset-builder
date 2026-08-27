"""
fix_judge_names.py
====================
Audits the :Judge nodes in Neo4j (created by neo4j_import.py from the
extractor's heuristic judge-name regex) and:

  1. REJECTS pure garbage -- values that are not a real judge's name at all:
     - OCR-corrupted spellings of the word "JUDGE" itself (e.g. "ludge",
       "Udge", "Jucige", "OL JLCge") -- edit-distance match against "judge".
     - Placeholders ("(Not provided)", "N/A", "Unknown").
     - Sentence-fragment leaks (2+ common English function words together,
       same heuristic family as legal_answer.py's is_plausible_case_number).
     - Typist-tag artifacts ("Tufail/PA", "Faisal Mumtaz/PS").
     - Single-token entries (a real judge's name is virtually always 2+
       words; a lone "Hafiz" or "Bilal" is far more likely a fragment).
     - Role/title-only entries ("Justice of Peace <place>") -- not the name
       of a judge on the bench.

  2. CLUSTERS the remaining plausible names into groups that are almost
     certainly the same judge:
     - Splits "X AND Y" / "X & Y" into two separate candidates first (a
       multi-judge bench concatenated into one string).
     - Strips common noise tokens (Mr, Mrs, Miss, Sir, Dr, Justice, J, JJ,
       and known case-type-leak words like Petition/Appeal/Criminal/...)
       to get a normalized core name.
     - Exact-matches on the normalized core first, then a fuzzy pass
       (difflib.SequenceMatcher, same tool legal_answer.py already uses for
       near-duplicate text) to catch OCR typos like "Mazhar"/"Mahzar" or
       "Zafar"/"Zalar" among the normalized forms.
     - Picks ONE canonical name per cluster (the most complete/proper-cased
       original spelling seen, tie-broken by frequency).

DOES NOT touch anything by default -- this is a DRY RUN that writes report
files for review. Pass --apply to actually:
     - MERGE all DECIDED_BY edges from a cluster's duplicate Judge nodes
       onto the canonical Judge node, then delete the duplicate nodes.
     - DETACH DELETE garbage Judge nodes entirely (their DECIDED_BY edges
       are removed -- a case losing a bad judge link is strictly better
       than keeping a wrong one).
     - Update each affected Case node's own `judge_name` property to match
       (or clear it, for garbage).

USAGE:
    python fix_judge_names.py                 # dry run, writes report files
    python fix_judge_names.py --apply          # actually write changes to Neo4j
"""

import os
import re
import sys
import json
import difflib
import argparse
from collections import defaultdict

from dotenv import load_dotenv
from neo4j import GraphDatabase
from neo4j.exceptions import Neo4jError

# Same Windows console-encoding fix as legal_answer.py: judge names pulled
# from OCR'd documents can contain characters cp1252 can't encode, which
# would otherwise crash a print() after all the real work is already done.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

load_dotenv()

NEO4J_URI = "bolt://127.0.0.1:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

GARBAGE_REPORT_FILE = "fix_judge_names_garbage.json"
CLUSTER_REPORT_FILE = "fix_judge_names_clusters.json"

FUZZY_MERGE_THRESHOLD = 0.90


# ==========================================================
# GARBAGE DETECTION
# ==========================================================
_PROSE_FUNCTION_WORDS = re.compile(
    r"\b(?:the|and|was|is|are|were|not|that|which|has|have|had|been|this|"
    r"these|those|from|with|for|will|shall|would|could|should|against|"
    r"them|others|by|preferred|reporting|approved|for)\b",
    re.IGNORECASE,
)
_PLACEHOLDER_RE = re.compile(
    r"^\s*(?:\(?not\s+(?:provided|available)\)?|n/?a|unknown|none|null|-+)\s*$", re.IGNORECASE
)
_JUSTICE_OF_PEACE_RE = re.compile(r"^\s*justice\s+of\s+peace\b", re.IGNORECASE)

# Stripped BEFORE garbage-checking (not rejection triggers by themselves):
# a real name is often still present once these are removed, e.g. "JUDGE
# Sulemen Khan/PA" (82 cases!) is Judge Suleman Khan with a typist-initials
# tag stuck on -- rejecting the whole string outright would have silently
# discarded 82 genuine DECIDED_BY links for one real judge.
_TYPIST_TAG_STRIP_RE = re.compile(r"\s*/\s*(?:PA|PS)\b", re.IGNORECASE)
_LEADING_TRAILING_JUDGE_WORD_RE = re.compile(r"(?:^\s*judge\s+|\s+judge\s*$)", re.IGNORECASE)


def clean_judge_name_artifacts(name: str) -> str:
    """Strips typist-tag suffixes ("/PA", "/PS") and any leading/trailing
    literal "JUDGE" word (repeatedly, so "JUDGE Judge Sabir" -> "Sabir")
    before the real garbage/plausibility check runs."""
    cleaned = _TYPIST_TAG_STRIP_RE.sub("", name)
    prev = None
    while prev != cleaned:
        prev = cleaned
        cleaned = _LEADING_TRAILING_JUDGE_WORD_RE.sub(" ", cleaned).strip()
    return cleaned.strip()


def _edit_distance_le(a: str, b: str, max_dist: int) -> bool:
    """Cheap bounded Levenshtein check -- used only to catch short OCR-
    corrupted spellings of the single word 'judge' (a, b both short), so
    an O(n*m) DP table is fine here."""
    a, b = a.lower(), b.lower()
    if abs(len(a) - len(b)) > max_dist:
        return False
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        for j, cb in enumerate(b, 1):
            cur[j] = min(
                prev[j] + 1,
                cur[j - 1] + 1,
                prev[j - 1] + (ca != cb),
            )
        prev = cur
    return prev[len(b)] <= max_dist


SINGLE_TOKEN_GARBAGE_MAX_COUNT = 3


def is_garbage_judge_name(name: str, case_count: int = 0) -> str:
    """Returns a non-empty reason string if `name` is garbage (not a real
    judge's name), else ''. `case_count` (how many Case nodes point to
    this exact name) is used only for the single-token rule below --
    everything else is judged on the string alone."""
    if not name or not name.strip():
        return "empty"
    text = name.strip()

    if _PLACEHOLDER_RE.match(text):
        return "placeholder"

    if _JUSTICE_OF_PEACE_RE.match(text):
        return "role_title_not_a_name"

    # OCR-corrupted spelling of the word "JUDGE" itself, standing alone
    # (optionally with junk punctuation) as if it were a person's name.
    core = re.sub(r"[^A-Za-z]", "", text)
    if 3 <= len(core) <= 7 and _edit_distance_le(core, "judge", 2):
        return "ocr_corrupted_word_judge"

    if len(text) > 80:
        return "too_long_likely_prose"

    if len(_PROSE_FUNCTION_WORDS.findall(text)) >= 2:
        return "prose_fragment_leak"

    words = [w for w in re.split(r"\s+", text) if w]
    if len(words) < 2:
        # A single token appearing only a handful of times is much more
        # likely a stray OCR fragment than a judge's actual identifier.
        # But one appearing on dozens/hundreds of cases (e.g. "Ahmad" on
        # 230 cases) is almost certainly a real judge whose full name the
        # extractor simply never captured beyond one word -- rejecting
        # that would silently destroy hundreds of real DECIDED_BY links
        # for one false-positive-prone heuristic. Keep it, imperfect name
        # and all; an incomplete-but-real name is a smaller harm here
        # than deleting genuine data.
        if case_count <= SINGLE_TOKEN_GARBAGE_MAX_COUNT:
            return "single_token_likely_fragment"

    return ""


# ==========================================================
# NORMALIZATION + CLUSTERING
# ==========================================================
_NOISE_TOKENS = {
    "mr", "mr.", "mrs", "mrs.", "miss", "sir", "dr", "dr.", "justice",
    "chief", "j", "jj", "j.", "the", "hon'ble", "honble", "honourable",
    "petition", "petitions", "appeal", "appeals", "criminal", "civil",
    "application", "applications", "bail", "jail", "revision", "case",
    "cases", "court", "judicial", "department",
}


def split_multi_judge(name: str) -> list:
    parts = re.split(r"\s+(?:AND|&)\s+", name.strip(), flags=re.IGNORECASE)
    return [p.strip() for p in parts if p.strip()]


def normalize_core_name(name: str) -> str:
    words = re.split(r"\s+", name.strip())
    kept = [w for w in words if re.sub(r"[^a-zA-Z]", "", w).lower() not in _NOISE_TOKENS]
    core = " ".join(kept).strip()
    core = re.sub(r"[^A-Za-z\s'\-]", "", core)
    core = re.sub(r"\s+", " ", core).strip()
    return core.upper()


def to_title_case(core: str) -> str:
    """Title-cases an already-normalized (noise-stripped, upper-cased)
    core name, without botching apostrophes/hyphens, e.g.
    "ABDU'L HAFIZ" -> "Abdu'l Hafiz"."""
    out_words = []
    for word in core.split(" "):
        if not word:
            continue
        pieces = re.split(r"([\'\-])", word)
        pieces = [p.capitalize() if p not in ("'", "-") else p for p in pieces]
        out_words.append("".join(pieces))
    return " ".join(out_words)


# ==========================================================
# MAIN AUDIT
# ==========================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true",
                         help="Actually write changes to Neo4j. Without this, dry-run preview only.")
    args = parser.parse_args()

    if not NEO4J_PASSWORD:
        raise SystemExit("NEO4J_PASSWORD not set in .env")

    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    driver.verify_connectivity()
    print("Connected to Neo4j.\n")

    with driver.session() as session:
        rows = list(session.run(
            """
            MATCH (j:Judge)
            OPTIONAL MATCH (c:Case)-[:DECIDED_BY]->(j)
            RETURN j.name AS name, count(c) AS case_count
            """
        ))

    print(f"Total Judge nodes: {len(rows)}\n")

    # -------------------------------------------------------
    # STEP 1: garbage vs candidate
    # -------------------------------------------------------
    garbage = []       # [{"name":..., "reason":..., "case_count":...}]
    candidates = []     # [(name, case_count)] -- possibly split from "X AND Y"

    for r in rows:
        raw_name, case_count = r["name"], r["case_count"]
        cleaned_name = clean_judge_name_artifacts(raw_name)
        if not cleaned_name:
            garbage.append({"name": raw_name, "reason": "empty_after_cleaning", "case_count": case_count})
            continue
        reason = is_garbage_judge_name(cleaned_name, case_count)
        if reason:
            garbage.append({"name": raw_name, "reason": reason, "case_count": case_count})
            continue
        for part in split_multi_judge(cleaned_name):
            part_reason = is_garbage_judge_name(part, case_count)
            if part_reason:
                garbage.append({"name": f"{raw_name} [split part: {part}]", "reason": part_reason, "case_count": case_count})
                continue
            candidates.append((part, case_count, raw_name))  # (cleaned part, count, ORIGINAL raw Neo4j node name)

    print(f"Garbage (not a real judge name): {len(garbage)}")
    print(f"Candidate real names (post-split): {len(candidates)}\n")

    # -------------------------------------------------------
    # STEP 2: exact-normalized clustering
    # -------------------------------------------------------
    groups = defaultdict(list)  # normalized_core -> [(original_node_name, case_count, part_name)]
    for part_name, case_count, original_node_name in candidates:
        core = normalize_core_name(part_name)
        if not core:
            garbage.append({"name": original_node_name, "reason": "empty_after_normalization", "case_count": case_count})
            continue
        groups[core].append((original_node_name, case_count, part_name))

    print(f"Distinct normalized cores: {len(groups)}")

    # -------------------------------------------------------
    # STEP 3: fuzzy pass over the normalized cores (catches OCR typos),
    # using union-find so a chain of typo variants (e.g. RAJPU / RAJPUL /
    # RAJPUT / RAPUL / RAPUT) all end up in one cluster regardless of
    # comparison order.
    # -------------------------------------------------------
    core_total_count = {core: sum(e[1] for e in entries) for core, entries in groups.items()}
    cores = list(groups.keys())
    parent = {c: c for c in cores}

    def find(c):
        while parent[c] != c:
            parent[c] = parent[parent[c]]
            c = parent[c]
        return c

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra == rb:
            return
        # Weighted union: the more heavily-used spelling stays as the
        # union-find root (purely internal bookkeeping -- canonical TEXT
        # is chosen separately below by actual case_count, not by root).
        if core_total_count.get(ra, 0) >= core_total_count.get(rb, 0):
            parent[rb] = ra
        else:
            parent[ra] = rb

    for i, core_a in enumerate(cores):
        for core_b in cores[i + 1:]:
            if abs(len(core_a) - len(core_b)) > 4:
                continue
            if len(core_a.split()) != len(core_b.split()):
                continue
            ratio = difflib.SequenceMatcher(None, core_a, core_b).ratio()
            if ratio >= FUZZY_MERGE_THRESHOLD:
                union(core_a, core_b)

    root_members = defaultdict(list)
    for core in cores:
        root_members[find(core)].append(core)

    print(f"Final clusters after fuzzy merge: {len(root_members)}\n")

    # -------------------------------------------------------
    # STEP 4: canonical name per cluster + report
    # --------------------------------------------------------
    # Revision fix: canonical name comes from the MOST-USED normalized
    # core (title-cased), never from a raw original node string -- a raw
    # original can still carry noise ("... Mr", "... Case", "X and Y")
    # that normalize_core_name() already stripped for grouping purposes,
    # and simply picking "the longest original spelling" (the previous
    # approach) let that noise leak straight into the canonical name.
    # -------------------------------------------------------
    cluster_report = []
    multi_node_clusters = 0
    for root, member_cores in root_members.items():
        entries = []
        for core in member_cores:
            entries.extend(groups[core])
        best_core = max(member_cores, key=lambda c: core_total_count[c])
        canonical = to_title_case(best_core)
        distinct_node_names = sorted({e[0] for e in entries})
        total_cases = sum(e[1] for e in entries)
        if len(distinct_node_names) > 1:
            multi_node_clusters += 1
        cluster_report.append({
            "canonical_name": canonical,
            "total_case_count": total_cases,
            "merged_from": distinct_node_names,
        })

    cluster_report.sort(key=lambda c: c["total_case_count"], reverse=True)

    with open(GARBAGE_REPORT_FILE, "w", encoding="utf-8") as f:
        json.dump(garbage, f, ensure_ascii=False, indent=2)
    with open(CLUSTER_REPORT_FILE, "w", encoding="utf-8") as f:
        json.dump(cluster_report, f, ensure_ascii=False, indent=2)

    print(f"Clusters that merge 2+ distinct Judge nodes into one: {multi_node_clusters}")
    print(f"\nReports written:\n  {GARBAGE_REPORT_FILE} ({len(garbage)} entries)\n  {CLUSTER_REPORT_FILE} ({len(cluster_report)} entries)\n")

    print("Sample of merge clusters (largest first):")
    shown = 0
    for c in cluster_report:
        if len(c["merged_from"]) > 1:
            print(f"  -> {c['canonical_name']!r}  (from {c['merged_from']}, {c['total_case_count']} cases)")
            shown += 1
        if shown >= 15:
            break

    if not args.apply:
        print("\nDRY RUN complete. Review the report files, then re-run with --apply to write changes.")
        driver.close()
        return

    # -------------------------------------------------------
    # APPLY -- two phases, in this order, so a Judge node that is a
    # duplicate source for TWO different clusters (a raw "X AND Y" bench
    # node split into two real judges) gets its edges redirected to BOTH
    # canonicals before it is deleted, instead of the first cluster
    # processed deleting it out from under the second.
    #   Phase A: redirect every duplicate's DECIDED_BY edges to its
    #            cluster's canonical Judge node (nothing deleted yet).
    #   Phase B: delete every duplicate node (now safe -- redirected),
    #            and every pure-garbage node not used as a redirect
    #            source (those get judge_name cleared instead of set).
    # -------------------------------------------------------
    print("\nAPPLYING changes to Neo4j...")

    redirect_pairs = []  # (duplicate_node_name, canonical_name)
    for c in cluster_report:
        canonical = c["canonical_name"]
        for dup in c["merged_from"]:
            if dup != canonical:
                redirect_pairs.append((dup, canonical))

    redirected_source_names = {dup for dup, _ in redirect_pairs}
    pure_garbage_names = sorted({
        g["name"].split(" [split part:")[0] for g in garbage
    } - redirected_source_names)

    with driver.session() as session:
        # Phase A: redirect edges (does not delete anything).
        for dup_name, canonical in redirect_pairs:
            try:
                session.run(
                    """
                    MERGE (canonical:Judge {name: $canonical})
                    WITH canonical
                    MATCH (dup:Judge {name: $dup_name})
                    OPTIONAL MATCH (c:Case)-[:DECIDED_BY]->(dup)
                    FOREACH (_ IN CASE WHEN c IS NOT NULL THEN [1] ELSE [] END |
                        MERGE (c)-[:DECIDED_BY]->(canonical)
                        SET c.judge_name = $canonical
                    )
                    """,
                    canonical=canonical, dup_name=dup_name,
                )
            except Neo4jError as e:
                print(f"  WARN: could not redirect {dup_name!r} -> {canonical!r}: {e}")

        # Phase B1: delete duplicate nodes now that their edges are safe.
        try:
            session.run(
                """
                UNWIND $names AS n
                MATCH (j:Judge {name: n})
                DETACH DELETE j
                """,
                names=sorted(redirected_source_names),
            )
        except Neo4jError as e:
            print(f"  WARN: could not delete redirected duplicate nodes: {e}")

        # Phase B2: pure garbage (never used as a redirect source) --
        # clear judge_name on any case pointing to it, then delete.
        for name in pure_garbage_names:
            try:
                session.run(
                    """
                    MATCH (j:Judge {name: $name})
                    OPTIONAL MATCH (c:Case)-[:DECIDED_BY]->(j)
                    SET c.judge_name = ''
                    DETACH DELETE j
                    """,
                    name=name,
                )
            except Neo4jError as e:
                print(f"  WARN: could not delete garbage judge {name!r}: {e}")

    print(f"Redirected {len(redirect_pairs)} duplicate-node edge set(s).")
    print(f"Deleted {len(redirected_source_names)} duplicate node(s) and {len(pure_garbage_names)} pure-garbage node(s).")
    print("Done.")
    driver.close()


if __name__ == "__main__":
    main()
