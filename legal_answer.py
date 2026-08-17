"""
legal_answer.py

Purpose
-------
RAG answer-generation layer for Hafsa's Pakistani Legal Research Assistant.
Sits ON TOP of the existing, working qdrant_to_neo4j_similarity.py pipeline.

This script does NOT:
  - regenerate embeddings
  - modify Qdrant data
  - create/modify/delete Neo4j nodes or relationships
  - recreate SIMILAR_TO relationships (it only READS existing ones)
  - touch qdrant_to_neo4j_similarity.py in any way

It ONLY:
  1. Takes a user's legal query (free text).
  2. Reuses qdrant_to_neo4j_similarity.py's own functions to:
       - embed the query (Ollama, nomic-embed-text)
       - search the existing "legal_chunks" Qdrant collection
       - group hits by case_id
       - resolve each case_id against Neo4j (authoritative case_number)
       - pick the Central Case (manual override or auto top-scoring case)
  3. Additionally (new in this file, read-only):
       - pulls fuller chunk text for those same hits directly from the raw
         Qdrant results (your existing script only keeps a 200-char preview)
       - pulls each retrieved case's own Neo4j metadata (court, year, etc.)
       - pulls existing SIMILAR_TO neighbours for internal/debug use only
         (never sent to the LLM, never citable)
  4. Selects a small, high-relevance slice of evidence, anonymizes it behind
     [E1]/[E2]/... tags, and sends it to a LOCAL Ollama LLM with a prompt
     that explicitly permits evidence-grounded synthesis.
  5. Converts the model's tagged output into a clean, professional,
     user-facing legal research answer with natural inline citations
     (e.g. "(W.P. No.61653 of 2020, Lahore High Court, 2022)") -- no
     case_id, chunk id, similarity score, or validation-status text is
     ever shown to the user.
  6. If the model fails to produce any usable [Ex] tags (even after one
     retry), or leaks an off-corpus (non-Pakistani) legal reference, Python
     assembles a deterministic, evidence-grounded answer directly from the
     retrieved case text itself, rather than telling the user the evidence
     was insufficient.

Run:
    python legal_answer.py
    python legal_answer.py --debug     (also prints internal grounding info)

--------------------------------------------------------------------------
CHANGELOG
--------------------------------------------------------------------------
Retrieval, Qdrant, Neo4j, and qdrant_to_neo4j_similarity.py are untouched in
every revision below. Everything is in the answer-generation / presentation
layer only.

REVISION 1 -- fixed over-refusal, timeouts, leaked technical fields, raw
(Source: ...) citations. REVISION 2 -- fixed case-specific facts leaking
into the user's hypothetical. REVISION 3 -- fixed bare (unbracketed)
evidence-label leaks and persistent over-refusal on bail/inheritance
queries. REVISION 4 -- stopped treating "no [Ex] tags" as "no evidence":
added a retry with a strengthened prompt, and a deterministic,
evidence-only fallback (no LLM call) for when the model still can't
produce a taggable answer. REVISION 5 -- added a Python-built
"SOURCES CONSULTED" list, independent of the model's own tagging, so real
verified citations always appear when evidence was retrieved. (See prior
revisions of this file for full detail; unchanged here.)

REVISION 6 (this revision) -- THREE QUALITY FIXES, all in the
answer-assembly / validation layer. Retrieval, evidence selection scoring
inputs, and the [Ex] grounding design are otherwise unchanged; the only
new *behaviour* is (a) one extra evidence-score floor at selection time,
and (b) one extra post-generation check that can trigger the SAME
retry/fallback machinery Revision 4 already built, for a new reason.

FIX A -- OFF-CORPUS / FOREIGN-LAW LEAKAGE:
  Problem: the 3B local model occasionally names non-Pakistani legal
  authorities it has seen in training data (e.g. "Bail Reform Act",
  "U.S.C.", "Indian Penal Code") that never appeared in the evidence text
  it was given. This is a hallucination risk specific to a Pakistan-only
  legal assistant.
  - `FOREIGN_LAW_PATTERN` / `contains_foreign_law_reference()` (new) -- a
    denylist regex scan of the model's raw output, checked against the
    literal evidence text so a term is only flagged if the model
    introduced it itself (not if the evidence genuinely quotes it).
  - `answer_legal_query()` -- the existing "zero valid tags -> retry once,
    then deterministic fallback" flow (Revision 4) now ALSO triggers on
    "evidence tags present but contains a foreign-law leak." A leaked
    foreign-law answer is never shown to the user, exactly like an
    untagged one.
  - `MIN_EVIDENCE_SCORE` (new) -- `build_evidence_items()` now drops
    candidate chunks below a score floor before selection, rather than
    filling all `MAX_TOTAL_EVIDENCE_ITEMS` slots regardless of how weakly
    a chunk actually matched. This reduces the chance of an off-topic
    chunk (e.g. a Section 376 PPC case pulled into a plain divorce query)
    getting synthesized alongside genuinely relevant evidence in the
    first place. The central case's own top chunk is still guaranteed a
    slot even if below the floor (unchanged safety-net behaviour from
    Revision 4), so a weak central match still produces an answer.

FIX B -- WEAK CITATION FALLBACK TEXT:
  Problem: `natural_citation()` fell back to the generic
  "(a retrieved <Court> case)" more often than necessary, because
  `authoritative_meta()` never checked Neo4j's own `case_number` field --
  it only ever used whatever `case_number` the Qdrant-side `CaseHit`
  object happened to carry.
  - `authoritative_meta()` -- now also reads `case_number` from the
    Neo4j graph metadata (authoritative, same as court/year already were)
    before falling back to the `CaseHit`'s own value.
  - `natural_citation()` -- now prefers `meta["case_number"]` (the
    graph-verified value) over `qn.display_label(case)`, falling back to
    the latter only if the graph genuinely has nothing. No new source of
    truth was added: this only reorders which already-existing verified
    field is checked first.

FIX C -- APPLICATION SECTION OVERSTATEMENT:
  Problem: the model's APPLICATION TO THE QUERY section sometimes blurred
  "what the case held," "the general principle," and "how it might apply
  to the user" into a single paragraph, occasionally producing
  conclusory language ("this suggests the court would prioritize...")
  that reads more confidently than the evidence supports.
  - `build_prompt()` -- the APPLICATION TO THE QUERY instructions now
    require three explicitly labeled parts per point ("Held:",
    "Principle:", "Relevance to your question:"), with the last part
    restricted to conditional phrasing only. This is additive to the
    existing CASE FACTS VS. THE USER'S HYPOTHETICAL rule, not a
    replacement for it.
  - `OVERASSERTION_PATTERN` / `check_overassertion()` (new) -- a
    lightweight post-generation regex backstop (same pattern as
    `strip_leaked_metadata`) that flags conclusive language ("will
    succeed", "is entitled to", "the court will rule") in the rendered
    APPLICATION section. This does not rewrite the model's prose (doing
    so risks distorting it); it appends a clarifying note to LIMITATIONS
    when triggered, same mechanism already used for invalid tags.
  - `build_deterministic_fallback_answer()`'s APPLICATION TO THE QUERY
    text was already fully conditional/non-conclusory by construction
    (Revision 4) and needed no change.

REVISION 7 (this revision) -- TWO BUGS FOUND FROM LIVE RUN LOGS. Both are
in the answer-assembly layer only; retrieval and evidence-scoring inputs
are unchanged except for the backfill described below (which only ever
ADDS candidates that retrieval had already found, never new ones).

FIX D -- DUPLICATE/LEAKED "LIMITATIONS" TEXT:
  Observed: a run's output contained a stray "LIMITATIONS: None" line
  inside the APPLICATION TO THE QUERY text, immediately followed by a
  second, real "LIMITATIONS:" section. Root cause: `_SECTION_PATTERN`
  required a heading to be ALONE on its own line; when the model wrote a
  short section's heading and content on the SAME line (e.g.
  "LIMITATIONS: None"), the pattern didn't match it at all, so that text
  was silently absorbed into the previous section's content instead of
  being recognized as its own heading.
  - `_SECTION_PATTERN` -- now only requires the heading name plus a
    mandatory trailing colon at the start of a line; it no longer
    requires the rest of that line to be empty. The colon stays
    mandatory (not optional) specifically to avoid accidentally matching
    a heading-like word inside ordinary prose.
  - `parse_llm_sections()` -- docstring updated to note that if a section
    name is matched more than once (the scenario above), the LAST
    occurrence wins, which correctly discards the earlier stray heading
    and reassigns the intervening text to where it actually belongs.

FIX E -- EVIDENCE STARVATION FROM THE REVISION 6 SCORE FLOOR:
  Observed: a query returned only 1 evidence chunk after the
  MIN_EVIDENCE_SCORE floor (Revision 6, FIX A) filtered out the rest, and
  the resulting answer was visibly confused -- the model put its
  Held/Principle/Relevance structure in the wrong section and left
  APPLICATION TO THE QUERY as "insufficient evidence" even though it had
  just used that evidence elsewhere. A near-empty evidence set gives the
  model nothing to synthesize or contrast, which appears to correlate
  with it losing track of the required section structure.
  - `MIN_EVIDENCE_ITEMS_BACKFILL_FLOOR` (new) -- if the score floor
    leaves fewer than this many candidate chunks, `build_evidence_items()`
    now backfills with the next-best BELOW-floor candidates (still
    score-ranked, not discarded outright) until the minimum is reached or
    candidates run out. This only engages when the strong (above-floor)
    candidate pool is thin; it never removes the floor's effect when
    enough strong evidence already exists.

REVISION 8 (this revision) -- ONE MORE LEAK FOUND FROM LIVE RUN LOGS, plus
a cosmetic spacing fix. Both in the answer-assembly layer only.

FIX F -- MULTI-TAG BRACKET LEAK, e.g. "[E2, E4]":
  Observed: a run's APPLICATION TO THE QUERY text ended with the raw,
  unconverted fragment "[E2, E4]" visible to the user. Root cause: the
  model wrote two evidence tags inside ONE bracket, comma-separated,
  instead of two separate brackets. SINGLE_TAG_PATTERN only ever matched
  a bracket containing nothing but "[Ee]<digits>", so "[E2, E4]" matched
  neither the valid-tag pattern nor anything that would mark it invalid
  and drop it -- it simply passed through untouched, exactly the class of
  raw technical leak this file exists to prevent (see REVISION 1).
  - `MULTI_TAG_BRACKET_PATTERN` / `normalize_multi_tag_brackets()` (new)
    -- splits a bracket containing multiple comma/semicolon/slash-
    separated tags into individual adjacent single-tag brackets (e.g.
    "[E2, E4]" -> "[E2][E4]") BEFORE the rest of the tag pipeline runs, so
    each one is then handled exactly like a normally-written tag: either
    converted to a real citation or silently dropped if invalid.
  - Applied in BOTH places tags are recognized: `count_valid_tags()` (so
    a multi-tag bracket counts correctly toward the retry decision) and
    `build_user_facing_answer()`'s cleanup step (so it's normalized before
    section parsing and citation conversion).

FIX G -- SPACING GLITCH BETWEEN ADJACENT CITATIONS (cosmetic):
  Observed: when the model wrote two citation groups back-to-back with no
  space between them, the rendered output looked like
  "...Court)and(Crl. Bail...". `convert_citations_to_natural()` now also
  inserts a space wherever a letter directly abuts a parenthesis in
  either direction, after citation substitution.

REVISION 9 (this revision) -- ONE MORE ISSUE FOUND FROM LIVE RUN LOGS, in
the answer-assembly / presentation layer only.

FIX H -- GARBLED case_number DISPLAYED AS A FAKE CITATION:
  Observed: "SOURCES CONSULTED" showed "- construction of the BHU. It was
  not, Lahore High Court, 2023" -- not a real case number, but a
  chunk-text sentence fragment that had leaked into that Case node's
  case_number field upstream (the same class of problem the module
  docstring's extractor.py history already describes, evidently not
  fully eliminated for every record). It was displayed to the user as if
  it were a genuine, verified citation label -- worse than showing
  nothing, since it looks plausible at a glance.
  - `is_plausible_case_number()` (new) -- a coarse heuristic guard, same
    spirit as the existing `is_valid_year()`: rejects a case_number value
    if it's implausibly long, contains no digit at all (real case
    numbers virtually always do), or contains two or more common English
    function words together (a strong signal of a sentence fragment
    rather than a citation).
  - `authoritative_meta()` -- now runs this guard on BOTH the
    graph-sourced case_number and the CaseHit-side fallback before
    trusting either one, exactly mirroring how `is_valid_year()` already
    gates the year field.
  - `natural_citation()` -- the same guard is applied to its
    `qn.display_label(case)` fallback path too (a route
    `authoritative_meta()` doesn't cover).
  - Net effect: a garbled case_number is now treated exactly like a
    missing one -- the citation falls back to the existing, honest
    "(a retrieved <Court> case)" text (or drops entirely if the court is
    also unverified) rather than showing fabricated-looking text. No
    Neo4j data is touched; this only changes what the presentation layer
    is willing to trust.

REVISION 10 (this revision) -- TWO ISSUES FOUND FROM LIVE RUN LOGS, both
in the answer-assembly / presentation layer only.

FIX I -- DUPLICATE-LOOKING SOURCES CONSULTED ENTRIES:
  Observed: "SOURCES CONSULTED" showed the same line twice --
  "- a retrieved Lahore High Court case" -- for a query where two
  different retrieved cases both lacked a verified case_number and so
  both fell back to the same generic natural_citation() text. Technically
  two different case_ids, but visually identical and redundant to the
  reader.
  - `build_supporting_references()` -- now dedupes by the RENDERED
    CITATION TEXT in addition to the existing case_id dedup, so two
    cases that both render as "(a retrieved <Court> case)" only appear
    once in the list.

FIX J -- STRAY OCR UNDERSCORE ARTIFACTS IN THE DETERMINISTIC FALLBACK:
  Observed: a deterministic-fallback answer (triggered when the LLM
  produced zero valid tags on both attempts) echoed raw evidence text
  containing an OCR underline artifact verbatim: "...offence charged._
  When a person...". This fallback path is designed to show evidence
  largely as retrieved, but a stray underscore character isn't
  meaningful content -- it's OCR noise from the original PDF.
  - `clean_evidence_snippet()` -- now strips runs of standalone
    underscores before trimming/truncating, in addition to its existing
    whitespace collapsing.

REVISION 11 (this revision) -- CONFIRMED MODEL HALLUCINATION, NOT A DATA
BUG. Following the recurring "welfare of the minor is treated as the
paramount consideration" text seen across multiple unrelated live-run
queries (bail, eviction, criminal appeal), a full client-side scan of the
entire Qdrant collection (55,829 points) confirmed ZERO points contain
this phrase anywhere. It is not a chunking, embedding, or duplicate-data
bug -- the model is inserting a well-known, memorized Pakistani custody-
law formula from its own training data and attaching it to whichever
nearby valid [Ex] tag happens to be available, which passes the existing
structural grounding check (a valid tag exists) even though that tag's
actual evidence has nothing to do with the claim. The module's grounding
design deliberately does not verify a claim's CONTENT against its
evidence (see "Note on citation-string normalization" -- general fuzzy
matching was rejected as unreliable and easy to fool); this revision does
not change that. Instead it extends the SAME evidence-membership check
already used for FIX A (foreign-law leaks) to a small, specific,
empirically-verified denylist -- not a guess at what "sounds
hallucinated."

FIX K -- KNOWN-HALLUCINATION-PHRASE DETECTION:
  - `KNOWN_HALLUCINATION_PATTERNS` / `contains_known_hallucination()`
    (new) -- same evidence-membership logic as
    `contains_foreign_law_reference()`: flags a denylisted phrase only if
    it does NOT appear in the evidence text actually given to the model
    for that query. Currently contains one entry (the confirmed custody
    formula above). Add further entries only after the same kind of
    corpus-wide verification -- this stays small and evidence-backed.
  - `answer_legal_query()` -- attempt 1 and the retry now check BOTH
    `contains_foreign_law_reference()` and `contains_known_hallucination()`
    (combined into one "unsupported phrase(s)" signal) and trigger the
    same retry -> deterministic-fallback path Revision 4/6 already built.
  - `build_prompt()`'s strengthen block -- generalized from "foreign-law
    leak" wording to cover either category, since the same parameter now
    carries both; it tells the model specifically which statement(s) were
    rejected and reminds it that even a well-known principle must not be
    written unless the evidence actually contains it.
  - The BASE (non-strengthened) prompt also gained a standing instruction
    against using memorized outside legal knowledge and against
    attaching a tag to a sentence just because that tag is nearby/
    available, to reduce how often this happens on the FIRST attempt,
    not just catch it on retry.

REVISION 12 (this revision) -- ONE CRITICAL REGRESSION FIX (introduced by
this file's own Revision 7) and one recurring-pattern corrective
heuristic, both found from live run logs.

FIX L (CRITICAL) -- REVISION 7 REGRESSION, ENTIRE ANSWER SWALLOWED INTO
ONE SECTION:
  Observed: a run's final answer showed a huge, unstructured block of
  text under "LEGAL ISSUE:", including a stray bare "ANSWER" heading and
  Held/Principle/Relevance content buried inside it, while every OTHER
  section (ANSWER / SUMMARY, RELEVANT LEGAL PRINCIPLES, RELEVANT CASE
  LAW, APPLICATION TO THE QUERY) showed only the standard empty-section
  fallback text. Root cause: Revision 7 made the heading colon
  MANDATORY to fix a same-line-content leak ("LIMITATIONS: None"). That
  fix was correct for that case, but had an unintended side effect: if
  the model writes a heading bare, alone on its own line, with NO colon
  at all (a common and previously-supported pattern, e.g. just "ANSWER"
  with content starting on the next line), it no longer matched
  `_SECTION_PATTERN` at all -- so that heading, and EVERYTHING after it
  through the end of the model's output, got swallowed into whichever
  earlier heading WAS recognized (here, LEGAL ISSUE, the only one with a
  colon that run).
  - `_SECTION_PATTERN` -- colon is optional again (`:?`, matching the
    original Revision 1-6 behavior), which restores recognition of bare,
    colon-less headings. The Revision 7 fix is fully preserved alongside
    it: the pattern still doesn't require the rest of the line to be
    empty (no trailing `$` anchor), so a heading with inline same-line
    content (e.g. "LIMITATIONS: None") is still captured correctly, and
    a heading with NO colon and NO inline content (e.g. bare "ANSWER" on
    its own line) is also now correctly captured again. Both known
    failure modes are covered by the one pattern.

FIX M -- MISPLACED APPLICATION-SECTION CONTENT (recurring pattern, not a
new bug in this file, but now large enough across live runs to warrant a
corrective heuristic):
  Observed, repeatedly: the model writes the required Held: / Principle:
  / Relevance to your question: structure (intended for APPLICATION TO
  THE QUERY) inside RELEVANT CASE LAW instead, leaving APPLICATION TO
  THE QUERY empty or just the standard insufficiency sentence -- even
  though the model clearly had something to say about application, it
  just filed it under the wrong heading.
  - `build_user_facing_answer()` -- now checks, on the RAW (pre-citation-
    conversion) section text, whether RELEVANT CASE LAW contains all
    three labels (Held:/Principle:/Relevance to your question:) while
    APPLICATION TO THE QUERY is empty or the standard insufficiency
    sentence. If so, the raw text from "Held:" onward is moved from
    RELEVANT CASE LAW into APPLICATION TO THE QUERY before citation
    conversion runs on either section -- so citations in the moved text
    are still resolved normally, just attached to the correct final
    section. This is a structural move of whole raw text, not content
    rewriting, and only activates on this specific, now well-established
    pattern.

REVISION 13 (this revision) -- TWO ISSUES FOUND FROM LIVE RUN LOGS, both
in the answer-assembly / presentation layer only.

FIX N -- DOUBLE PARENTHESES AROUND CITATIONS:
  Observed: "...no other adequate remedy is provided by law. ( (W.P.No.2571
  of 2021, Lahore High Court, 2022))" -- doubled parens. Root cause: the
  model sometimes wraps its OWN parentheses directly around a tag with
  nothing else inside, e.g. "([E4])". `CITATION_GROUP_PATTERN` previously
  only matched the bare "[E4]" portion, leaving the model's own "(" and
  ")" untouched -- so `natural_citation()`'s own parenthesized output
  ended up nested inside them.
  - `CITATION_GROUP_PATTERN` -- now tries to match a FULLY parenthesized
    tag-only group first (both "(" and ")" immediately and only wrapping
    the tag(s), nothing else inside) before falling back to the bare tag
    group. Deliberately NOT independently-optional parens on each side --
    that would risk swallowing an unrelated trailing ")" from a genuine
    enclosing phrase like "(see [E4])" (whose "(" belongs to "see", not
    the tag) while leaving its matching "(" behind, unbalancing the
    result in the opposite direction. Requiring both sides together (or
    neither) avoids that failure mode.

FIX O -- REVISION 9 GUARD GAP, GARBLED case_number STILL SLIPPING THROUGH:
  Observed: "SOURCES CONSULTED" showed "- constitutional petition are
  that Respondent No.1, Lahore High Court, 2018" -- another chunk-text
  fragment leaked into a case_number field (same class of problem FIX H /
  Revision 9 targeted), but this one passed `is_plausible_case_number()`
  because its function-word denylist was missing the word "are" -- the
  fragment only tripped one denylisted word ("that"), under the
  2-word-minimum threshold.
  - `_PROSE_FUNCTION_WORDS` -- added "are" to the denylist. Verified this
    doesn't create false positives against real case-number formats
    (e.g. "W.P. No.2050 of 2024", "constitutional petitions No. 25057 of
    2020" both still pass).

Note on citation-string normalization: still not needed as a general
mechanism (see Revision 3 note) -- citation validity remains structural
(Python-owned [Ex] -> case_id mapping), not string-matched, in every
code path including the deterministic fallback. The Revision 6/11
denylist checks and the Revision 12 section-relocation heuristic are
narrow, targeted exceptions for specific confirmed failure patterns, not
a general content-matching or rewriting system.
"""

import re
import sys
import textwrap
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Optional

import requests
from neo4j.exceptions import Neo4jError

# ---------------------------------------------------------------------
# Reuse the existing, working retrieval pipeline. Importing this module
# does NOT execute anything -- qdrant_to_neo4j_similarity.py only runs
# its pipeline under `if __name__ == "__main__":`.
# ---------------------------------------------------------------------
import qdrant_to_neo4j_similarity as qn


# ==================================================================
# CONFIGURATION
# ==================================================================
LLM_MODEL = "llama3.2:3b"
TOP_N_CASES_FOR_ANSWER = 4          # central + up to 3 related, from retrieval
MAX_CHUNK_CHARS_FOR_LLM = 800        # per-chunk cap sent to the LLM (was 1200)
MAX_CHUNKS_PER_CASE = 2              # candidate chunks pulled per case (was 3)
MAX_TOTAL_EVIDENCE_ITEMS = 8         # global cap on evidence items sent to the LLM
MAX_GRAPH_NEIGHBORS = 5              # internal/debug only, never sent to LLM
OLLAMA_GENERATE_TIMEOUT = 180         # seconds; safety net, not the primary fix
OLLAMA_NUM_PREDICT = 700              # bounds response length -> bounds latency
OLLAMA_NUM_CTX = 4096                 # explicit context window

# --- Revision 4: fallback/retry tuning -----------------------------------
FALLBACK_SNIPPET_CHARS = 320          # per-evidence-excerpt cap shown in the
                                       # deterministic fallback's case-law text
FALLBACK_MAX_SNIPPETS_PER_CASE = 2    # matches MAX_CHUNKS_PER_CASE

# --- Revision 6, FIX A: evidence relevance floor --------------------------
# Tuned against the pipeline's confirmed cosine-similarity ranges
# (~0.69 baseline match, ~0.80 strong match -- see project notes). A chunk
# below this score is treated as too weak to safely hand to the LLM as
# supporting evidence, even if there's still room in MAX_TOTAL_EVIDENCE_ITEMS.
MIN_EVIDENCE_SCORE = 0.72

# --- Revision 7: evidence-starvation backstop -----------------------------
# If the MIN_EVIDENCE_SCORE floor leaves fewer than this many candidate
# chunks, backfill with the next-best (still ranked, just below-floor)
# chunks rather than letting the model work from a near-empty evidence set.
# A model given only 1 evidence item has nothing to synthesize or contrast
# against and tends to lose track of the required section structure.
MIN_EVIDENCE_ITEMS_BACKFILL_FLOOR = 3

NOT_AVAILABLE = "(not available)"
INSUFFICIENT_EVIDENCE_TEXT = "The retrieved evidence is insufficient to determine this point."

log = qn.log  # reuse the same logger/formatting as the retrieval script


# ==================================================================
# STEP 1 -- FULL CHUNK TEXT (untruncated), built from qn's raw results
# ==================================================================
def collect_full_chunk_text(raw_results: list, case_ids: set) -> dict:
    """For each case_id we care about, collect its matched chunks with
    FULL (capped, not 200-char-preview) text, straight from the raw
    Qdrant points. Returns {case_id: [ {chunk_id, score, text}, ... ]},
    each list sorted by score descending."""
    by_case: dict[str, list] = {}
    for point in raw_results:
        payload = point.payload or {}
        case_id = payload.get("case_id")
        if case_id not in case_ids:
            continue
        text = (payload.get("chunk_text", "") or "")[:MAX_CHUNK_CHARS_FOR_LLM]
        by_case.setdefault(case_id, []).append({
            "chunk_id": payload.get("chunk_id", ""),
            "score": point.score,
            "text": text,
        })
    for case_id in by_case:
        by_case[case_id].sort(key=lambda c: c["score"], reverse=True)
    return by_case


# ==================================================================
# STEP 2 -- READ-ONLY GRAPH CONTEXT (additive, no writes)
# ==================================================================
def get_case_graph_metadata(driver, case_id: str) -> Optional[dict]:
    """Read-only lookup of a Case node's own properties. Does not modify
    anything."""
    query = """
    MATCH (c:Case {case_id: $case_id})
    OPTIONAL MATCH (c)-[:APPLIES]->(s:LawSection)
    OPTIONAL MATCH (c)-[:HAS_TOPIC]->(t:Topic)
    OPTIONAL MATCH (c)-[:DECIDED_BY]->(j:Judge)
    OPTIONAL MATCH (c)-[:HEARD_IN]->(court:Court)
    RETURN c.case_id AS case_id,
           c.case_number AS case_number,
           c.year AS year,
           c.case_type AS case_type,
           collect(DISTINCT s.name) AS sections_cited,
           collect(DISTINCT t.name) AS topics,
           collect(DISTINCT j.name) AS judges,
           collect(DISTINCT court.name) AS courts
    """
    try:
        with driver.session() as session:
            record = session.run(query, case_id=case_id).single()
            if not record:
                return None
            return {
                "case_id": record["case_id"],
                "case_number": record["case_number"],
                "year": record["year"],
                "case_type": record["case_type"],
                "sections_cited": [s for s in record["sections_cited"] if s],
                "topics": [t for t in record["topics"] if t],
                "judges": [j for j in record["judges"] if j],
                "courts": [c for c in record["courts"] if c],
            }
    except Neo4jError as e:
        log.warning(f"Could not read graph metadata for {case_id}: {e}")
        return None


def get_existing_similar_cases(driver, case_id: str, limit: int) -> list:
    """Read-only lookup of SIMILAR_TO edges that ALREADY exist for this
    case. INTERNAL/DEBUG USE ONLY. Never passed to the LLM and never
    rendered as a citation."""
    query = """
    MATCH (a:Case {case_id: $case_id})-[r:SIMILAR_TO]->(b:Case)
    RETURN b.case_id AS case_id, b.case_number AS case_number,
           r.score AS score
    ORDER BY r.score DESC
    LIMIT $limit
    """
    try:
        with driver.session() as session:
            result = session.run(query, case_id=case_id, limit=limit)
            return [
                {"case_id": r["case_id"], "case_number": r["case_number"], "score": r["score"]}
                for r in result
            ]
    except Neo4jError as e:
        log.warning(f"Could not read existing SIMILAR_TO edges for {case_id}: {e}")
        return []


# ==================================================================
# STEP 3 -- RETRIEVAL ORCHESTRATION (read-only; unchanged logic)
# ==================================================================
class RetrievalError(Exception):
    """Raised for any pre-flight or retrieval failure, with a message
    already safe to show the user."""
    pass


def retrieve_evidence(legal_query: str, central_identifier: Optional[str] = None) -> dict:
    """Runs the retrieval side of the pipeline and returns a dict:
        {
          "central_case": CaseHit,
          "central_auto_selected": bool,
          "related_cases": [CaseHit, ...],
          "chunk_text_by_case": {case_id: [ {chunk_id, score, text}, ... ]},
          "graph_metadata": {case_id: {...} | None},
          "existing_similar": [ {case_id, case_number, score}, ... ],  # debug only
        }
    Raises RetrievalError with a user-safe message on any failure."""
    if not qn.check_ollama():
        raise RetrievalError("The legal research assistant is temporarily unavailable. Please try again shortly.")

    qdrant_client = qn.get_qdrant_client()
    if qdrant_client is None:
        raise RetrievalError("The legal research assistant is temporarily unavailable. Please try again shortly.")

    neo4j_driver = qn.get_neo4j_driver()
    if neo4j_driver is None:
        raise RetrievalError("The legal research assistant is temporarily unavailable. Please try again shortly.")

    try:
        query_vector = qn.get_query_embedding(legal_query)
        if query_vector is None:
            raise RetrievalError("The legal research assistant could not process this query. Please try rephrasing it.")

        raw_results = qn.search_qdrant(qdrant_client, query_vector)
        if not raw_results:
            raise RetrievalError(
                "No relevant Pakistani case law could be found for this query. "
                "Try rephrasing it with more specific legal terms."
            )

        cases = qn.group_by_case_id(raw_results)
        ranked_cases = sorted(cases.values(), key=lambda c: c.score, reverse=True)
        if not ranked_cases:
            raise RetrievalError(
                "No relevant Pakistani case law could be found for this query. "
                "Try rephrasing it with more specific legal terms."
            )

        case_ids = [c.case_id for c in ranked_cases]
        neo4j_info = qn.find_cases_in_neo4j(neo4j_driver, case_ids)

        for c in ranked_cases:
            info = neo4j_info.get(c.case_id)
            c.case_number = info["case_number"] if info and info["exists"] else None

        found_cases = [c for c in ranked_cases if neo4j_info.get(c.case_id, {}).get("exists")]
        if not found_cases:
            raise RetrievalError(
                "No verified Pakistani case law could be found for this query. "
                "Try rephrasing it with more specific legal terms."
            )

        # --- Central case selection (mirrors qn.main()'s logic) ---------
        auto_selected = False
        if central_identifier:
            central = qn.resolve_central_case(neo4j_driver, central_identifier)
            if central is None:
                raise RetrievalError(f"No case could be found matching '{central_identifier}'.")
            central_case_id = central["case_id"]
            central_lookup = next((c for c in found_cases if c.case_id == central_case_id), None)
            if central_lookup is None:
                central_case = qn.CaseHit(
                    case_id=central_case_id,
                    score=0.0,
                    matched_chunk_id="",
                    matched_chunk_score=0.0,
                    case_number=central["case_number"],
                )
            else:
                central_case = central_lookup
        else:
            central_case = found_cases[0]
            auto_selected = True

        related_cases = [c for c in found_cases if c.case_id != central_case.case_id][: TOP_N_CASES_FOR_ANSWER - 1]

        evidence_case_ids = {central_case.case_id} | {c.case_id for c in related_cases}
        chunk_text_by_case = collect_full_chunk_text(raw_results, evidence_case_ids)

        graph_metadata = {cid: get_case_graph_metadata(neo4j_driver, cid) for cid in evidence_case_ids}
        existing_similar = get_existing_similar_cases(neo4j_driver, central_case.case_id, MAX_GRAPH_NEIGHBORS)

        return {
            "central_case": central_case,
            "central_auto_selected": auto_selected,
            "related_cases": related_cases,
            "chunk_text_by_case": chunk_text_by_case,
            "graph_metadata": graph_metadata,
            "existing_similar": existing_similar,
        }
    finally:
        neo4j_driver.close()


# ==================================================================
# EVIDENCE-ID GROUNDING
# --------------------------------------------------------------------
# Every chunk given to the LLM gets a stable, Python-assigned ID (E1, E2,
# ...) permanently mapped to its real case_id. The LLM only ever sees the
# anonymous evidence text -- never case identity -- so it structurally
# cannot invent or misattribute a citation to the wrong case. Citation
# validity is therefore never checked by string-comparing case numbers;
# it is checked purely by whether the tag the model used is one Python
# actually handed out. This design is UNCHANGED in Revision 6.
#
# Selection is GLOBAL, not a flat per-case cap: candidate chunks from
# every retrieved case are pooled and the top-scoring MAX_TOTAL_EVIDENCE_
# ITEMS overall are kept (with the central case's best chunk guaranteed a
# slot). Revision 6, FIX A: candidates are now also filtered by
# MIN_EVIDENCE_SCORE before that pooling happens, so a weakly-matched
# chunk can no longer occupy a slot just because MAX_TOTAL_EVIDENCE_ITEMS
# hadn't been reached yet.
# ==================================================================
@dataclass
class EvidenceItem:
    eid: str           # "E1", "E2", ...
    case_id: str        # permanent, real case_id -- never altered after assignment
    chunk_id: str
    score: float
    text: str            # only ever shown to the LLM, never to the end user


def build_evidence_items(evidence: dict) -> list:
    central = evidence["central_case"]
    related = evidence["related_cases"]
    chunks_by_case = evidence["chunk_text_by_case"]

    candidates = []  # list of (case, chunk_dict)
    for case in [central] + related:
        for ch in chunks_by_case.get(case.case_id, [])[:MAX_CHUNKS_PER_CASE]:
            candidates.append((case, ch))

    # --- Revision 6, FIX A: drop weakly-matched chunks before selection ---
    # The central case's guarantee below still applies even if its best
    # chunk is under the floor, so a weak central match still produces an
    # answer -- this only stops WEAK RELATED chunks from being pooled in
    # alongside genuinely relevant ones.
    strong_candidates = [(case, ch) for case, ch in candidates if ch["score"] >= MIN_EVIDENCE_SCORE]

    # --- Revision 7: backfill if the floor left too few items -----------
    # A near-empty evidence set (e.g. 1 chunk) starves the model of enough
    # material to synthesize/contrast, and in practice correlated with the
    # model losing track of the required section structure. If the floor
    # leaves fewer than MIN_EVIDENCE_ITEMS_BACKFILL_FLOOR items, add back
    # the next-best below-floor candidates (still score-ranked, just not
    # discarded) until that minimum is reached or candidates run out. This
    # never removes the quality floor for cases where enough strong
    # evidence already exists -- it only engages when strong_candidates is
    # thin.
    if len(strong_candidates) < MIN_EVIDENCE_ITEMS_BACKFILL_FLOOR:
        weak_candidates = sorted(
            [c for c in candidates if c[1]["score"] < MIN_EVIDENCE_SCORE],
            key=lambda pair: pair[1]["score"],
            reverse=True,
        )
        needed = MIN_EVIDENCE_ITEMS_BACKFILL_FLOOR - len(strong_candidates)
        strong_candidates += weak_candidates[:needed]

    candidates = strong_candidates
    candidates.sort(key=lambda pair: pair[1]["score"], reverse=True)

    selected = []
    central_included = False
    for case, ch in candidates:
        if len(selected) >= MAX_TOTAL_EVIDENCE_ITEMS:
            break
        selected.append((case, ch))
        if case.case_id == central.case_id:
            central_included = True

    if not central_included:
        central_chunks = chunks_by_case.get(central.case_id, [])[:1]
        if central_chunks:
            if len(selected) >= MAX_TOTAL_EVIDENCE_ITEMS:
                selected = selected[:-1]
            selected.insert(0, (central, central_chunks[0]))

    items = []
    counter = 1
    for case, ch in selected:
        items.append(EvidenceItem(
            eid=f"E{counter}",
            case_id=case.case_id,
            chunk_id=ch["chunk_id"],
            score=ch["score"],
            text=ch["text"],
        ))
        counter += 1
    return items


SINGLE_TAG_PATTERN = re.compile(r"\[\s*[Ee](\d+)\s*\]")
# --------------------------------------------------------------------
# Revision 13 FIX: previously this only matched the bare tag group
# itself, e.g. "[E4]" or "[E1][E3]". When the model wrote its OWN
# enclosing parentheses directly around a tag with nothing else inside,
# e.g. "([E4])", only the "[E4]" portion got replaced -- the model's own
# "(" and ")" were left untouched, so natural_citation()'s own
# parenthesized output ended up nested inside them, producing double
# parens like "( (W.P.No.2571 of 2021, Lahore High Court, 2022))".
#
# Fix: try to match a FULLY parenthesized tag-only group first (both the
# "(" and ")" immediately and only wrap the tag(s), nothing else inside),
# and only fall back to the bare tag group if that doesn't apply. This
# is intentionally NOT "\(?...\)?" (independently optional on each side)
# -- that would let a stray trailing ")" from an unrelated enclosing
# phrase (e.g. "(see [E4])", where the "(" belongs to "see" not the tag)
# get silently swallowed while its matching "(" is left behind,
# unbalancing the parentheses in the opposite direction. Requiring BOTH
# sides together (or neither) avoids that.
# --------------------------------------------------------------------
CITATION_GROUP_PATTERN = re.compile(
    r"\((?:\[\s*[Ee]\d+\s*\]\s*)+\)|(?:\[\s*[Ee]\d+\s*\]\s*)+"
)

# --------------------------------------------------------------------
# Backstop for BARE (unbracketed) evidence mentions, e.g. a model writing
# "...as shown in E6..." instead of "...as shown in [E6]...". Requires a
# word boundary on both sides so it doesn't match inside unrelated tokens.
# Matches are normalized into bracket form BEFORE section parsing, so they
# flow through the exact same valid/invalid tag pipeline as a properly
# bracketed tag: a genuine tag becomes a real citation, an invented one
# (an E-number beyond what was actually supplied) is silently dropped and
# counted -- never left as raw "E6" text in the user-facing answer.
# --------------------------------------------------------------------
BARE_TAG_PATTERN = re.compile(r"(?<![\[\w])[Ee](\d{1,3})(?![\]\w])")


def normalize_bare_evidence_mentions(text: str) -> tuple[str, int]:
    """Wraps any bare 'E<number>' mention in brackets so it is handled by
    the normal tag pipeline. Returns (normalized_text, count_normalized)."""
    count = 0

    def _wrap(match: re.Match) -> str:
        nonlocal count
        count += 1
        return f"[E{match.group(1)}]"

    new_text = BARE_TAG_PATTERN.sub(_wrap, text)
    return new_text, count


# --------------------------------------------------------------------
# Revision 8 -- NEW: backstop for MULTIPLE evidence tags written inside a
# SINGLE bracket, e.g. "[E2, E4]" instead of "[E2][E4]". Observed in a
# live run: the model wrote this comma form for a combined citation, and
# because it doesn't match SINGLE_TAG_PATTERN (which requires the bracket
# to contain nothing but "[Ee]<digits>"), it wasn't recognized as a tag
# at all -- so it was neither converted to a citation nor dropped as
# invalid, and leaked straight through to the user as raw "[E2, E4]"
# bracket syntax. This is exactly the class of technical leak this file
# exists to prevent (see REVISION 1). Fix: split any such bracket into
# individual single-tag brackets BEFORE the normal tag pipeline runs, so
# each one is then handled exactly as if the model had written it that
# way to begin with -- either converted to a real citation (if valid) or
# silently dropped (if invalid), never left as raw syntax.
# --------------------------------------------------------------------
MULTI_TAG_BRACKET_PATTERN = re.compile(
    r"\[\s*[Ee]\d+(?:\s*[,;/]\s*[Ee]\d+)+\s*\]"
)


def normalize_multi_tag_brackets(text: str) -> tuple[str, int]:
    """Rewrites a single bracket containing multiple comma/semicolon/
    slash-separated evidence tags, e.g. "[E2, E4]", into individual
    adjacent bracket tags "[E2][E4]". Returns (normalized_text,
    count_of_brackets_split)."""
    count = 0

    def _split(match: re.Match) -> str:
        nonlocal count
        count += 1
        ids = re.findall(r"[Ee]\d+", match.group(0))
        return "".join(f"[{i.upper()}]" for i in ids)

    new_text = MULTI_TAG_BRACKET_PATTERN.sub(_split, text)
    return new_text, count


# --------------------------------------------------------------------
# Revision 4 -- structural pre-check used purely to decide whether a
# retry / deterministic fallback is needed. It does not do any fuzzy or
# lexical matching of claims to evidence text on purpose: this codebase
# deliberately avoids string-similarity-based grounding (see the module
# docstring's "Note on citation-string normalization") because it is
# unreliable and easy to fool. The only reliable signal for "is this
# claim tied to a real, retrieved case" is still the same structural
# [Ex] -> case_id mapping used everywhere else in this file. This
# function just asks: did the model use ANY of those real tags at all?
# --------------------------------------------------------------------
def count_valid_tags(text: Optional[str], valid_ids: set) -> int:
    """Counts how many bracketed (or bare, once normalized) [Ex] mentions in
    `text` are actual valid evidence tags. Used both for the boolean
    has_any_valid_tag() check and for INFO-level tracing so it's visible at
    runtime exactly how many tags each generation attempt produced --
    without ever logging the evidence text itself."""
    if not text:
        return 0
    normalized, _ = normalize_multi_tag_brackets(text)
    normalized, _ = normalize_bare_evidence_mentions(normalized)
    found_ids = [f"E{n}" for n in SINGLE_TAG_PATTERN.findall(normalized)]
    return sum(1 for eid in found_ids if eid in valid_ids)


def has_any_valid_tag(text: Optional[str], valid_ids: set) -> bool:
    return count_valid_tags(text, valid_ids) > 0


# ==================================================================
# Revision 6, FIX A -- OFF-CORPUS / FOREIGN-LAW LEAKAGE DETECTION
# --------------------------------------------------------------------
# The model is told (in build_prompt()) to never reference non-Pakistani
# law unless it appears verbatim in the evidence text. That's a prompt
# instruction, not an enforced guarantee -- this is the structural
# backstop, checked purely against the literal evidence text supplied for
# THIS query, so it can never falsely flag a term the evidence itself
# actually contains (e.g. a judgment comparatively citing foreign law).
# Not exhaustive by design: it targets the specific, recurring failure
# mode of a small local model pulling in familiar US/UK/pre-partition
# terms from its training data, not a full legal-NER system.
# ==================================================================
FOREIGN_LAW_PATTERN = re.compile(
    r"\b("
    r"Bail Reform Act|"
    r"U\.?S\.?C\.?|United States Code|"
    r"Federal Rules of (?:Criminal|Civil) Procedure|"
    r"English Law|Indian Penal Code|IPC \d|"
    r"Common Law of England|"
    r"Miranda (?:rights|warning)"
    r")\b",
    re.IGNORECASE,
)


def contains_foreign_law_reference(text: Optional[str], evidence_items: list) -> list:
    """Returns the list of denylisted foreign-law terms found in `text`
    that do NOT appear (case-insensitively) anywhere in the evidence text
    the model was actually given. An empty list means either no
    denylisted term was used, or every term used genuinely came from the
    evidence itself (in which case it isn't a hallucination and isn't
    flagged)."""
    if not text:
        return []
    evidence_blob = " ".join(it.text for it in evidence_items).lower()
    hits = []
    for m in FOREIGN_LAW_PATTERN.finditer(text):
        term = m.group(0)
        if term.lower() not in evidence_blob:
            hits.append(term)
    return hits


# --------------------------------------------------------------------
# Revision 11 -- NEW: known-hallucination-phrase denylist, same
# evidence-membership check as FOREIGN_LAW_PATTERN above, extended to
# specific phrases CONFIRMED (via a full-corpus scan of all 55,829
# Qdrant points, 0 matches) to never appear in the retrieved evidence at
# all. This is not general fuzzy content-matching -- the module
# deliberately avoids that (see "Note on citation-string normalization")
# because it's unreliable and easy to fool. This is the opposite: a
# small, specific, EMPIRICALLY VERIFIED list of phrases the small model
# is known to insert from its own memorized training data (well-known
# legal formulas it has clearly seen many times) and then attach to
# whatever nearby valid [Ex] tag happens to be available -- passing the
# structural grounding check even though that tag's actual evidence text
# has nothing to do with the claim. Observed live: this exact custody-law
# formula appeared, each time attached to a different real citation,
# across bail, eviction, and criminal-appeal queries where it had no
# business appearing -- and a full scan confirmed the phrase is not in
# the corpus at all, in any chunk, for any case. Add further confirmed
# phrases here only after the same kind of verification (a corpus-wide
# scan showing zero genuine matches) -- this list should stay small and
# evidence-backed, not a guess at what "sounds hallucinated."
# --------------------------------------------------------------------
KNOWN_HALLUCINATION_PATTERNS = [
    re.compile(
        r"welfare of (?:the |a |)minor(?:'s|) is(?: treated as|) the paramount consideration",
        re.IGNORECASE,
    ),
]


def contains_known_hallucination(text: Optional[str], evidence_items: list) -> list:
    """Same evidence-membership logic as contains_foreign_law_reference(),
    applied to KNOWN_HALLUCINATION_PATTERNS. Returns the list of matched
    phrases that do NOT appear in the evidence text actually given to the
    model for this query."""
    if not text:
        return []
    evidence_blob = " ".join(it.text for it in evidence_items).lower()
    hits = []
    for pattern in KNOWN_HALLUCINATION_PATTERNS:
        for m in pattern.finditer(text):
            term = m.group(0)
            if term.lower() not in evidence_blob:
                hits.append(term)
    return hits


# ==================================================================
# Revision 6, FIX C -- APPLICATION-SECTION OVERASSERTION DETECTION
# --------------------------------------------------------------------
# Lightweight post-generation regex backstop, same pattern as
# strip_leaked_metadata(). Does not rewrite the model's prose (rewriting
# risks distorting meaning) -- it only signals that a clarifying note
# should be appended to LIMITATIONS, exactly like the existing
# invalid_tags handling in build_user_facing_answer().
# ==================================================================
OVERASSERTION_PATTERN = re.compile(
    r"\b("
    r"will (?:succeed|win|be granted|be entitled)|"
    r"is entitled to|"
    r"the court will (?:rule|decide|find)"
    r")\b",
    re.IGNORECASE,
)


def check_overassertion(application_text: str) -> bool:
    """True if the rendered APPLICATION TO THE QUERY section contains
    conclusive ('will succeed', 'is entitled to', ...) rather than
    conditional language about the user's own situation."""
    if not application_text:
        return False
    return bool(OVERASSERTION_PATTERN.search(application_text))


# ==================================================================
# LEAKED-METADATA STRIPPING (hard backstop)
# --------------------------------------------------------------------
# The model is never given case metadata, so it shouldn't be able to
# produce these lines at all. This stays as a backstop in case a model
# ignores instructions and states something that looks like a metadata
# field anyway.
# ==================================================================
_LEAKED_METADATA_LINE = re.compile(
    r"^\s*[-*]?\s*(Case ID|Case Number|Court|Judge|Year|Similarity Score|"
    r"Similarity|Sections Cited|Topics|Matched Chunk|case_id|Source)\s*[:=]",
    re.IGNORECASE,
)


def strip_leaked_metadata(llm_answer: str) -> str:
    kept_lines = [line for line in llm_answer.splitlines() if not _LEAKED_METADATA_LINE.match(line)]
    cleaned = "\n".join(kept_lines)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


# ==================================================================
# SECTION PARSING
# ==================================================================
SECTION_NAMES = [
    "LEGAL ISSUE",
    "ANSWER",
    "RELEVANT LEGAL PRINCIPLES",
    "RELEVANT CASE LAW",
    "APPLICATION TO THE QUERY",
    "LIMITATIONS",
]

# --------------------------------------------------------------------
# Revision 12 FIX -- REGRESSION FROM REVISION 7: making the colon
# mandatory (Revision 7) fixed the "heading + inline content on one
# line" leak, but broke a DIFFERENT, more common model behavior: writing
# a heading bare, alone on its own line, with NO colon at all (e.g.
# "ANSWER" by itself, content starting on the next line). Since that no
# longer matched at all, the entire REST of the model's output -- every
# subsequent intended section -- got swallowed into whichever earlier
# heading WAS recognized (observed live: an entire six-section answer
# collapsed into the LEGAL ISSUE section alone). Fix: colon is optional
# again (`:?`), restoring recognition of colon-less bare headings, while
# still NOT requiring the rest of the line to be empty (no trailing `$`
# anchor) -- so the Revision 7 fix (heading + inline content on the same
# line, e.g. "LIMITATIONS: None") is also still correctly handled. Both
# failure modes are now covered by one pattern.
# --------------------------------------------------------------------
_SECTION_PATTERN = re.compile(
    r"^[ \t]*[#*]{0,3}[ \t]*("
    + "|".join(re.escape(n) for n in SECTION_NAMES)
    + r")[ \t]*:?[ \t]*",
    re.IGNORECASE | re.MULTILINE,
)


def parse_llm_sections(text: str) -> dict:
    """Splits the model's raw output into {SECTION_NAME: content}. Robust
    to minor markdown decoration around headings, and to a heading and its
    content appearing on the same line. If the same section name is
    matched more than once (e.g. the model stray-echoed a heading mid-
    section, as with the "LIMITATIONS: None" case above), the LAST
    occurrence wins -- which also correctly discards whatever spurious
    text sat between the first, unintended heading and the real one, since
    that text now belongs to the (now correctly split-off) intervening
    section content rather than lingering in the previous section. If
    parsing finds nothing (model didn't follow the structure at all),
    returns an empty dict and the caller falls back to treating the whole
    text as the answer."""
    matches = list(_SECTION_PATTERN.finditer(text))
    sections = {}
    for i, m in enumerate(matches):
        name = m.group(1).upper()
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        sections[name] = text[start:end].strip()
    return sections


# ==================================================================
# METADATA -> NATURAL CITATION FORMATTING
# ==================================================================
MIN_VALID_YEAR = 1947
MAX_VALID_YEAR = datetime.now().year


def is_valid_year(value) -> bool:
    if value is None:
        return False
    try:
        year_int = int(str(value).strip()[:4])
    except (ValueError, TypeError):
        return False
    return MIN_VALID_YEAR <= year_int <= MAX_VALID_YEAR


# --------------------------------------------------------------------
# Revision 9 -- NEW: plausibility guard for case_number values, same
# spirit as is_valid_year() above. Observed in a live run: a Neo4j Case
# node's case_number field contained "construction of the BHU. It was
# not" -- a chunk-text sentence fragment that leaked into that field
# upstream (a known failure mode; see the module docstring's earlier
# extractor.py / case_number history) -- and it was displayed to the
# user as if it were a real citation label. A garbled case_number is
# worse than a missing one: it reads as a plausible-looking but fake
# reference. This does not touch Neo4j or fix the underlying data --
# it only stops the presentation layer from trusting a field that
# clearly isn't a case number, falling back to the existing
# "(a retrieved <Court> case)" text instead, exactly as if that field
# had been empty.
# --------------------------------------------------------------------
_PROSE_FUNCTION_WORDS = re.compile(
    r"\b(?:the|and|was|is|are|were|not|that|which|has|have|had|been|this|"
    r"these|those|from|with|for|will|shall|would|could|should)\b",
    re.IGNORECASE,
)


def is_plausible_case_number(value) -> bool:
    """Rejects values that read as prose (a chunk-text leak) rather than
    a genuine case reference like "W.P. No.2050 of 2024" or "PLD 2022 SC
    85". A real case number is short, generally contains at least one
    digit, and doesn't read like a sentence fragment. This is
    intentionally a coarse heuristic -- it only needs to catch obvious
    prose leaks, not validate correct case-number formatting."""
    if value is None:
        return False
    text = str(value).strip()
    if not text or len(text) > 80:
        return False
    if not re.search(r"\d", text):
        return False  # genuine case numbers virtually always contain digits
    if len(_PROSE_FUNCTION_WORDS.findall(text)) >= 2:
        return False  # two or more common English function words together
                       # strongly suggests a sentence fragment, not a citation
    return True


def authoritative_meta(case: "qn.CaseHit", graph_meta: Optional[dict]) -> dict:
    """Neo4j graph metadata (looked up by THIS case's own case_id) is
    authoritative; the Qdrant payload's fields are only a fallback for
    when the graph node has nothing. An invalid/garbage year is never
    shown as if it were confirmed.

    Revision 6, FIX B: now also resolves `case_number` the same way
    court/year already were -- graph value first, CaseHit's own value
    only as a fallback. Previously `case_number` was never read from
    graph_meta here at all, so natural_citation() had to fall back to
    the weaker "(a retrieved <Court> case)" text more often than the
    data actually required."""
    court = NOT_AVAILABLE
    year = NOT_AVAILABLE
    case_number = NOT_AVAILABLE
    if graph_meta:
        if graph_meta.get("courts"):
            court = graph_meta["courts"][0]
        if is_valid_year(graph_meta.get("year")):
            year = str(int(str(graph_meta["year"]).strip()[:4]))
        # Revision 9: only trust a graph-sourced case_number if it
        # actually looks like one -- see is_plausible_case_number().
        if graph_meta.get("case_number") and is_plausible_case_number(graph_meta["case_number"]):
            case_number = graph_meta["case_number"]
    if court == NOT_AVAILABLE and getattr(case, "court", None):
        court = case.court
    if year == NOT_AVAILABLE and is_valid_year(getattr(case, "year", None)):
        year = str(int(str(case.year).strip()[:4]))
    if case_number == NOT_AVAILABLE and getattr(case, "case_number", None):
        # Revision 9: same guard applied to the CaseHit-side fallback.
        if is_plausible_case_number(case.case_number):
            case_number = case.case_number
    return {"court": court, "year": year, "case_number": case_number}


def natural_citation(case: "qn.CaseHit", meta: dict) -> Optional[str]:
    """Renders a verified, human-readable inline citation, e.g.
    '(W.P. No.61653 of 2020, Lahore High Court, 2022)'. Falls back to
    '(a retrieved <Court> case)' if only the court is verified. Returns
    None (no citation at all -- never a guess) if nothing is verified.

    Revision 6, FIX B: prefers the graph-verified meta["case_number"]
    over qn.display_label(case) (which only reflected the Qdrant-side
    CaseHit). Falls back to qn.display_label(case) only if the graph
    genuinely had no case_number, so this never removes a citation that
    was previously available -- it only adds a stronger source to check
    first."""
    graph_case_number = meta.get("case_number", NOT_AVAILABLE)
    if graph_case_number and graph_case_number != NOT_AVAILABLE:
        label = graph_case_number
    else:
        # Revision 9: qn.display_label() reads from the CaseHit directly,
        # a path authoritative_meta() doesn't cover -- apply the same
        # prose-fragment guard here before trusting it as a citation.
        candidate_label = qn.display_label(case)
        label = candidate_label if is_plausible_case_number(candidate_label) else None
    has_number = bool(label) and label != NOT_AVAILABLE
    has_court = meta["court"] != NOT_AVAILABLE
    has_year = meta["year"] != NOT_AVAILABLE

    if has_number:
        parts = [label]
        if has_court:
            parts.append(meta["court"])
        if has_year:
            parts.append(meta["year"])
        return "(" + ", ".join(parts) + ")"
    if has_court:
        return f"(a retrieved {meta['court']} case)"
    return None


def convert_citations_to_natural(text: str, evidence_by_id: dict, valid_ids: set, meta_cache: dict):
    """Replaces each group of adjacent [Ex] tags with natural, verified
    citation text (deduplicating repeated same-case tags within a group).
    Unknown/invalid tags -- including bare mentions that were normalized
    into bracket form upstream -- are silently dropped (never shown as raw
    syntax, never converted into a fabricated citation) and recorded for
    the internal debug info. Returns (clean_text, used_case_ids, invalid_tags)."""
    used_case_ids = set()
    invalid_tags = set()

    def _replace_group(match: re.Match) -> str:
        group_text = match.group(0)
        eids = [f"E{n}" for n in SINGLE_TAG_PATTERN.findall(group_text)]
        citations = []
        seen_case_ids = []
        for eid in eids:
            if eid not in valid_ids:
                invalid_tags.add(eid)
                continue
            item = evidence_by_id[eid]
            if item.case_id in seen_case_ids:
                continue
            seen_case_ids.append(item.case_id)
            used_case_ids.add(item.case_id)
            case, meta = meta_cache[item.case_id]
            cite = natural_citation(case, meta)
            if cite:
                citations.append(cite)
        if not citations:
            return ""
        return " " + " ".join(citations)

    new_text = CITATION_GROUP_PATTERN.sub(_replace_group, text)
    new_text = re.sub(r"\s+([.,;:])", r"\1", new_text)
    # Revision 8 -- FIX: when the model writes two citation groups back to
    # back with no space (e.g. "...evidence[E1][E3]and[E5][E7], which..."),
    # the rendered text ends up as "...evidence (cite1)and(cite2), which..."
    # -- readable but visibly glitchy. Ensure a space wherever a letter
    # directly abuts a parenthesis in either direction.
    new_text = re.sub(r"([A-Za-z])\(", r"\1 (", new_text)
    new_text = re.sub(r"\)([A-Za-z])", r") \1", new_text)
    new_text = re.sub(r"[ \t]{2,}", " ", new_text)
    new_text = re.sub(r"\n{3,}", "\n\n", new_text)
    return new_text.strip(), used_case_ids, invalid_tags


# ==================================================================
# PROMPT CONSTRUCTION
# --------------------------------------------------------------------
# The model receives ONLY anonymous evidence text under [E1], [E2], ...
# It is never given case numbers, courts, judges, years, or similarity
# scores, so it cannot invent, mix up, or restate any of that -- Python
# is the sole source of that information in the final output. Synthesis
# across evidence items is explicitly encouraged, "insufficient evidence"
# is explicitly scoped to genuine gaps only, case-specific facts are kept
# separate from the user's hypothetical facts (via a domain-neutral
# example, so the model doesn't over-generalize caution to one specific
# legal topic), and the model is told which common topics are almost
# always answerable so it doesn't default to hedging on them.
#
# Revision 4: added the optional `strengthen` flag, used only for the
# single automatic retry when the first generation contained zero valid
# [Ex] tags, a foreign-law leak (Revision 6), or a known hallucination-
# prone phrase not present in the evidence (Revision 11). It changes
# nothing about the evidence or the six-section structure -- it only
# adds one more blunt instruction block at the end, tailored to whichever
# problem triggered the retry.
#
# Revision 6, FIX C: the APPLICATION TO THE QUERY instructions now
# require three explicitly labeled parts (Held / Principle / Relevance to
# your question) per point, so the "case fact vs. principle vs. relevance
# to the user" distinction is structural in the model's own output, not
# just implied by prose instructions.
# ==================================================================
def build_prompt(
    legal_query: str,
    evidence_items: list,
    strengthen: bool = False,
    foreign_law_terms: Optional[list] = None,
) -> str:
    valid_ids_block = "\n".join(f"[{it.eid}]" for it in evidence_items)
    evidence_lines = []
    for it in evidence_items:
        snippet = it.text.replace("\n", " ").strip()
        evidence_lines.append(f"[{it.eid}] \"{snippet}\"")
    evidence_text = "\n\n".join(evidence_lines)

    system_instructions = textwrap.dedent(f"""\
        You are a legal RESEARCH assistant for Pakistani case law. You are NOT
        a lawyer, and your output is NOT legal advice.

        You are given EVIDENCE ITEMS below, each a short excerpt from a real,
        retrieved Pakistani court judgment, labeled [E1], [E2], etc. You are
        not told which specific case, court, judge, or year each item comes
        from -- only the text itself.

        VALID EVIDENCE TAGS (the only tags you may ever use):
        {valid_ids_block}

        HOW TO USE THE EVIDENCE:
        - Read all the evidence items and identify what they actually
          establish about the user's question.
        - SYNTHESIZE: if multiple evidence items support or restate the same
          point, combine them into one clear explanation rather than
          repeating each one separately. Combining consistent information
          that is actually present in the evidence is expected -- it is not
          the same as inventing information, and is exactly what a good
          legal researcher does.
        - Tag each distinct claim once, right after you make it, e.g.
          "...the welfare of the minor is the paramount consideration
          [E2]." You do NOT need to tag every sentence individually. You may
          combine tags where several items support the same point, e.g.
          [E1][E3].
        - Every reference to evidence MUST use one of the exact bracketed
          tags listed above (e.g. "[E2]"). NEVER refer to an evidence item
          in plain prose without brackets (e.g. never write "as shown in
          E2" or "the evidence in E2" -- always "[E2]"). NEVER invent, or
          refer to, an evidence number that is not in the VALID EVIDENCE
          TAGS list above (for example, if the highest valid tag is [E5],
          never write or imply "[E6]" or "E6" anywhere).
        - Never invent any other citation format -- do not write
          "(Source: ...)", a case number, a section number, a court name, or
          a year anywhere; you do not have this information.
        - Use ONLY the evidence text given. Do not add outside knowledge of
          Pakistani law, statutes, or legal principles that isn't present in
          the evidence.

        CASE FACTS VS. THE USER'S HYPOTHETICAL -- DO NOT MIX THESE:
        - Evidence items describe what happened in a SPECIFIC retrieved
          case (its facts, its parties, what that court decided on those
          facts). The user's query describes a DIFFERENT, hypothetical
          situation. These are never the same thing unless the user's query
          itself stated that fact.
        - In RELEVANT CASE LAW, describe case-specific facts and outcomes as
          belonging to that retrieved case only -- e.g. "In one retrieved
          case, the court found that a specific fact was established before
          ruling in the plaintiff's favour [E4]."
        - Only a LEGAL PRINCIPLE (a general rule the evidence states applies
          beyond one case) may be applied directly to the user's facts. A
          single case's specific facts, findings, or outcome may not.
        - Example of what NOT to do: user asks about a boundary dispute
          between neighbours; evidence [E4] is from a retrieved case where
          the plaintiff had already obtained a survey report confirming
          encroachment before the court ruled in their favour. WRONG:
          "Since the survey report already confirms the encroachment, the
          user's claim will succeed [E4]." CORRECT: "In the retrieved case,
          the court relied on a survey report confirming encroachment
          before ruling in the plaintiff's favour [E4]. The evidence does
          not establish that a survey report exists in the user's
          situation; whether one does is not addressed here."
        - This rule applies equally to every legal topic -- custody, bail,
          inheritance, tenancy, or anything else. It is about keeping ANY
          one case's own specific facts from being stated as the user's
          facts; it does not mean you should hesitate to explain what the
          evidence generally establishes.

        COMMON TOPICS THAT ARE ALMOST ALWAYS ANSWERABLE:
        - Topics such as child custody, bail (including anticipatory/
          pre-arrest bail under Sections 497/498/498-A CrPC), inheritance
          and succession among legal heirs, and tenancy/eviction disputes
          come up often in this evidence corpus. If ANY evidence item
          discusses the general subject area of the user's question -- even
          using different wording, a different fact pattern, or a related
          statutory provision -- treat that as usable evidence and produce
          a real, synthesized answer. Do not default to the insufficiency
          sentence for these topics merely because no single evidence item
          is a word-for-word match to the user's exact question.

        WHEN (AND ONLY WHEN) TO SAY THE EVIDENCE IS INSUFFICIENT:
        - Use the exact sentence "{INSUFFICIENT_EVIDENCE_TEXT}" ONLY for a
          specific sub-point that the evidence genuinely does not address at
          all.
        - Do NOT use this sentence as a general disclaimer, and do NOT use it
          just because the evidence doesn't cover every possible nuance of
          the user's exact fact pattern. If the evidence substantially
          speaks to the user's question -- even via a general principle
          rather than the user's precise facts -- explain what it says and
          connect it to the query. That is a real, useful answer.
        - Example: a user asks about a child-custody dispute, and one
          evidence item states that the welfare and best interests of the
          minor are the paramount consideration in custody matters. This IS
          usable evidence -- explain that principle and how it bears on the
          question. Do not say the evidence is insufficient merely because
          it doesn't describe this user's exact family situation.

        WHAT TO PRIORITIZE:
        - Focus on the evidence items that most directly address the user's
          question. You do not need to discuss every evidence item equally
          -- it is fine to rely mainly on the one or two most relevant items
          and only briefly note or skip items that turn out not to be
          relevant.

        WHAT NOT TO DO:
        - Never state or imply a specific case number, citation, court name,
          judge name, section number, or year -- you don't have this
          information.
        - Never state a legal principle, holding, or rule -- however
          well-known or standard it may be -- unless the evidence items
          given to you actually say it. You may already know common legal
          principles from your training; that knowledge must NOT be used
          here. If a principle you recognize as generally true is not
          actually present in the evidence below, do not write it, and do
          not attach an [Ex] tag to it just because a tag is available --
          a tag must genuinely support the specific sentence it's attached
          to, not merely be nearby.
        - Never give personalized legal advice or tell the user what they
          "should" do; frame this as research findings, not counsel.
        - Never reference any non-Pakistani statute, code, act, or legal
          system (for example, U.S. or U.K. federal/state law, a "Bail
          Reform Act," Indian Penal Code, foreign case law, or any law from
          another country) unless that exact term appears verbatim in the
          evidence text below. Pakistani case law only cites Pakistani law
          (PPC, CrPC, the Constitution, provincial acts, etc.) -- if the
          evidence doesn't name a specific law, do not supply one from
          outside knowledge, and do not guess at a foreign equivalent, even
          one that sounds similar (e.g. do not substitute "Indian Penal
          Code" for "Pakistan Penal Code").

        Structure your answer in EXACTLY these six sections, each heading
        alone on its own line, in this order:

        LEGAL ISSUE:
        ANSWER:
        RELEVANT LEGAL PRINCIPLES:
        RELEVANT CASE LAW:
        APPLICATION TO THE QUERY:
        LIMITATIONS:

        - LEGAL ISSUE: one or two sentences framing the legal question raised
          by the user's query.
        - ANSWER: the most important section. Directly and concisely answer
          the user's question in plain professional language, grounded in
          the cited evidence.
        - RELEVANT LEGAL PRINCIPLES: explain, in your own words, the legal
          principles the evidence establishes that bear on this issue, each
          tagged.
        - RELEVANT CASE LAW: explain what the retrieved judgments actually
          say or hold, each point tagged. Do not name or describe the cases
          yourself -- Python will attach the verified citation automatically
          wherever you place a valid tag.
        - APPLICATION TO THE QUERY: for each point you raise here, write
          exactly three labeled parts, each starting on its own line:
            "Held: " -- one sentence, ONLY what the retrieved case actually
              decided on ITS OWN facts, tagged with the supporting [Ex].
              Never state this as true of the user's situation.
            "Principle: " -- one sentence, the general legal rule extracted
              from that holding, stated independently of the specific
              case's facts (no tag needed here if it repeats the tag
              already given under "Held:").
            "Relevance to your question: " -- one or two sentences using
              ONLY conditional language ("if similar facts existed," "this
              principle would suggest," "by extension," "applying this
              principle here would mean") about how the principle may bear
              on the user's question. NEVER use conclusive language here
              (e.g. never write "will succeed," "is entitled to," "the
              court will rule") -- you do not know the user's actual facts
              well enough to conclude anything.
          If the evidence doesn't support a clean Held/Principle split for
          a point, omit that point rather than forcing a weak one.
        - LIMITATIONS: honestly note any genuine gap, conflict, or
          insufficiency -- only for points the evidence truly does not
          cover.

        Do not mention evidence IDs, "the evidence items," "retrieved
        chunks," Qdrant, Neo4j, or any technical/retrieval language in your
        prose -- write as a research answer, not a system description.
    """)

    if strengthen:
        system_instructions += textwrap.dedent(f"""
        MANDATORY RETRY NOTICE -- READ CAREFULLY:
        Your previous attempt at this exact question was rejected. Read the
        specific reason(s) below and correct them in THIS attempt -- these
        are hard, non-negotiable requirements, not style preferences.
        """)

        if not foreign_law_terms:
            # Original Revision 4 wording -- zero-tag failure.
            system_instructions += textwrap.dedent("""
        REASON: your previous attempt contained ZERO valid bracketed
        evidence tags anywhere in it.
        - Every sentence you write in RELEVANT LEGAL PRINCIPLES and RELEVANT
          CASE LAW MUST end with at least one bracketed tag drawn from the
          VALID EVIDENCE TAGS list above, e.g.:
          "The welfare of the minor is treated as the paramount
          consideration in custody matters [E2]."
        - At least one sentence in ANSWER should also carry a tag if it
          relies on the evidence.
        - Do not write a single paragraph of legal analysis without a tag
          attached to it.
        - If you genuinely cannot connect any evidence item to the question
          at all, you must still tag whichever evidence items are even
          loosely on-topic rather than omitting tags entirely -- omitting
          tags is worse than an imperfect tag, because an answer with zero
          tags cannot be shown to the user at all and will be discarded.
        """)
        else:
            # Revision 6/11 -- the model inserted a flagged term/phrase
            # not actually present in its evidence. Could be a foreign-law
            # leak (FIX A) or a known hallucination-prone phrase (FIX K in
            # Revision 11) -- named dynamically either way, since small
            # models respond far better to "you wrote X, that's wrong"
            # than to a generic rule stated once at the top.
            leaked = ", ".join(f'"{t}"' for t in foreign_law_terms)
            system_instructions += textwrap.dedent(f"""
        REASON: your previous attempt included the following statement(s),
        which did NOT appear anywhere in the evidence items given to you:
        {leaked}.
        - Every claim you make must come from the evidence text itself, not
          from general legal knowledge you may already have. Even a
          well-known, commonly-cited legal principle must not be stated
          UNLESS the evidence items actually say it -- if you recognize a
          principle but it isn't in the evidence given to you this time, do
          not write it.
        - Do not name ANY statute, act, or legal system -- Pakistani or
          foreign -- unless its exact name is present, verbatim, in the
          evidence text. If the evidence doesn't name a specific law, write
          about the principle in general terms instead of naming a law at
          all.
        - Do not substitute a foreign law you recognize for a Pakistani one
          that sounds similar (e.g. do not write "Indian Penal Code" or a
          U.S./U.K. statute name under any circumstances in this answer).
        - Tag every claim with the [Ex] tag for the evidence item it
          actually came from -- do not attach a tag to a sentence just
          because that tag is available; the tag must genuinely support
          that specific sentence's content.
        """)

    user_content = textwrap.dedent(f"""\
        USER'S LEGAL QUERY:
        {legal_query}

        EVIDENCE ITEMS (the ONLY information you may use; tag each claim with
        its supporting [Ex] tag(s)):

        {evidence_text}

        Now write the answer following the required section structure and
        rules exactly. Give a real, useful, synthesized answer whenever the
        evidence substantially supports one -- reserve
        "{INSUFFICIENT_EVIDENCE_TEXT}" for genuine gaps only. Keep case-
        specific facts and general legal principles clearly separated per
        the CASE FACTS VS. THE USER'S HYPOTHETICAL rule above, use the
        Held / Principle / Relevance to your question structure for
        APPLICATION TO THE QUERY, and only use the exact bracketed tags
        listed above -- never a bare or invented evidence number, and never
        a non-Pakistani law name that isn't verbatim in the evidence.
    """)

    return system_instructions + "\n" + user_content


# ==================================================================
# OLLAMA LLM CALL (graceful timeout handling, bounded generation)
# ==================================================================
def call_ollama_llm(prompt: str, model: str = LLM_MODEL, timeout: int = OLLAMA_GENERATE_TIMEOUT) -> Optional[str]:
    try:
        resp = requests.post(
            f"{qn.OLLAMA_URL}/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.1,
                    "num_predict": OLLAMA_NUM_PREDICT,  # bounds response length -> bounds latency
                    "num_ctx": OLLAMA_NUM_CTX,
                },
            },
            timeout=timeout,
        )
        resp.raise_for_status()
        data = resp.json()
        text = data.get("response")
        if not text:
            log.error("Ollama returned no 'response' field for the answer generation call.")
            return None
        return text.strip()
    except requests.exceptions.Timeout:
        log.error(f"Ollama call timed out after {timeout}s (model={model}).")
        return None
    except requests.exceptions.RequestException as e:
        log.error(f"Failed to get an answer from Ollama ({model}): {e}")
        return None
    except Exception as e:  # never let an unexpected error escape as a raw traceback
        log.error(f"Unexpected error calling Ollama ({model}): {e}")
        return None


# ==================================================================
# CASE / METADATA LOOKUP CACHE
# --------------------------------------------------------------------
# Revision 4: pulled out of build_user_facing_answer() so the
# LLM-grounded path and the deterministic fallback path build citations
# through the exact same code -- there is no separate "fallback citation
# logic" that could drift out of sync with the real one, or accidentally
# show something unverified.
# ==================================================================
def build_meta_cache(evidence: dict) -> dict:
    central = evidence["central_case"]
    related = evidence["related_cases"]
    graph_meta = evidence["graph_metadata"]
    case_lookup = {c.case_id: c for c in [central] + related}
    return {
        cid: (case_lookup[cid], authoritative_meta(case_lookup[cid], graph_meta.get(cid)))
        for cid in case_lookup
    }


def build_supporting_references(evidence: dict, evidence_items: list) -> str:
    """Revision 5 -- Independent of whether the LLM remembered to tag
    anything: lists the real, Neo4j-verified citations for the cases that
    were actually retrieved and handed to the model as evidence, built
    entirely in Python from the same natural_citation()/build_meta_cache()
    used everywhere else in this file. Never includes scores, chunk ids,
    Qdrant ids, or case_id filenames -- only the same natural_citation()
    strings shown elsewhere. Returns "" if no case in the evidence has any
    verifiable citation.

    Revision 10 FIX: dedupe by the RENDERED CITATION TEXT, not just by
    case_id. Two different cases can both lack a verified case_number and
    therefore both fall back to the same generic "(a retrieved <Court>
    case)" text -- without this, they showed up as two identical-looking
    bullet points, which reads as a bug/redundancy to the user even
    though they were technically two different case_ids."""
    meta_cache = build_meta_cache(evidence)
    seen_case_ids = []
    seen_citation_text = set()
    lines = []
    for it in evidence_items:
        if it.case_id in seen_case_ids:
            continue
        seen_case_ids.append(it.case_id)
        case, meta = meta_cache[it.case_id]
        cite = natural_citation(case, meta)
        if cite:
            line = cite.strip("()")
            if line in seen_citation_text:
                continue
            seen_citation_text.add(line)
            lines.append(line)
    return "\n".join(f"- {line}" for line in lines)


# ==================================================================
# FINAL USER-FACING ANSWER ASSEMBLY (normal, LLM-grounded path)
# ==================================================================
def _default_legal_issue(legal_query: str) -> str:
    return f"This research question concerns: {legal_query.strip()}"


def build_user_facing_answer(legal_query: str, evidence: dict, llm_answer_raw: str):
    """Returns (clean_answer_text, debug_info). Converts every valid [Ex]
    citation into natural, verified inline text, drops invalid/unknown
    tags (including bare mentions normalized into bracket form) without
    ever showing them or fabricating a citation, and applies a single
    whole-answer grounding check (not a per-sentence one).

    By the time this is called from answer_legal_query(), llm_answer_raw
    is already known to contain at least one valid [Ex] tag and to be
    free of flagged foreign-law terms (both checks now happen upstream,
    with a retry + deterministic-fallback path if either fails -- see
    has_any_valid_tag(), contains_foreign_law_reference(), and
    build_deterministic_fallback_answer()). The "not grounded" branch
    below is kept as a safety net only.

    Revision 6, FIX C: also runs check_overassertion() on the rendered
    APPLICATION TO THE QUERY section and, if triggered, appends a
    clarifying note to LIMITATIONS (same mechanism already used for
    invalid_tags) rather than rewriting the model's own prose."""
    evidence_items = build_evidence_items(evidence)
    evidence_by_id = {it.eid: it for it in evidence_items}
    valid_ids = set(evidence_by_id)

    meta_cache = build_meta_cache(evidence)

    cleaned_raw = strip_leaked_metadata(llm_answer_raw)
    cleaned_raw, multi_tag_count = normalize_multi_tag_brackets(cleaned_raw)
    cleaned_raw, bare_tag_count = normalize_bare_evidence_mentions(cleaned_raw)
    sections = parse_llm_sections(cleaned_raw)
    if not sections:
        # Model didn't follow the structure -- fall back gracefully rather
        # than showing nothing or raising an error.
        sections = {"ANSWER": cleaned_raw}

    # --------------------------------------------------------------------
    # Revision 12, FIX L -- MISPLACED APPLICATION CONTENT: observed
    # recurring in live runs -- the model sometimes writes the required
    # Held: / Principle: / Relevance to your question: structure (meant
    # for APPLICATION TO THE QUERY) inside RELEVANT CASE LAW instead,
    # leaving APPLICATION TO THE QUERY empty or just the standard
    # insufficiency sentence. Detected by the presence of all three
    # labels together in RELEVANT CASE LAW's raw content while
    # APPLICATION TO THE QUERY's raw content is empty or that sentence.
    # This operates on RAW (pre-citation-conversion) section text, moving
    # whole raw text between sections, so citation conversion below still
    # runs once per (now-corrected) section exactly as normal.
    # --------------------------------------------------------------------
    raw_case_law = sections.get("RELEVANT CASE LAW", "")
    raw_application = sections.get("APPLICATION TO THE QUERY", "")
    _application_is_empty = (
        not raw_application.strip()
        or raw_application.strip().rstrip(".").lower() == INSUFFICIENT_EVIDENCE_TEXT.rstrip(".").lower()
    )
    if (
        _application_is_empty
        and re.search(r"\bHeld:\s", raw_case_law, re.IGNORECASE)
        and re.search(r"\bPrinciple:\s", raw_case_law, re.IGNORECASE)
        and re.search(r"\bRelevance to your question:\s", raw_case_law, re.IGNORECASE)
    ):
        split_at = re.search(r"(?=\bHeld:\s)", raw_case_law, re.IGNORECASE)
        if split_at:
            sections["RELEVANT CASE LAW"] = raw_case_law[: split_at.start()].strip()
            sections["APPLICATION TO THE QUERY"] = raw_case_law[split_at.start():].strip()

    used_case_ids = set()
    invalid_tags = set()
    rendered = {}
    for name in SECTION_NAMES:
        raw_section = sections.get(name, "")
        text, u, inv = convert_citations_to_natural(raw_section, evidence_by_id, valid_ids, meta_cache)
        used_case_ids |= u
        invalid_tags |= inv
        rendered[name] = text

    grounded = len(used_case_ids) > 0

    legal_issue = rendered["LEGAL ISSUE"] or _default_legal_issue(legal_query)
    answer = rendered["ANSWER"] or INSUFFICIENT_EVIDENCE_TEXT
    principles = rendered["RELEVANT LEGAL PRINCIPLES"] or INSUFFICIENT_EVIDENCE_TEXT
    case_law = rendered["RELEVANT CASE LAW"] or "No specific retrieved case law could be clearly identified for this answer."
    application = rendered["APPLICATION TO THE QUERY"] or INSUFFICIENT_EVIDENCE_TEXT
    limitations = rendered["LIMITATIONS"]

    # --- Revision 6, FIX C: overassertion backstop --------------------
    overasserted = check_overassertion(application)
    if overasserted:
        note = (
            "This section has been flagged as potentially overstating what the "
            "evidence supports; treat any statement about a specific outcome for "
            "your situation as illustrative only, not a conclusion."
        )
        limitations = (limitations + "\n\n" + note).strip() if limitations else note

    if not grounded:
        answer = INSUFFICIENT_EVIDENCE_TEXT
        principles = INSUFFICIENT_EVIDENCE_TEXT
        case_law = "No specific retrieved case law could be verified for this answer."
        application = INSUFFICIENT_EVIDENCE_TEXT
        note = (
            "No part of the generated answer could be verified against the retrieved "
            "Pakistani case law, so a full research answer could not be produced for "
            "this query. Try rephrasing it with more specific legal terms."
        )
        limitations = (limitations + "\n\n" + note).strip() if limitations else note
    elif invalid_tags:
        note = "Some statements in this answer referenced evidence that could not be verified and were treated as unsupported."
        limitations = (limitations + "\n\n" + note).strip() if limitations else note

    final_text = (
        f"LEGAL ISSUE:\n{legal_issue}\n\n"
        f"ANSWER / SUMMARY:\n{answer}\n\n"
        f"RELEVANT LEGAL PRINCIPLES:\n{principles}\n\n"
        f"RELEVANT CASE LAW:\n{case_law}\n\n"
        f"APPLICATION TO THE QUERY:\n{application}\n\n"
        f"LIMITATIONS:\n{limitations if limitations else 'None identified.'}"
    )

    # Revision 5: always append a Python-built, independently-verified
    # reference list for the cases that were actually retrieved as
    # evidence -- this does NOT depend on the model having tagged
    # anything correctly, so the user still sees real case references
    # even on a run where the model's own inline citations were sparse.
    if grounded:
        supporting_refs = build_supporting_references(evidence, evidence_items)
        if supporting_refs:
            final_text += f"\n\nSOURCES CONSULTED:\n{supporting_refs}"

    debug_info = {
        "mode": "llm_grounded",
        "grounded": grounded,
        "used_case_ids": sorted(used_case_ids),
        "invalid_evidence_tags": sorted(invalid_tags),
        "bare_evidence_labels_normalized": bare_tag_count,
        "multi_tag_brackets_split": multi_tag_count,
        "valid_tags_found_in_raw_answer": count_valid_tags(llm_answer_raw, valid_ids),
        "overassertion_flagged": overasserted,
        "evidence_items": [asdict(it) for it in evidence_items],
        "central_case_id": evidence["central_case"].case_id,
        "central_auto_selected": evidence["central_auto_selected"],
        "existing_similar_background_only": evidence["existing_similar"],
    }
    return final_text, debug_info


# ==================================================================
# DETERMINISTIC, EVIDENCE-BASED FALLBACK (Revision 4)
# --------------------------------------------------------------------
# Used when the LLM could not (after one retry) produce any usable [Ex]
# tags, produced a foreign-law leak on both attempts (Revision 6), or
# Ollama failed/timed out on both attempts. No LLM call happens in this
# function at all -- it assembles the six required sections directly from
# the SAME evidence_items and the SAME natural_citation()/
# authoritative_meta() functions the normal path uses, so citations here
# are exactly as verified as in the normal path. It only ever echoes
# trimmed excerpts of the retrieved evidence text itself -- it never
# invents a case number, court, year, holding, section, or fact, and
# therefore cannot introduce a foreign-law reference either.
# ==================================================================
def clean_evidence_snippet(text: str, max_chars: int = FALLBACK_SNIPPET_CHARS) -> str:
    """Collapse whitespace and trim an evidence chunk to a short,
    sentence-boundary-aware excerpt suitable for direct display.

    Revision 10 FIX: also strips stray standalone underscore runs left
    over from OCR (used in the original PDFs as underline markers, e.g.
    "...offence charged._ When a person..."), which otherwise showed up
    verbatim in this direct-extraction fallback path."""
    snippet = " ".join(text.split())
    snippet = re.sub(r"\s*_{1,}\s*", " ", snippet)
    snippet = re.sub(r"[ \t]{2,}", " ", snippet).strip()
    if len(snippet) <= max_chars:
        return snippet
    truncated = snippet[:max_chars]
    last_period = truncated.rfind(". ")
    if last_period > max_chars * 0.4:
        return truncated[: last_period + 1]
    last_space = truncated.rfind(" ")
    if last_space > 0:
        truncated = truncated[:last_space]
    return truncated.rstrip(",;: ") + "..."


def build_deterministic_fallback_answer(
    legal_query: str,
    evidence: dict,
    evidence_items: list,
    reason_note: str,
):
    """Returns (clean_answer_text, debug_info) built entirely from the
    already-retrieved evidence, without any LLM call. Never exposes
    scores, chunk ids, case_id filenames, or [Ex] tags. Its APPLICATION
    TO THE QUERY text is fully conditional/non-conclusory by construction
    (see application_text below), so it needed no change for Revision 6
    FIX C, and it contains no LLM-generated language at all, so it cannot
    trigger the FIX A foreign-law check either."""
    meta_cache = build_meta_cache(evidence)

    # Group the already-selected, already-ranked evidence items by case,
    # preserving the order in which each case first appears (i.e. by its
    # best-scoring chunk, since evidence_items is already score-sorted).
    by_case: dict[str, list] = {}
    case_order: list = []
    for it in evidence_items:
        if it.case_id not in by_case:
            by_case[it.case_id] = []
            case_order.append(it.case_id)
        by_case[it.case_id].append(it)

    case_law_lines = []
    principle_bullets = []
    for case_id in case_order:
        case, meta = meta_cache[case_id]
        cite = natural_citation(case, meta)
        items = by_case[case_id][:FALLBACK_MAX_SNIPPETS_PER_CASE]
        combined_snippet = " ".join(clean_evidence_snippet(it.text) for it in items)
        if cite:
            # cite is parenthesized (e.g. "(Cr.B.A.No.326 of 2023, Lahore High
            # Court, 2023)") for inline use elsewhere; strip the parens here
            # since it's the grammatical subject of this sentence.
            case_law_lines.append(f"In {cite.strip('()')}, the retrieved judgment states: {combined_snippet}")
        else:
            case_law_lines.append(
                f"In a retrieved case (its citation could not be independently verified), "
                f"the judgment states: {combined_snippet}"
            )
        principle_bullets.append(combined_snippet)

    legal_issue = _default_legal_issue(legal_query)

    answer_text = (
        "Based directly on the Pakistani case law retrieved for this query, the following "
        "points emerge from the judgments themselves:\n\n"
        + "\n\n".join(f"- {b}" for b in principle_bullets)
    )

    principles_text = "\n\n".join(f"- {b}" for b in principle_bullets)

    case_law_text = "\n\n".join(case_law_lines)

    application_text = (
        "The passages above are shown as they appear in the retrieved judgments. Whether, "
        "and how, these principles apply to your specific facts has not been independently "
        "assessed here -- they should be read as the general position taken in these cases, "
        "not as a determination of your situation. A case's own specific facts (as opposed "
        "to a general legal principle it states) should not be assumed to also be true of "
        "your situation."
    )

    limitations_lines = [
        "This answer was assembled directly from the retrieved case excerpts because the "
        "research assistant's underlying language model could not produce a synthesized "
        "narrative answer for this query. The excerpts above are shown largely as retrieved "
        "and have not been further interpreted, combined, or explained in the model's own "
        "words.",
        reason_note,
        "This is legal research information, not legal advice.",
    ]
    limitations_text = "\n\n".join(l for l in limitations_lines if l)

    final_text = (
        f"LEGAL ISSUE:\n{legal_issue}\n\n"
        f"ANSWER / SUMMARY:\n{answer_text}\n\n"
        f"RELEVANT LEGAL PRINCIPLES:\n{principles_text}\n\n"
        f"RELEVANT CASE LAW:\n{case_law_text}\n\n"
        f"APPLICATION TO THE QUERY:\n{application_text}\n\n"
        f"LIMITATIONS:\n{limitations_text}"
    )

    debug_info = {
        "mode": "deterministic_fallback",
        "grounded": True,
        "used_case_ids": case_order,
        "evidence_items": [asdict(it) for it in evidence_items],
        "central_case_id": evidence["central_case"].case_id,
        "central_auto_selected": evidence["central_auto_selected"],
        "existing_similar_background_only": evidence["existing_similar"],
        "fallback_reason": reason_note,
    }
    return final_text, debug_info


def user_facing_error(message: str) -> str:
    """Same six-section shape, used only for genuine RETRIEVAL failures
    (no evidence was ever found) so the caller never sees a raw
    exception, traceback, or inconsistent format. Not used for "the LLM
    didn't tag its answer" or "the LLM leaked a foreign-law term" -- both
    of those go through build_deterministic_fallback_answer() instead,
    because retrieval itself succeeded in those cases."""
    return (
        "LEGAL ISSUE:\nUnable to process this request.\n\n"
        f"ANSWER / SUMMARY:\n{message}\n\n"
        "RELEVANT LEGAL PRINCIPLES:\nNot applicable.\n\n"
        "RELEVANT CASE LAW:\nNot applicable.\n\n"
        "APPLICATION TO THE QUERY:\nNot applicable.\n\n"
        "LIMITATIONS:\nThis response could not be grounded in retrieved case law due to a system issue."
    )


# ==================================================================
# ORCHESTRATION (Revision 6 -- retry now also triggers on foreign-law leaks)
# --------------------------------------------------------------------
# Flow:
#   1. Retrieve evidence. Genuine retrieval failure (nothing found at
#      all) -> user_facing_error(), unchanged.
#   2. Generate with the normal prompt.
#   3. If generation failed (timeout/error), produced zero valid tags, OR
#      (Revision 6) leaked a flagged foreign-law term: retry ONCE with
#      build_prompt(..., strengthen=True, foreign_law_terms=...).
#   4. If the retry succeeded, has a valid tag, AND is free of foreign-law
#      leaks -> proceed normally.
#   5. Otherwise (retry also failed, still untagged, or still leaking) ->
#      Python builds the answer itself from the retrieved evidence via
#      build_deterministic_fallback_answer(). The user still gets a real,
#      evidence-based answer; only the LIMITATIONS section transparently
#      notes that this is a direct extract rather than a full synthesis.
#   6. As a final safety net, if the normal LLM-grounded path somehow
#      still comes back ungrounded (debug_info["grounded"] is False),
#      swap in the deterministic fallback rather than showing the old
#      wipe text.
# ==================================================================
def answer_legal_query(
    legal_query: str,
    central_identifier: Optional[str] = None,
    model: str = LLM_MODEL,
    ollama_timeout: int = OLLAMA_GENERATE_TIMEOUT,
    return_debug: bool = False,
):
    """Full pipeline: retrieve -> select evidence -> call LLM (with one
    retry if untagged or foreign-law-leaking) -> validate -> format,
    falling back to a deterministic evidence-based answer rather than an
    "insufficient evidence" message whenever real evidence was actually
    retrieved. Never raises to the caller.

    Returns the clean answer string, or (answer, debug_info) if
    return_debug=True.
    """
    try:
        evidence = retrieve_evidence(legal_query, central_identifier)
    except RetrievalError as e:
        text = user_facing_error(str(e))
        return (text, {"error": str(e)}) if return_debug else text
    except Exception as e:
        log.error(f"Unexpected error during retrieval: {e}")
        text = user_facing_error("An unexpected error occurred while retrieving case law. Please try again.")
        return (text, {"error": "unexpected_retrieval_error"}) if return_debug else text

    evidence_items = build_evidence_items(evidence)
    if not evidence_items:
        text = user_facing_error("No usable case-law text could be retrieved for this query.")
        return (text, {"error": "no_evidence_items"}) if return_debug else text

    valid_ids = {it.eid for it in evidence_items}
    case_ids_in_evidence = sorted({it.case_id for it in evidence_items})
    log.info(
        f"Evidence ready: {len(evidence_items)} chunk(s) across {len(case_ids_in_evidence)} "
        f"case(s) (central={evidence['central_case'].case_id}, "
        f"auto_selected={evidence['central_auto_selected']})."
    )

    # --- Attempt 1: normal prompt -----------------------------------
    prompt = build_prompt(legal_query, evidence_items)
    llm_answer = call_ollama_llm(prompt, model=model, timeout=ollama_timeout)
    attempt1_tags = count_valid_tags(llm_answer, valid_ids)
    attempt1_foreign = contains_foreign_law_reference(llm_answer, evidence_items)
    attempt1_hallucination = contains_known_hallucination(llm_answer, evidence_items)
    attempt1_unsupported = attempt1_foreign + attempt1_hallucination
    log.info(
        f"Attempt 1: {'no response (timeout/error)' if llm_answer is None else f'{attempt1_tags} valid tag(s) found'}"
        + (f"; unsupported phrase(s): {attempt1_unsupported}" if attempt1_unsupported else "")
        + "."
    )

    used_retry = False
    fallback_reason = None

    if attempt1_tags == 0 or attempt1_unsupported:
        # Either the call failed/timed out, the model forgot every tag,
        # or (Revision 6/11) it inserted a foreign-law term or a known
        # hallucination-prone phrase not actually present in its
        # evidence. Either way: one automatic retry with a strengthened
        # prompt before giving up on the LLM entirely. NOTE: this branch
        # can only ever be entered when attempt 1 already has a problem --
        # a clean, tagged, unsupported-phrase-free attempt 1 never reaches
        # retry, so retry cannot turn a working answer into a failing one.
        if llm_answer is None:
            log.warning("First generation attempt failed/timed out; retrying once with a strengthened prompt.")
        elif attempt1_unsupported:
            log.warning(f"First generation attempt inserted unsupported phrase(s) {attempt1_unsupported}; retrying once with a strengthened prompt.")
        else:
            log.warning("First generation attempt had zero valid evidence tags; retrying once with a strengthened prompt.")

        retry_prompt = build_prompt(
            legal_query, evidence_items, strengthen=True, foreign_law_terms=attempt1_unsupported or None
        )
        llm_answer_retry = call_ollama_llm(retry_prompt, model=model, timeout=ollama_timeout)
        used_retry = True
        attempt2_tags = count_valid_tags(llm_answer_retry, valid_ids)
        attempt2_foreign = contains_foreign_law_reference(llm_answer_retry, evidence_items)
        attempt2_hallucination = contains_known_hallucination(llm_answer_retry, evidence_items)
        attempt2_unsupported = attempt2_foreign + attempt2_hallucination
        log.info(
            f"Attempt 2 (retry): {'no response (timeout/error)' if llm_answer_retry is None else f'{attempt2_tags} valid tag(s) found'}"
            + (f"; unsupported phrase(s): {attempt2_unsupported}" if attempt2_unsupported else "")
            + "."
        )

        if attempt2_tags > 0 and not attempt2_unsupported:
            llm_answer = llm_answer_retry
        else:
            # Retry also failed, still untagged, or still inserted an
            # unsupported phrase. Do NOT tell the user the evidence was
            # insufficient -- it wasn't; the synthesis step was the
            # problem, not the retrieval. Build the answer directly from
            # the retrieved evidence instead.
            if llm_answer_retry is None:
                fallback_reason = (
                    "The system's automatic answer-writing step did not complete in time on two "
                    "attempts, so this summary was generated directly from the retrieved case "
                    "excerpts instead of a fully synthesized narrative."
                )
            elif attempt2_unsupported:
                fallback_reason = (
                    "The system's automatic answer-writing step included a statement that could not "
                    "be verified against the retrieved cases (even after one retry), so this summary "
                    "was generated directly from the retrieved case excerpts instead of a fully "
                    "synthesized narrative."
                )
            else:
                fallback_reason = (
                    "The system's automatic answer-writing step could not produce a response with "
                    "verifiable citations to the retrieved cases (even after one retry), so this "
                    "summary was generated directly from the retrieved case excerpts instead of a "
                    "fully synthesized narrative."
                )
            final_text, debug_info = build_deterministic_fallback_answer(
                legal_query, evidence, evidence_items, fallback_reason
            )
            debug_info["used_retry"] = used_retry
            return (final_text, debug_info) if return_debug else final_text

    # --- We now have an llm_answer with at least one valid [Ex] tag and
    # no flagged foreign-law reference -----------------------------------
    try:
        final_text, debug_info = build_user_facing_answer(legal_query, evidence, llm_answer)
    except Exception as e:
        log.error(f"Unexpected error while assembling the final answer: {e}")
        # Retrieval succeeded and we had a tagged model answer; prefer the
        # deterministic fallback over a generic error so the user still
        # gets a real answer from the real evidence.
        fallback_reason = (
            "The system's automatic answer-writing step produced a response that could not be "
            "formatted correctly, so this summary was generated directly from the retrieved case "
            "excerpts instead of a fully synthesized narrative."
        )
        final_text, debug_info = build_deterministic_fallback_answer(
            legal_query, evidence, evidence_items, fallback_reason
        )
        debug_info["used_retry"] = used_retry
        return (final_text, debug_info) if return_debug else final_text

    # --- Safety net: build_user_facing_answer() still reports ungrounded
    # (should be rare given the pre-checks above, e.g. a tag existed only
    # in a section that failed to parse). Prefer the deterministic
    # fallback over the old wipe text.
    if not debug_info.get("grounded", False):
        fallback_reason = (
            "The system's automatic answer-writing step referenced evidence in a way that could "
            "not be verified against the retrieved cases, so this summary was generated directly "
            "from the retrieved case excerpts instead."
        )
        final_text, debug_info = build_deterministic_fallback_answer(
            legal_query, evidence, evidence_items, fallback_reason
        )

    debug_info["used_retry"] = used_retry
    return (final_text, debug_info) if return_debug else final_text


# ==================================================================
# CLI ENTRY POINT
# ==================================================================
def main():
    debug_mode = "--debug" in sys.argv

    print("=" * 60)
    print("Pakistani Legal Research Assistant")
    print("=" * 60)

    print("\nDescribe your legal question (e.g. 'A tenant is being illegally")
    print("evicted by the landlord without proper notice.'):\n")
    legal_query = input("> ").strip()
    if not legal_query:
        print("Please enter a question.")
        sys.exit(1)

    result = answer_legal_query(legal_query, return_debug=debug_mode)
    if debug_mode:
        answer_text, debug_info = result
        print("\n" + answer_text)
        print("\n" + "-" * 60)
        print("[DEBUG INFO -- internal use only]")
        print("-" * 60)
        for key, value in debug_info.items():
            print(f"{key}: {value}")
    else:
        print("\n" + result)


if __name__ == "__main__":
    main()