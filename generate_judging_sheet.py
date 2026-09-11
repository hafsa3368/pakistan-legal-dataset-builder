"""
generate_judging_sheet.py
==========================
Runs retrieval (no LLM call) for every query in eval_queries.json, and
writes a human-readable judging sheet showing each candidate case with
enough context (court, case_number, judge, a text snippet) to quickly
decide "is this genuinely relevant to the query?" -- without needing to
search the full 58k-case corpus by hand.

USAGE:
  python generate_judging_sheet.py [--top_k 10]

Output: judging_sheet.txt -- open it, read each candidate's snippet, and
copy the case_id of every genuinely relevant one into the matching
query's "relevant_case_ids" list in eval_queries.json.
"""
import sys
import json
import argparse

sys.path.insert(0, r"D:\hafsa_thesis material\supreme_court_scraper")
import legal_answer as la

QUERIES_FILE = "eval_queries.json"
OUTPUT_FILE = "judging_sheet.txt"


def snippet_for(evidence, case_id, chars=300):
    chunks = evidence["chunk_text_by_case"].get(case_id, [])
    if not chunks:
        return "(no text available)"
    text = " ".join(c.get("text", "") for c in chunks[:2])
    return text[:chars].strip() + ("..." if len(text) > chars else "")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--top_k", type=int, default=10)
    args = parser.parse_args()

    with open(QUERIES_FILE, "r", encoding="utf-8") as f:
        items = json.load(f)

    out_lines = []
    for idx, it in enumerate(items, start=1):
        query = it["query"]
        category = it.get("category", "")
        out_lines.append("=" * 70)
        out_lines.append(f"[{idx}] ({category}) {query}")
        out_lines.append("=" * 70)

        print(f"[{idx}/{len(items)}] Retrieving: {query!r} ...", flush=True)
        try:
            evidence = la.retrieve_evidence(query)
        except Exception as e:
            out_lines.append(f"  ERROR during retrieval: {e}\n")
            continue

        ranked = [evidence["central_case"]] + evidence["related_cases"]
        ranked = ranked[: args.top_k]

        for rank, case in enumerate(ranked, start=1):
            meta = evidence["graph_metadata"].get(case.case_id, {}) or {}
            courts = meta.get("courts") or []
            judges = meta.get("judges") or []
            court = ", ".join(c for c in courts if c) or "?"
            case_number = meta.get("case_number") or "(no case number)"
            judge = ", ".join(j for j in judges if j) or "(no judge)"
            snip = snippet_for(evidence, case.case_id)

            out_lines.append(f"\n  [{rank}] case_id: {case.case_id}")
            out_lines.append(f"      court: {court}   case_number: {case_number}   judge: {judge}")
            out_lines.append(f"      snippet: {snip}")

        out_lines.append("\n  RELEVANT (fill into eval_queries.json): []\n")

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(out_lines))

    print(f"\nDone. Judging sheet written to {OUTPUT_FILE}")
    print("Open it, read each candidate's snippet, and copy the case_id of every")
    print("genuinely relevant one into eval_queries.json's relevant_case_ids list.")


if __name__ == "__main__":
    main()
