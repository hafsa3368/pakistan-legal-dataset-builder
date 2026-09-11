"""
evaluate_retrieval.py
======================
Computes Precision@K, Recall@K, and MRR for the retrieval pipeline against
manually-labeled ground truth, and compares two ranking strategies:

  - "hybrid": legal_answer.retrieve_evidence()'s full pipeline (Qdrant
    vector search, re-ranked by 0.7*vector_score + 0.3*graph_score).
  - "vector": Qdrant vector search alone, ranked by cosine similarity
    only -- no Neo4j graph expansion/re-ranking at all. This is the
    baseline the hybrid design is meant to improve on.

HOW TO USE:
  1. Open eval_queries.json. For each query, fill in "relevant_case_ids"
     with the case_id(s) you (or a domain expert) judge as genuinely
     relevant to that query. This is a manual step -- there is no way to
     automate "is this case actually relevant to this legal question".
  2. Run: python evaluate_retrieval.py --mode compare
     (or --mode hybrid / --mode vector to evaluate just one)
  3. Queries with an empty "relevant_case_ids" list are skipped (not yet
     annotated) and reported separately, so partial progress still works.

METRICS:
  Precision@K = (relevant docs in top K) / K
  Recall@K    = (relevant docs in top K) / (total relevant docs for that query)
  MRR         = average of 1/rank_of_first_relevant_result across queries
                (0 if no relevant result was retrieved at all)
"""
import sys
import json
import argparse

sys.path.insert(0, r"D:\hafsa_thesis material\supreme_court_scraper")
import legal_answer as la
import qdrant_to_neo4j_similarity as qn

QUERIES_FILE = "eval_queries.json"
DEFAULT_K = [5, 10]


def get_hybrid_ranked_case_ids(query: str) -> list:
    """Full hybrid pipeline: Qdrant vector search, re-ranked by combined
    vector+graph score. Returns [central_case] + related_cases, in the
    same order legal_answer.py's own re-ranking already produces."""
    evidence = la.retrieve_evidence(query)
    ranked = [evidence["central_case"].case_id]
    ranked += [c.case_id for c in evidence["related_cases"]]
    return ranked


def get_vector_only_ranked_case_ids(query: str) -> list:
    """Baseline: Qdrant vector search ranked by cosine similarity ONLY --
    no Neo4j graph expansion or re-ranking. Uses the exact same Qdrant
    call and case-grouping legal_answer.py's own pipeline uses internally,
    just without the graph re-ranking step applied afterward."""
    query_vector = qn.get_query_embedding(query)
    if query_vector is None:
        return []
    client = qn.get_qdrant_client()
    raw_results = qn.search_qdrant(client, query_vector)
    if not raw_results:
        return []
    cases = qn.group_by_case_id(raw_results)
    ranked_cases = sorted(cases.values(), key=lambda c: c.score, reverse=True)
    return [c.case_id for c in ranked_cases]


RANKERS = {
    "hybrid": get_hybrid_ranked_case_ids,
    "vector": get_vector_only_ranked_case_ids,
}


def precision_at_k(ranked: list, relevant: set, k: int) -> float:
    top_k = ranked[:k]
    if not top_k:
        return 0.0
    hits = sum(1 for cid in top_k if cid in relevant)
    return hits / len(top_k)


def recall_at_k(ranked: list, relevant: set, k: int) -> float:
    if not relevant:
        return 0.0
    top_k = ranked[:k]
    hits = sum(1 for cid in top_k if cid in relevant)
    return hits / len(relevant)


def reciprocal_rank(ranked: list, relevant: set) -> float:
    for i, cid in enumerate(ranked, start=1):
        if cid in relevant:
            return 1.0 / i
    return 0.0


def evaluate(annotated: list, ranker_name: str, ranker_fn, k_values: list, verbose: bool = True) -> dict:
    per_query_results = []
    for it in annotated:
        query = it["query"]
        relevant = set(it["relevant_case_ids"])
        if verbose:
            print(f"[{ranker_name}] Retrieving for: {query!r} ...", flush=True)
        try:
            ranked = ranker_fn(query)
        except Exception as e:
            if verbose:
                print(f"  ERROR during retrieval: {e}")
            continue

        row = {"query": query, "ranked": ranked, "relevant": relevant}
        for k in k_values:
            row[f"precision@{k}"] = precision_at_k(ranked, relevant, k)
            row[f"recall@{k}"] = recall_at_k(ranked, relevant, k)
        row["reciprocal_rank"] = reciprocal_rank(ranked, relevant)
        per_query_results.append(row)

        if verbose:
            print(f"  Ranked (top {max(k_values)}): {ranked[:max(k_values)]}")
            print(f"  Relevant (ground truth): {sorted(relevant)}")
            for k in k_values:
                print(f"  Precision@{k}: {row[f'precision@{k}']:.3f}   Recall@{k}: {row[f'recall@{k}']:.3f}")
            print(f"  Reciprocal Rank: {row['reciprocal_rank']:.3f}")
            print()

    return per_query_results


def summarize(per_query_results: list, k_values: list) -> dict:
    n = len(per_query_results)
    if n == 0:
        return {}
    summary = {}
    for k in k_values:
        summary[f"precision@{k}"] = sum(r[f"precision@{k}"] for r in per_query_results) / n
        summary[f"recall@{k}"] = sum(r[f"recall@{k}"] for r in per_query_results) / n
    summary["mrr"] = sum(r["reciprocal_rank"] for r in per_query_results) / n
    summary["n"] = n
    return summary


def print_summary(title: str, summary: dict, k_values: list):
    print("=" * 50)
    print(title)
    print("=" * 50)
    for k in k_values:
        print(f"Precision@{k}: {summary[f'precision@{k}']:.3f}")
        print(f"Recall@{k}:    {summary[f'recall@{k}']:.3f}")
    print(f"MRR:           {summary['mrr']:.3f}")
    print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, nargs="+", default=DEFAULT_K,
                         help="K values for Precision@K / Recall@K (default: 5 10)")
    parser.add_argument("--mode", choices=["hybrid", "vector", "compare"], default="hybrid",
                         help="Which ranker to evaluate, or 'compare' to run both and show the difference")
    args = parser.parse_args()

    with open(QUERIES_FILE, "r", encoding="utf-8") as f:
        items = json.load(f)

    annotated = [it for it in items if it.get("relevant_case_ids")]
    skipped = [it for it in items if not it.get("relevant_case_ids")]

    if skipped:
        print(f"Skipping {len(skipped)} query(ies) with no relevant_case_ids filled in yet:")
        for it in skipped:
            print(f"  - {it['query']!r}")
        print()

    if not annotated:
        print("No annotated queries to evaluate. Fill in relevant_case_ids in "
              f"{QUERIES_FILE} first.")
        return

    modes = ["hybrid", "vector"] if args.mode == "compare" else [args.mode]
    summaries = {}
    for mode in modes:
        results = evaluate(annotated, mode, RANKERS[mode], args.k)
        summaries[mode] = summarize(results, args.k)
        print_summary(f"{mode.upper()} -- averaged over {summaries[mode].get('n', 0)} queries", summaries[mode], args.k)

    if args.mode == "compare" and summaries.get("hybrid") and summaries.get("vector"):
        print("=" * 50)
        print("HYBRID vs VECTOR-ONLY (difference, hybrid minus vector)")
        print("=" * 50)
        for k in args.k:
            dp = summaries["hybrid"][f"precision@{k}"] - summaries["vector"][f"precision@{k}"]
            dr = summaries["hybrid"][f"recall@{k}"] - summaries["vector"][f"recall@{k}"]
            print(f"Precision@{k}: {dp:+.3f}")
            print(f"Recall@{k}:    {dr:+.3f}")
        dmrr = summaries["hybrid"]["mrr"] - summaries["vector"]["mrr"]
        print(f"MRR:           {dmrr:+.3f}")


if __name__ == "__main__":
    main()
