"""
test_single_embed.py
Ek specific chunk ka text nikal kar seedha Ollama ko bhejta hai aur
raw response (including Ollama ka asli error message) print karta hai.
Ye humein "500 Server Error" ke peeche ka exact reason dega.

USAGE:
    python test_single_embed.py --json_dir "d:\hafsa_thesis material\supreme_court_scraper\extracted_text_clean" --chunk_id "31084651974ce34a_3"
"""

import json
import argparse
import requests
from pathlib import Path

OLLAMA_URL = "http://localhost:11434/api/embeddings"
EMBED_MODEL = "nomic-embed-text"


def find_chunk_text(json_dir: str, chunk_id_query: str):
    for jf in Path(json_dir).glob("*.json"):
        try:
            with open(jf, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue
        for chunk in data.get("chunks", []):
            chunk_index = chunk.get("chunk_index", 0)
            full_chunk_id = f"{jf.stem}_{chunk_index}"
            if chunk_id_query in full_chunk_id:
                return jf.name, full_chunk_id, chunk.get("text", "")
    return None, None, None


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--json_dir", required=True)
    parser.add_argument("--chunk_id", required=True)
    args = parser.parse_args()

    filename, chunk_id, text = find_chunk_text(args.json_dir, args.chunk_id)
    if text is None:
        print(f"Chunk not found: {args.chunk_id}")
        raise SystemExit(1)

    print(f"Found chunk in {filename} ({chunk_id}), length={len(text)} chars")

    # Test 1: tiny hardcoded string (baseline sanity check)
    print("\n--- Test 1: baseline 'hello world' ---")
    try:
        r = requests.post(OLLAMA_URL, json={"model": EMBED_MODEL, "prompt": "hello world"}, timeout=30)
        print(f"Status: {r.status_code}")
        print(f"Body: {r.text[:500]}")
    except Exception as e:
        print(f"Request failed: {e}")

    # Test 2: the actual problematic chunk, full length
    print("\n--- Test 2: full problematic chunk ---")
    try:
        r = requests.post(OLLAMA_URL, json={"model": EMBED_MODEL, "prompt": text}, timeout=90)
        print(f"Status: {r.status_code}")
        print(f"Body: {r.text[:1000]}")
    except Exception as e:
        print(f"Request failed: {e}")

    # Test 3: first half of the chunk only (binary-search style — helps isolate WHERE in the text the problem is)
    half = text[: len(text) // 2]
    print(f"\n--- Test 3: first half only ({len(half)} chars) ---")
    try:
        r = requests.post(OLLAMA_URL, json={"model": EMBED_MODEL, "prompt": half}, timeout=90)
        print(f"Status: {r.status_code}")
        print(f"Body: {r.text[:500]}")
    except Exception as e:
        print(f"Request failed: {e}")

    # Test 4: second half of the chunk only
    second_half = text[len(text) // 2 :]
    print(f"\n--- Test 4: second half only ({len(second_half)} chars) ---")
    try:
        r = requests.post(OLLAMA_URL, json={"model": EMBED_MODEL, "prompt": second_half}, timeout=90)
        print(f"Status: {r.status_code}")
        print(f"Body: {r.text[:500]}")
    except Exception as e:
        print(f"Request failed: {e}")