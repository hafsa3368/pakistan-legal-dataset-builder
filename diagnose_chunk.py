"""
diagnose_chunk.py
Ek specific chunk_id ka text nikaal kar uska deep inspection karta hai —
taake pata chale Ollama ko crash kyun kar raha hai.

USAGE:
    python diagnose_chunk.py --json_dir "d:\hafsa_thesis material\supreme_court_scraper\extracted_text_clean" --chunk_id "2014LHC5934_3"
python diagnose_chunk.py --json_dir "d:\hafsa_thesis material\supreme_court_scraper\extracted_text_clean" --chunk_id "31084651974ce34a_3"
Note: chunk_id woh partial suffix hai jo file stem ke baad aata hai
(e.g. agar file "LHC_civil_2014_civil_general_lhc_unknown_2014LHC5934.json" hai
aur chunk_index 3 hai, to full chunk_id
"LHC_civil_2014_civil_general_lhc_unknown_2014LHC5934_3" banega).
Ye script filename ya chunk_id suffix, dono se match kar sakta hai —
bas jo string do us se chunk dhoondega.
"""

import json
import argparse
import unicodedata
from pathlib import Path


def analyze_text(text: str):
    length = len(text)
    byte_length = len(text.encode("utf-8"))

    # Control characters (excluding normal whitespace \n \t \r)
    control_chars = [
        (i, ch, hex(ord(ch)))
        for i, ch in enumerate(text)
        if unicodedata.category(ch) == "Cc" and ch not in "\n\t\r"
    ]

    # Zero-width / bidi control characters — ye reversed-Urdu extraction
    # ka classic symptom hote hain (RLM, LRM, RLE, PDF, ALM, ZWJ, ZWNJ, etc.)
    bidi_zero_width = [
        (i, ch, hex(ord(ch)), unicodedata.name(ch, "UNKNOWN"))
        for i, ch in enumerate(text)
        if ord(ch) in (
            0x200B, 0x200C, 0x200D, 0x200E, 0x200F,  # ZWSP, ZWNJ, ZWJ, LRM, RLM
            0x202A, 0x202B, 0x202C, 0x202D, 0x202E,  # LRE, RLE, PDF, LRO, RLO
            0x2066, 0x2067, 0x2068, 0x2069,           # LRI, RLI, FSI, PDI
            0x061C,                                    # Arabic Letter Mark
        )
    ]

    # Null bytes
    null_bytes = text.count("\x00")

    # Surrogate pairs / invalid unicode (would cause encode errors)
    surrogates = [i for i, ch in enumerate(text) if 0xD800 <= ord(ch) <= 0xDFFF]

    # Longest "word" (no-space run) — huge no-space runs can break tokenizers
    words = text.split()
    longest_word = max(words, key=len) if words else ""

    # Rough script detection
    arabic_range_count = sum(1 for ch in text if "\u0600" <= ch <= "\u06FF")
    latin_count = sum(1 for ch in text if ch.isascii() and ch.isalpha())

    print("=" * 60)
    print(f"Text length (chars): {length}")
    print(f"Text length (UTF-8 bytes): {byte_length}")
    print(f"Estimated tokens (chars/4): {length // 4}")
    print(f"Arabic/Urdu-range characters: {arabic_range_count}")
    print(f"Latin/ASCII letters: {latin_count}")
    print(f"Null bytes: {null_bytes}")
    print(f"Surrogate code points: {len(surrogates)} {surrogates[:10] if surrogates else ''}")
    print(f"Control characters (excl. \\n\\t\\r): {len(control_chars)}")
    if control_chars:
        print(f"  First few: {control_chars[:10]}")
    print(f"Bidi/zero-width control characters: {len(bidi_zero_width)}")
    if bidi_zero_width:
        print(f"  First few: {bidi_zero_width[:10]}")
    print(f"Longest whitespace-free run: {len(longest_word)} chars")
    if len(longest_word) > 200:
        print(f"  Preview: {longest_word[:200]}...")
    print("-" * 60)
    print("First 300 chars:")
    print(repr(text[:300]))
    print("-" * 60)
    print("Last 300 chars:")
    print(repr(text[-300:]))
    print("=" * 60)


def find_chunk(json_dir: str, chunk_id_query: str):
    json_files = list(Path(json_dir).glob("*.json"))
    for jf in json_files:
        try:
            with open(jf, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue

        chunks = data.get("chunks", [])
        for chunk in chunks:
            chunk_index = chunk.get("chunk_index", 0)
            full_chunk_id = f"{jf.stem}_{chunk_index}"
            if chunk_id_query in full_chunk_id or chunk_id_query == jf.stem:
                yield jf.name, full_chunk_id, chunk.get("text", "")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inspect a chunk's text for characters that might crash Ollama.")
    parser.add_argument("--json_dir", required=True)
    parser.add_argument("--chunk_id", required=True, help="Full or partial chunk_id / filename to search for")
    args = parser.parse_
    args()

    found_any = False
    for filename, chunk_id, text in find_chunk(args.json_dir, args.chunk_id):
        found_any = True
        print(f"\nFOUND in file: {filename}")
        print(f"chunk_id: {chunk_id}")
        if not text.strip():
            print("  (empty text)")
            continue
        analyze_text(text)

    if not found_any:
        print(f"No chunk found matching '{args.chunk_id}' in {args.json_dir}")