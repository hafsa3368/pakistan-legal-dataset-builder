from qdrant_client import QdrantClient
import json

client = QdrantClient('localhost', port=6333)

# Step 1: Detect the filename field in payload automatically
sample, _ = client.scroll(collection_name='legal_chunks', limit=1, with_payload=True)
payload_keys = list(sample[0].payload.keys())
print('Payload fields found:', payload_keys)

# Try common filename field names
candidates = ['source_file', 'filename', 'file', 'json_file', 'doc_id', 'case_id', 'file_name']
field = next((f for f in candidates if f in payload_keys), None)

if not field:
    print('Filename field not auto-detected. Please check payload_keys above and tell me which one holds the filename.')
else:
    print(f'Using field: {field}')

    # Step 2: Scroll through ALL points, collect filenames + count chunks
    all_files = set()
    total_chunks = 0
    offset = None

    while True:
        points, offset = client.scroll(
            collection_name='legal_chunks',
            limit=1000,
            with_payload=[field],
            offset=offset
        )
        for p in points:
            fname = p.payload.get(field)
            if fname:
                all_files.add(fname)
            total_chunks += 1
        if offset is None:
            break

    # Step 3: Load the repaired-files scope list
    with open('repair_metadata_progress.json') as f:
        repaired = json.load(f)
    repaired_set = set(repaired) if isinstance(repaired, list) else set(repaired.keys())

    # Step 4: Compute done vs remaining within scope
    done_in_scope = all_files & repaired_set
    remaining_in_scope = repaired_set - all_files

    print()
    print('Abhi tak:')
    print(f'* Total (all-time): {len(all_files):,} files / {total_chunks:,} chunks embed ho chuki hain.')
    print(f'* Is run ke scope ({len(repaired_set):,} repaired files) mein se: {len(done_in_scope):,} done, {len(remaining_in_scope):,} abhi baqi.')
