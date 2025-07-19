# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "requests",
# ]
# ///


import csv

from pathlib import Path

import requests


raw_dir = Path('data/raw')
raw_dir.mkdir(parents=True, exist_ok=True)

count = 0
failed = 0
skipped = 0
with open('data/zenodo_links_filtered.csv', 'r') as f:
    reader = csv.DictReader(f)
    for r in reader:
        url = r['link']
        id = r['id']
        out_file = raw_dir / f'{id}.jpg'
        if out_file.exists():
            print(f'Skipping {id}, already exists')
            count += 1
            skipped += 1
            continue

        print(f'Downloading {id} {count=} {skipped=} {failed=}')
        resp = requests.get(url)
        if not resp.ok:
            print(f'Error downloading {id}: {resp.status_code}')
            failed += 1
            count += 1
            continue

        out_file.write_bytes(resp.content)
        count += 1



