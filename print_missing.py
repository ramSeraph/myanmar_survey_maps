import csv
import re
from pprint import pprint

special_cases = {
    '1099': ['1099_01'],
    '0997': ['0997_13', '0997_14'],
    '1097_10 & 11': ['1097_10', '1097_11'],
    '1393 _ 01&05 & 1493_04&08': ['1393_01', '1393_05', '1493_04', '1493_08'],
    '1798_02+03': ['1798_02', '1798_03'],
}

seen = set()
with open('data/zenodo_links_filtered.csv', 'r') as f:
    reader = csv.DictReader(f)
    for r in reader:
        id = r['id']
        sheets = id.split('-')
        seen.update(sheets)

with open('data/ia_files.txt', 'r') as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        # Remove the leading 'ia_' and any trailing '.txt'
        line = line.replace('.jpg', '')
        line = line.replace('-', '_')
        ids = []
        if line in special_cases:
            ids = special_cases[line]
        else:
            match = re.match(r'^(\d{4,5}_\d{2,3})', line)
            if match:
                id = match.group(1)
                parts = id.split('_')
                if len(parts[1]) == 3:
                    id = f'{parts[0]}_{parts[1][1:]}'
                ids = [id]
            else:
                print(f"Skipping line: {line} as it does not match the expected format.")
        for id in ids:
            if id not in seen:
                print(f"Missing: {id}")
                seen.add(id)

