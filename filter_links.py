import csv
import re

special_cases = {
    "86 F]04 and 08 Yangon Southern District Myanmar (2007)1393 _ 01&05 & 1493_04&08.jpg": ('1393_01-1393_05-1493_04-1493_08','86F_04-86F_08'),
    "94 K]02 and 03 Pa-An District Myanmar (2007) 1798_02+03.jpg": ('1798_02-1798_03', '94K_02-94K_03'),
    "96 F]10 and F]11 Kawthoung District Myanmar (2008)1097_10 and 11.jpg": ('1097_10-1097_11','96F_10-96F_11'),
    "96 G]14 Kawthoung District Myanmar (2007) 0997_14.jpg": ('0997_13-0997_14', '96G_13-96G_14'),
    "95 O]01 Dawei District (2008) 1399_01.jpg": ('1399_01-1499_04', '95O_01-95N_04'),
}

def parse_name(name):
    if name in special_cases:
        return special_cases[name]

    match = re.search(r'^(\d+\s[A-Z])\](\d{2}).*?\s*(\d{4,5}[_-]\d{1,2})\.jpg', name)
    if not match:
        raise Exception(f"Name format not recognized: {name}")

    old_id = match.group(1).replace(' ', '') + '_' + match.group(2)
    id = match.group(3).replace('-', '_')
    return id, old_id


with open('data/zenodo_links_filtered.csv', 'w') as of:
    wr = csv.writer(of)
    wr.writerow(['id', 'old_id', 'year', 'full_name', 'link'])
    with open('data/zenodo_links.csv', 'r') as f:
        reader = csv.reader(f)
        for r in reader:
            name = r[0]
            link = r[1]
            match = re.search(r'\((\d+)\)', name)
            if not match:
                #print(f"Skipping {name} as it does not match the expected format.")
                continue
            year = match.group(1)
            year = int(year)
            #print(year)
            if year > 2000:
                if name in  ['83 O]15 Hkamti Township (2005).jpg', '83 O]11 Hkamti Township (2005).jpg']:
                    continue
                link=link.split('?')[0]
                id, old_id = parse_name(name)
                wr.writerow([id, old_id, year, name, link])
