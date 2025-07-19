
import re
import csv

def calculate_top_left(latitude, longitude, index):
    """
    Calculate the top-left coordinates based on the index.
    
    The index follows a column-first pattern in a 4x4 grid:
    01 05 09 13
    02 06 10 14
    03 07 11 15
    04 08 12 16
    
    Args:
        latitude (str): The latitude coordinate from the filename
        longitude (str): The longitude coordinate from the filename
        index (str): The index number (01-16)
        
    Returns:
        tuple: (top_latitude, left_longitude) - the coordinates of the top-left corner
    """
    # Convert inputs to integers/floats
    lat = float(latitude)
    lon = float(longitude)
    idx = int(index)
    
    # Calculate row and column based on index (1-based)
    # Columns come first, then rows
    col = (idx - 1) // 4  # 0, 1, 2, 3 for columns
    row = (idx - 1) % 4   # 0, 1, 2, 3 for rows
    
    # Calculate the top-left of this specific cell
    # Latitude INCREASES as row increases
    # Longitude INCREASES going east
    left_lon = lon + col * 0.25  # Add because going east increases longitude
    top_lat = lat + 1.0 - row * 0.25   # Add because latitude increases with row
    
    return (top_lat, left_lon)
    

def extract_coordinates(id):
    ids = id.split('-')
    if len(ids) > 1:
        print(f"Warning: Could not extract coordinates from {id}")
        return None

    id = ids[0]
    parts = id.split('_')
    coordinates = parts[0]
    index = parts[1]
    
    # Parse coordinates into longitude and latitude
    # Last digits are longitude, first digits are latitude
    if len(coordinates) == 4:
        latitude = coordinates[:2]
        longitude = coordinates[2:]
    elif len(coordinates) == 5:
        # Handle 5-digit case (e.g., "21100")
        latitude = coordinates[:2]
        longitude = coordinates[2:]
    else:
        print(f"Warning: Unexpected coordinates format in {id}")
        return None
    
    t, l = calculate_top_left(latitude, longitude, index)
    return [[l,t], [l, t - 0.25], [l + 0.25, t - 0.25], [l + 0.25, t], [l,t]] 

special_cases = {
    "1393_01-1393_05-1493_04-1493_08": [[93.1833333, 14.2166667], [93.1833333, 13.9666667], [93.4333333, 13.9666667], [93.4333333, 14.2166667], [93.1833333, 14.2166667]],
    "1798_02-1798_03": '1798_03',
    "1097_10-1097_11": [[97.50, 10.6666667], [97.50, 10.4166667], [97.75, 10.4166667], [97.75, 10.6666667], [97.50, 10.6666667]],
    "0997_13-0997_14": '0997_14',
    '1399_01-1499_04': '1399_01',

}

done = set()
feats = []
with open('data/zenodo_links_filtered.csv', 'r') as f:
    reader = csv.DictReader(f)
    for r in reader:
        id = r['id']
        if id in done:
            continue
        if id in special_cases:
            if type(special_cases[id]) is str:
               box = extract_coordinates(special_cases[id])
            else:
                box = special_cases[id]
        else:
            box = extract_coordinates(id)
        done.add(id)
        feat = {
            'type': 'Feature',
            'geometry': {'type': 'Polygon', 'coordinates': [box]},
            'properties': r
        }
        feats.append(feat)
 
with open('data/index.geojson', 'w') as f:
    geojson = {
        'type': 'FeatureCollection',
        'features': feats
    }
    import json
    json.dump(geojson, f)
