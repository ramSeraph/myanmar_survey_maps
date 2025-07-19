#!/bin/bash

# 1. get the data list
wget -P data/ https://zenodo.org/records/15028333/files/Map%20Selection%20and%20Download%20Spreadsheet%2020250314.xlsx?download=1
uv run --with openpyxl extract_links_from_zenodo_sheet.py data/Map\ Selection\ and\ Download\ Spreadsheet\ 20250314.xlsx data/zenodo_links.csv "63k Burma"

# 2. filter the links to get only the ones that are of the typpe we are looking for, also parse the names to extract useful information
# creates data/zenodo_links_filtered.csv
uv run filter_links.py

# 3. get available sheet list from internet archive
uvx --from internetarchive ia list myanmar-maps-utm | grep -v thumb | grep -v "myanmar-maps" | grep -v Index | grep -v "\.TAB" > data/ia_files.txt

# 4. see if there are any extra files in ia
uv run print_missing.py

# 5. create an extra ia_files.csv file manually and append
cat ia_files.csv >> data/zenodo_links_filtered.csv

# 6. download the files 
uv run download.py

# 7. create the index file
uv run create_index.py

# 8. parse the sheets
uv run parse.py

# 9. create the tiles
uvx --with numpy --with pillow --with gdal==$GDAL_VERSION --from topo_map_processor tile --tiffs-dir export/gtiffs --tiles-dir export/tiles --max-zoom 14

# 10. partition the tiles
uvx --from topo_map_processor partition --only-disk --from-tiles-dir export/tiles --to-pmtiles-prefix export/pmtiles/Myanmar_50k --attribution-file attribution.txt --name "Myanmar_50k" --description "Myanmar 1:50000 Topo maps from Survey Department"

# 11. push to releases
gh release upload 50k-pmtiles export/pmtiles/Myanmar_50k*
