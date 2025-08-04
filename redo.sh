#!/bin/bash

# this assumes that the raw files and files in the export directory have been cleared
# rm -rf data/raw export/gtiffs export/bounds.geojson export/bounds export/pmtiles

# 1) create a retile file list at redo_list.txt without the 'jpg' extensions 


# 2) download the relevant raw files
mkdir -p data/raw

gh release download 50k-orig listing_files.csv

cat redo_list.txt | xargs -I {} sh -c "cat listing_files.csv|grep '^{}.jpg,'" | cut -d"," -f3 | xargs -I {} wget -P data/raw/ {}

rm redo_list.txt listing_files.csv

# 3) fix files and run the parse command
uv run parse.py

# 4) recreate the bounds file and push it
gh release download 50k-georef -p bounds.geojson 
uvx --from topo_map_processor collect-bounds --preexisting-file bounds.geojson --bounds-dir export/bounds --output-file export/bounds.geojson
gh release upload 50k-georef export/bounds.geojson --clobber
rm export/bounds.geojson bounds.geojson

# 5) push the new geotiffs and update
uvx --from topo_map_processor upload-to-release 50k-georef export/gtiffs/ tif yes
uvx --from topo_map_processor generate-lists.sh 50k-georef .tif

# 6) recreate the pmtiles and reupload
GDAL_VERSION=$(gdalinfo --version | cut -d"," -f1 | cut -d" " -f2)

uvx --with numpy \
    --with pillow \
    --with gdal==$GDAL_VERSION \
    --from topo_map_processor \
    retile-e2e -p 50k-pmtiles -g 50k-georef -x Myanmar_50k -l listing_files.csv



