# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "topo_map_processor[parse]",
# ]
#
# ///


import os
import json
import shutil
from pathlib import Path

from pyproj import Transformer
from pyproj.database import query_utm_crs_info
from pyproj.aoi import AreaOfInterest

from topo_map_processor.processor import TopoMapProcessor, LineRemovalParams

index_map = None
def get_index_map():
    global index_map
    if index_map is not None:
        return index_map

    index_file = Path('data/index.geojson')
    index_map = {}
    data = json.loads(index_file.read_text())
    for feat in data['features']:
        props = feat['properties']
        geom  = feat['geometry']
        id = props['id']
        poly = geom['coordinates'][0]
        index_map[id] = (poly, props)

    return index_map

def get_index_data(name):
    index_map = get_index_map()
    name = name.replace('.jpg', '')
    poly, props = index_map[name]
    return poly, props

class MyanmarProcessor(TopoMapProcessor):

    def __init__(self, filepath, extra, index_box, index_props):
        super().__init__(filepath, extra, index_box, index_props)

        self.resize_factor = extra.get('resize_factor', 0.5)
        self.auto_rotate_thresh = extra.get('auto_rotate_thresh', 0.0)
        self.band_color = extra.get('band_color', None)
        self.band_color_choices = extra.get('band_color_choices', ['black', ['black', 'greyish']])
        self.collar_erode = extra.get('collar_erode', -2)

        self.remove_corner_text = extra.get('remove_corner_text', False)
        self.corner_max_dist_ratio = extra.get('corner_max_dist_ratio', 0.8)
        self.corner_min_dist_ratio = extra.get('corner_min_dist_ratio', 0.3)
        self.min_expected_points = extra.get('min_expected_points', 1)
        self.max_corner_angle_diff = extra.get('max_corner_angle_diff', 5)
        self.max_corner_angle_diff_cutoff = extra.get('max_corner_angle_diff_cutoff', 20)

        self.find_line_iter = extra.get('find_line_iter', 1)
        self.find_line_scale = extra.get('find_line_scale', 8)
        self.line_color = extra.get('line_color', None)
        self.line_color_choices = extra.get('line_color_choices', ['black', ['black', 'greyish'], 'not_white'])

        self.should_remove_grid_lines = False

        # unused: left for reference
        self.grid_bounds_check_buffer_ratio = extra.get('grid_bounds_check_buffer_ratio', 40.0 / 7000.0)
        self.remove_meter_line_buf_ratio = extra.get('remove_meter_line_buf_ratio', 3.0 / 6500.0)
        self.remove_meter_line_blur_buf_ratio = extra.get('remove_meter_line_blur_buf_ratio', 20.0 / 6500.0)
        self.remove_meter_line_blur_kern_ratio = extra.get('remove_meter_line_blur_kern_ratio', 13.0 / 6500.0)
        self.remove_meter_line_blur_repeat = extra.get('remove_meter_line_blur_repeat', 6)

    def get_inter_dir(self):
        return Path('data/inter')

    def get_gtiff_dir(self):
        return Path('export/gtiffs')

    def get_bounds_dir(self):
        return Path('export/bounds')

    def get_crs_proj(self):
        return '+proj=longlat +a=6377276.345 +rf=300.8017 +towgs84=246.632,784.833,276.923,0,0,0,0 +units=m +no_defs'

    def get_utm_zone(self):
        ibox = self.get_sheet_ibox()

        l = ibox[0][0] # noqa
        t = ibox[0][1]
        center = [l + 0.125, t - 0.125]
        utm_crs_list = query_utm_crs_info(
            datum_name="WGS 84",
            area_of_interest=AreaOfInterest(
                west_lon_degree=center[0],
                south_lat_degree=center[1],
                east_lon_degree=center[0],
                north_lat_degree=center[1],
            ),
        )
        utm = utm_crs_list[0].name.split('/')[1].strip()
        return utm


    def get_crs_proj_for_meter_lines(self):
        utm_zone = self.get_utm_zone()
        if utm_zone.endswith('46N'):
            return '+proj=utm +zone=46 +a=6377276.345 +rf=300.8017 +towgs84=246.632,784.833,276.923,0,0,0,0 +units=m +no_defs'
        if utm_zone.endswith('47N'):
            return '+proj=utm +zone=47 +a=6377276.345 +rf=300.8017 +towgs84=246.632,784.833,276.923,0,0,0,0 +units=m +no_defs'

        raise Exception(f'unexpected utm zone {utm_zone}')


    def get_scale(self):
        return 50000

    def prompt1(self):
        pass

    # unused but leaving it in here for reference
    def locate_grid_lines(self):

        full_img = self.get_full_img()
        h,w = full_img.shape[:2]
        bounds_check_buffer = int(self.grid_bounds_check_buffer_ratio * h)

        meter_crs = self.get_crs_proj_for_meter_lines()
        geo_crs = self.get_crs_proj()
        transformer = Transformer.from_crs(geo_crs, meter_crs, always_xy=True)

        gcps = self.get_gcps()

        meter_gcps = []
        for gcp in gcps:
            corner = gcp[0]
            idx = gcp[1]
            meter_idx = transformer.transform(idx[0], idx[1])
            meter_gcp = [corner, meter_idx]
            meter_gcps.append(meter_gcp)

        pixel_transformer = self.get_transformer_from_gcps(meter_gcps)
        lines, lines_xy = self.locate_grid_lines_using_trasformer(pixel_transformer, 1, 1000, bounds_check_buffer)

        meter_params = LineRemovalParams(
            self.remove_meter_line_buf_ratio,
            self.remove_meter_line_blur_buf_ratio,
            self.remove_meter_line_blur_kern_ratio,
            self.remove_meter_line_blur_repeat
        )
        lines = [ (line, meter_params) for line in lines ]

        return lines



    def get_intersection_point(self, img, direction, anchor_angle):
        if self.line_color is not None:
            line_color_choices = [ self.line_color ]
        else:
            line_color_choices = self.line_color_choices

        expect_band_count = 1
        min_expected_points = 1
        ip = None
        for line_color in line_color_choices:
            try:
                ip = self.get_nearest_intersection_point(
                    img, direction, anchor_angle,
                    line_color, self.remove_corner_text,
                    expect_band_count,
                    self.find_line_scale, self.find_line_iter,
                    self.corner_max_dist_ratio, self.corner_min_dist_ratio,
                    min_expected_points,
                    self.max_corner_angle_diff,
                    self.max_corner_angle_diff_cutoff,
                )
                return ip
            except Exception as ex:
                print(f'Failed to find intersection point with color {line_color}: {ex}')
                import traceback
                traceback.print_exc()

        raise Exception(f'Failed to find intersection point with any of the colors: {line_color_choices}')

    def process(self):
        band_color_choices = [self.band_color] if self.band_color is not None else self.band_color_choices
        num_colors = len(band_color_choices)
        for i,band_color in enumerate(band_color_choices):
            try:
                self.band_color = band_color
                super().process()
                return
            except Exception as ex:
                print(f'Failed to process with band color {band_color}: {ex}')
                import traceback
                traceback.print_exc()
                if i == num_colors - 1:
                    continue
                shutil.rmtree(self.get_workdir(), ignore_errors=True)
                self.full_img = None
                self.small_img = None
                self.mapbox_corners = None

        raise Exception(f'Failed to process with all band colors: {band_color_choices}')

def process_files():
    
    data_dir = Path('data/raw')
    
    from_list_file = os.environ.get('FROM_LIST', None)
    if from_list_file is not None:
        fnames = Path(from_list_file).read_text().split('\n')
        image_files = [ Path(f'{data_dir}/{f.strip()}') for f in fnames if f.strip() != '']
    else:
        # Find all jpg files
        print(f"Finding jpg files in {data_dir}")
        image_files = list(data_dir.glob("**/*.jpg"))
    print(f"Found {len(image_files)} jpg files")
    
    
    special_cases_file = Path(__file__).parent / 'special_cases.json'

    special_cases = {}
    if special_cases_file.exists():
        special_cases = json.loads(special_cases_file.read_text())

    total = len(image_files)
    processed_count = 0
    failed_count = 0
    success_count = 0
    # Process each file
    for filepath in image_files:
        print(f'==========  Processed: {processed_count}/{total} Success: {success_count} Failed: {failed_count} processing {filepath.name} ==========')
        extra = special_cases.get(filepath.name, {})
        index_box, index_props = get_index_data(filepath.name)
        processor = MyanmarProcessor(filepath, extra, index_box, index_props)

        try:
            processor.process()
            success_count += 1
        except Exception as ex:
            print(f'parsing {filepath} failed with exception: {ex}')
            failed_count += 1
            import traceback
            traceback.print_exc()
            processor.prompt()
        processed_count += 1

    print(f"Processed {processed_count} images, failed_count {failed_count}, success_count {success_count}")


if __name__ == "__main__":

    process_files()
    
