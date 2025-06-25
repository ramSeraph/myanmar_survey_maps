# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "imgcat",
#     "imutils",
#     "numpy",
#     "opencv-python-headless",
#     "pillow",
#     "pyproj",
#     "rasterio",
#     "shapely",
# ]
# ///


import os
import re
import time
import json
import subprocess
import shutil
from pathlib import Path
from functools import cmp_to_key
from pprint import pprint

import cv2
import imutils
import numpy as np
from PIL import Image
from imgcat import imgcat

from shapely.affinity import translate
from shapely.geometry import LineString, LinearRing, Polygon, CAP_STYLE

from pyproj import CRS
from pyproj.transformer import Transformer
from pyproj.database import query_utm_crs_info
from pyproj.aoi import AreaOfInterest

class GridLinesException(Exception):
    pass

SHOW_IMG = os.environ.get('SHOW_IMG', '0') == '1'
def run_external(cmd):
    print(f'running cmd - {cmd}')
    start = time.time()
    res = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    end = time.time()
    print(f'STDOUT: {res.stdout}')
    print(f'STDERR: {res.stderr}')
    print(f'command took {end - start} secs to run')
    if res.returncode != 0:
        raise Exception(f'command {cmd} failed with exit code: {res.returncode}')

# from camelot.. too slow for a big image
def find_lines(
    threshold, direction="horizontal", line_scale=15, iterations=0
):
    lines = []

    if direction == "vertical":
        size = threshold.shape[0] // line_scale
        el = cv2.getStructuringElement(cv2.MORPH_RECT, (1, size))
    elif direction == "horizontal":
        size = threshold.shape[1] // line_scale
        el = cv2.getStructuringElement(cv2.MORPH_RECT, (size, 1))
    elif direction is None:
        raise ValueError("Specify direction as either 'vertical' or 'horizontal'")

    threshold = cv2.erode(threshold, el)
    #imgcat(Image.fromarray(threshold))
    threshold = cv2.dilate(threshold, el)
    dmask = cv2.dilate(threshold, el, iterations=iterations)
    #imgcat(Image.fromarray(dmask))

    contours, _ = cv2.findContours(
        threshold.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        x1, x2 = x, x + w
        y1, y2 = y, y + h
        if direction == "vertical":
            lines.append(((x1 + x2) // 2, y2, (x1 + x2) // 2, y1))
        elif direction == "horizontal":
            lines.append((x1, (y1 + y2) // 2, x2, (y1 + y2) // 2))

    return dmask, lines


def show_points(points, img_grey, color, radius=5):

    if not SHOW_IMG:
        return
    img_rgb = np.stack([img_grey, img_grey, img_grey], axis=-1)

    h, w = img_rgb.shape[:2]

    for point in points:
        center_x = point[0]
        center_y = point[1]
        y_coords, x_coords = np.ogrid[:h, :w]
        distances = np.sqrt((x_coords - center_x)**2 + (y_coords - center_y)**2)
        circle_mask = distances <= radius
        img_rgb[circle_mask] = color
    imgcat(Image.fromarray(img_rgb))

def show_contours(o_bimg, contours):
    if not SHOW_IMG:
        return
    b = o_bimg.copy()
    rgb = cv2.merge([b*255,b*255,b*255])
    cv2.drawContours(rgb, contours, -1, (0, 255, 0), 2, cv2.LINE_AA)
    if SHOW_IMG:
        imgcat(Image.fromarray(rgb))
    #cv2.imwrite('temp.jpg', rgb)

def reorder_poly_points(poly_points):
    # sort points in anti clockwise order
    num_corners = len(poly_points)
    box = LinearRing(poly_points + [poly_points[0]])
    if not box.is_ccw:
        poly_points = poly_points.copy()
        poly_points.reverse()

    center = box.centroid.coords[0]
    #print(center)
    indices = range(0, num_corners)
    indices = [ i for i in indices if poly_points[i][0] < center[0] and poly_points[i][1] < center[1] ]
    def cmp(ci1, ci2):
        c1 = poly_points[ci1]
        c2 = poly_points[ci2]
        if c1[1] == c2[1]:
            return c2[0] - c1[0]
        else:
            return c2[1] - c1[1]

    s_indices = sorted(indices, key=cmp_to_key(cmp), reverse=True)
    #print(f'{s_indices=}')
    first = s_indices[0]
    poly_reordered = [ poly_points[first] ]
    for i in range(1, num_corners):
        idx = (first - i) % num_corners
        poly_reordered.append(poly_points[idx])
    #print(f'{poly_reordered=}')
    return poly_reordered


def lonlat_to_easting_northing(longitude, latitude, proj_string):
    proj_crs = CRS.from_string(proj_string)

    geo_crs = proj_crs.geodetic_crs
        
    # Create transformer from geographic to projected coordinates
    transformer = Transformer.from_crs(geo_crs, proj_crs, always_xy=True)
        
    # Transform coordinates
    easting, northing = transformer.transform(longitude, latitude)
        
    return easting, northing

def ensure_dir(d):
    d.mkdir(parents=True, exist_ok=True)

def crop_img(img, bbox):
    x, y, w, h = bbox
    return img[y:y+h, x:x+w]

def get_color_mask(img_hsv, color):
    if not isinstance(color, list):
        colors = [color]
    else:
        colors = color

    # https://colorizer.org/ for what the HSV values look like..
    # N.B: the scale there is H:0-359 S:0-99 V:0-99
    #      in opencv it is H:0-179 S:0-255 V:0-255
    img_masks = []
    for color in colors:
        if color == 'black':
            lower = np.array([0, 0, 0])
            #upper = np.array([179, 255, 80])
            upper = np.array([179, 255, 130])
            #23.53, 47.22, 42.35
            #13, 84, 76.23 

            #31.58, 37.5, 59.61
            #15.79, 67.5. 

            #19.46, 30.83, 47.06

            #25.88, 35.17, 

        elif color == 'greyish':
            lower = np.array([0, 0, 50])
            #upper = np.array([179, 130, 145])
            #upper = np.array([179, 150, 155])
            upper = np.array([179, 150, 165])
            # 26.47, 70.34, 56.86
            # 13.23, 126.6, 102.3

            # 48.14, 69.92, 48.24

            # 353.75, 58.18, 64.71
            # 176.8, , 116.
        else:
            raise Exception(f'{color} not handled')
        img_mask = cv2.inRange(img_hsv, lower, upper)
        img_masks.append(img_mask)

    final_mask = img_masks[0]
    for img_mask in img_masks[1:]:
        orred = np.logical_or(final_mask, img_mask)
        final_mask = orred

    return final_mask



def scale_bbox(bbox, rw, rh):
    b = bbox
    return (int(b[0]*rw), int(b[1]*rh), int(b[2]*rw), int(b[3]*rh))

def scale_point(point, rw, rh):
    return [int(point[0]*rw), int(point[1]*rh)]

def translate_bbox(bbox, ox, oy):
    b = bbox
    return (b[0] + ox, b[1] + oy, b[2], b[3])

def extract_coordinates(file_path):
    filename = file_path.name

    if filename == '94 K]02 and 03 Pa-An District Myanmar (2007) 1798_02+03.jpg':
        return {
            'filename': filename,
            'longitude': 98,
            'latitude': 17,
            'index': 3,
            'file_path': file_path
        }
    
    pattern = r'(\d+)_(\d+)(?:\s*\([^)]*\))?.jpg'
    # Extract coordinates and index using regex
    match = re.search(pattern, filename)
    if match:
        coordinates = match.group(1)  # e.g., "2294"
        index = match.group(2)        # e.g., "08"
        
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
            print(f"Warning: Unexpected coordinates format in {filename}")
            return None
        
        # Return data dictionary
        return {
            'filename': filename,
            'longitude': longitude,
            'latitude': latitude,
            'index': index,
            'file_path': file_path
        }
    else:
        print(f"Warning: Could not extract coordinates from {filename}")
        return None

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
    

def get_ext_count(point, img_mask, ext_thresh, delta, factor, cwidth):
    x, y = point
    h, w = img_mask.shape[:2]
    uc = 0
    ext_length = 10*factor
    ye = min(y+ext_length, h - 1)
    print(ye)
    uc += np.count_nonzero(img_mask[y:ye, x])
    for i in range(cwidth):
        uc += np.count_nonzero(img_mask[y:ye, x+i])
        uc += np.count_nonzero(img_mask[y:ye, x-i])

    dc = 0
    ye = max(y-ext_length, 0)
    dc += np.count_nonzero(img_mask[ye:y, x])
    for i in range(cwidth):
        dc += np.count_nonzero(img_mask[ye:y, x+i])
        dc += np.count_nonzero(img_mask[ye:y, x-i])

    lc = 0
    xe = min(x+ext_length, w - 1)
    lc += np.count_nonzero(img_mask[y, x:xe])
    for i in range(cwidth):
        lc += np.count_nonzero(img_mask[y+i, x:xe])
        lc += np.count_nonzero(img_mask[y-i, x:xe])

    rc = 0
    xe = max(x-ext_length, 0)
    rc += np.count_nonzero(img_mask[y, xe:x])
    for i in range(cwidth):
        rc += np.count_nonzero(img_mask[y+i, xe:x])
        rc += np.count_nonzero(img_mask[y-i, xe:x])

    counts = [ uc, dc, rc, lc ]
    print(f'{point=}, {counts=}, {delta=}, {ext_thresh=} {factor=}')
    exts = [ c > (ext_thresh - delta)*factor*(2*cwidth + 1)/3 for c in counts ]
    return exts.count(True)
 
def find_grid_crossings_on_line(x1, y1, x2, y2, grid_spacing):
    """
    Find all points on a line segment where either x or y coordinate 
    is a multiple of grid_spacing.
    """
    grid_points = []
    
    # Parametric line equation: P(t) = P1 + t*(P2-P1), where t ∈ [0,1]
    dx = x2 - x1
    dy = y2 - y1
    
    # Find intersections with vertical grid lines (x = multiple of grid_spacing)
    if abs(dx) > 1e-10:  # Line is not vertical
        # Find range of grid lines that might intersect
        x_min, x_max = min(x1, x2), max(x1, x2)
        x_grid_min = int(np.floor(x_min / grid_spacing)) * grid_spacing
        x_grid_max = int(np.ceil(x_max / grid_spacing)) * grid_spacing
        
        x_grid = x_grid_min
        while x_grid <= x_grid_max:
            if x_grid % grid_spacing == 0:  # Ensure it's exactly a multiple
                t = (x_grid - x1) / dx
                if 0 <= t <= 1:  # Point is on the line segment
                    y_intersect = y1 + t * dy
                    grid_points.append((x_grid, y_intersect))
            x_grid += grid_spacing
    
    # Find intersections with horizontal grid lines (y = multiple of grid_spacing)
    if abs(dy) > 1e-10:  # Line is not horizontal
        # Find range of grid lines that might intersect
        y_min, y_max = min(y1, y2), max(y1, y2)
        y_grid_min = int(np.floor(y_min / grid_spacing)) * grid_spacing
        y_grid_max = int(np.ceil(y_max / grid_spacing)) * grid_spacing
        
        y_grid = y_grid_min
        while y_grid <= y_grid_max:
            if y_grid % grid_spacing == 0:  # Ensure it's exactly a multiple
                t = (y_grid - y1) / dy
                if 0 <= t <= 1:  # Point is on the line segment
                    x_intersect = x1 + t * dx
                    grid_points.append((x_intersect, y_grid))
            y_grid += grid_spacing
    
    ## Remove duplicates (corner points might be found twice)
    #unique_points = []
    #for point in grid_points:
    #    is_duplicate = False
    #    for existing in unique_points:
    #        if (abs(point[0] - existing[0]) < 1e-6 and 
    #            abs(point[1] - existing[1]) < 1e-6):
    #            is_duplicate = True
    #            break
    #    if not is_duplicate:
    #        unique_points.append(point)
    #
    #return unique_points

    return grid_points

def convert_to_row_col(e_pix, e_proj, p_proj):
    # Convert to numpy arrays for easier computation
    e1_proj = np.array(e_proj[0])
    e2_proj = np.array(e_proj[1])
    e1_pix = np.array(e_pix[0])
    e2_pix = np.array(e_pix[1])

    p_proj = np.array(p_proj)
    
    # Calculate edge vector in (x1,y1) space
    edge_proj = e2_proj - e1_proj
    
    # Calculate vector from e1 to point in (x1,y1) space
    p_vec_proj = p_proj - e1_proj
    
    # Calculate parametric value t (0 <= t <= 1 for points on the edge)
    # Using the component with larger magnitude for numerical stability
    if abs(edge_proj[0]) >= abs(edge_proj[1]):
        t = p_vec_proj[0] / edge_proj[0] if edge_proj[0] != 0 else 0
    else:
        t = p_vec_proj[1] / edge_proj[1] if edge_proj[1] != 0 else 0
    
    # Transform to (x,y) space using the same parametric value
    p_xy = e1_pix + t * (e2_pix - e1_pix)
    
    return tuple(p_xy)


def get_distance(p1, p2):
    p1 = np.array(p1)
    p2 = np.array(p2)
    return np.linalg.norm(p2 - p1)

def get_angle(p1, p2):
    p1 = np.array(p1)
    p2 = np.array(p2)
    vector = p2 - p1
    angle_radians = np.arctan2(vector[1], vector[0])
    return np.degrees(angle_radians)


class MyanmarSurveyProcessor:
    def __init__(self, filepath, extra):
        self.filepath = filepath
        self.auto_rotate_thresh = extra.get('auto_rotate_thresh', 0.0)
        self.poly_approx_factor = extra.get('poly_approx_factor', 0.001)
        self.collar_erode = extra.get('collar_erode', -2)
        self.shrunk_map_area_corners = extra.get('shrunk_map_area_corners', None)
        self.band_color = extra.get('band_color', 'black')
        self.use_bbox_area = extra.get('use_bbox_area', True)
        self.corner_overrides = extra.get('corner_overrides', None)
        self.resize_factor = extra.get('resize_factor', 0.5)
        self.corner_max_dist_ratio = extra.get('corner_max_dist_ratio', 0.8)
        self.min_expected_points = extra.get('min_expected_points', 1)
        self.max_corner_angle_diff = extra.get('max_corner_angle_diff', 5)
        self.min_corner_angle_diff = extra.get('min_corner_angle_diff', 30)

        self.find_line_iter = extra.get('find_line_iter', 1)
        self.find_line_scale = extra.get('find_line_scale', 4)
        self.line_color = extra.get('line_color', ['black', 'greyish'])
        #self.line_color = extra.get('line_color', ['greyish'])
        self.grid_lines = extra.get('grid_lines', None)

        self.cwidth = extra.get('cwidth', 1)
        self.mapbox_corners = None
        self.ext_thresh_ratio = extra.get('ext_thresh_ratio', 20.0 / 18000.0)
        #self.ext_thresh_ratio = extra.get('ext_thresh_ratio', 15.0 / 7183.0)
        self.jpeg_export_quality = extra.get('jpeg_export_quality', 75)
        self.warp_jpeg_export_quality = extra.get('warp_jpeg_export_quality', 75)
                    
        self.corner_ratio = extra.get('corner_ratio', 400.0 / 9000.0)
        # Create output directory if it doesn't exist
        item = extract_coordinates(filepath)
        if item is None:
            raise Exception(f'Unable to parse file {filepath}')

        self.base_longitude = item['longitude']
        self.base_latitude  = item['latitude']
        self.index = item['index']
    
        self.top_lat, self.left_lon = calculate_top_left(
            self.base_latitude, self.base_longitude, self.index
        )

        self.workdir = Path('data/inter') / self.get_id()

        self.full_img = None
        self.small_img = None
        self.grid_lines = None
        self.map_img = None

        self.crs_proj = '+proj=longlat +a=6377276.345 +rf=300.8017 +towgs84=246.632,784.833,276.923,0,0,0,0 +units=m +no_defs'
        self.crs_proj_utm = '+proj=utm +zone=47 +south +a=6377276.345 +rf=300.8017 +towgs84=246.632,784.833,276.923,0,0,0,0 +units=m +no_defs'
    
    def get_id(self):
        return f'{self.base_latitude}{self.base_longitude}_{self.index}' 

    def get_map_img(self):
        if self.map_img is not None:
            return self.map_img
        mapbox_file = self.workdir.joinpath('mapbox.jpg')
        self.map_img = cv2.imread(str(mapbox_file))
        return self.map_img
        

    def get_corners(self):
        if self.mapbox_corners is not None:
            return self.mapbox_corners
        corners_file = self.workdir.joinpath('corners.json')
        with open(corners_file, 'r') as f:
            corners = json.load(f)

        self.mapbox_corners = corners
        return corners

    def get_full_img(self):
        if self.full_img is not None:
            return self.full_img
        
        print('loading full image')
        start = time.time()
        rotated_img_file = self.workdir / 'full.rotated.jpg'
        if rotated_img_file.exists():
            file = rotated_img_file
        else:
            file = self.filepath
        self.full_img = cv2.imread(str(file))
        end = time.time()
        print(f'loading image took {end - start} secs')
        return self.full_img

    def get_shrunk_img(self):
        if self.small_img is not None:
            return self.small_img

        small_img_file = self.workdir.joinpath('small.jpg')
        if small_img_file.exists():
            self.small_img = cv2.imread(str(small_img_file))
            return self.small_img

        #sw = 1800

        img = self.get_full_img()
        h, w = img.shape[:2]

        #r = float(sw) / w
        r = self.resize_factor
        #r = 1.00
        #dim = (int(h*r), sw)
        #dim = (int(h*r), int(w*r))
        dim = (int(w*r), int(h*r))
        small_img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(small_img_file), small_img)

        # for some reason this fixes some of the issues
        self.small_img = cv2.imread(str(small_img_file))
        return self.small_img

    
    def get_intersection_point_new(self, img_hsv, direction, anchor_angle):
        img_mask = get_color_mask(img_hsv, self.line_color)
        img_mask_g = img_mask.astype(np.uint8)
        h, w = img_mask.shape[:2]
        corner = [
            0 + w * ( 1 if direction[0] < 0 else 0), 
            0 + h * ( 1 if direction[1] < 0 else 0), 
        ]
        diag_len = get_distance((0,0), (h,w))
        img_mask_g = img_mask_g*255

        v_mask, v_lines = find_lines(img_mask_g, direction='vertical', line_scale=self.find_line_scale, iterations=self.find_line_iter)
        h_mask, h_lines = find_lines(img_mask_g, direction='horizontal', line_scale=self.find_line_scale, iterations=self.find_line_iter)
        print(f'{v_lines=}')
        print(f'{h_lines=}')
        print(f'{direction=}')
      
        ips = []
        only_lines = np.multiply(v_mask, h_mask)
        jcs, _ = cv2.findContours(only_lines, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        for j in jcs:
            #if len(jc) <= 4:  # remove contours with less than 4 joints
            #    continue
            jx, jy, jw, jh = cv2.boundingRect(j)
            c1, c2 = (2 * jx + jw) // 2, (2 * jy + jh) // 2
            if 0 < c1 < w - 1 and 0 < c2 < h - 1:
                ips.append((c1, c2))
        
        print(f'{ips=}')
    
        show_points(ips, img_mask_g, [255,0,0])
    
        #sorted_ips = sorted(ips, key=lambda p: (p[0]*direction[0], p[1]*direction[1]))
        sorted_ips = sorted(ips, key=lambda p: get_distance(corner, p))
        show_points([sorted_ips[0]], img_mask_g, [0,0,255])
    
        anchor_point = sorted_ips[0]
    
        remaining = sorted_ips[1:]

        angles_before = [ abs(get_angle(anchor_point, r) - anchor_angle) for r in remaining ]
        distances_before = [ get_distance(anchor_point, r)/diag_len for r in remaining ]
        print(f'{angles_before=}')
        print(f'{distances_before=}')
        remaining = [ r for r in remaining if self.corner_max_dist_ratio > get_distance(anchor_point, r)/diag_len > 0.3 ]
        remaining = [ r for r in remaining if abs(get_angle(anchor_point, r) - anchor_angle) < self.min_corner_angle_diff ]
        show_points(remaining, img_mask_g, [255,0,0])


        if len(remaining) < self.min_expected_points:
            raise Exception('too few remaining points')
    
        angles = [ abs(get_angle(anchor_point, r) - anchor_angle) for r in remaining ]
        distances = [ get_distance(anchor_point, r)/diag_len for r in remaining ]
        print(f'{angles=}')
        print(f'{distances=}')
    
        dist_max_index = np.argmin(np.array(distances))
    
        ip = remaining[dist_max_index]
        show_points([ip], img_mask_g, [0,255,0])
        angle = angles[dist_max_index]
        if angle > self.max_corner_angle_diff:
            raise Exception(f'angle too high: {angle}')
        print(f'angle: {angle}')

        return ip


    def get_intersection_point(self, img_hsv, ext_thresh):
        img_mask = get_color_mask(img_hsv, self.line_color)
        img_mask_g = img_mask.astype(np.uint8)
        h, w = img_mask.shape[:2]
        img_mask_g = img_mask_g*255
    
        v_mask, v_lines = find_lines(img_mask_g, direction='vertical', line_scale=self.find_line_scale, iterations=self.find_line_iter)
        h_mask, h_lines = find_lines(img_mask_g, direction='horizontal', line_scale=self.find_line_scale, iterations=self.find_line_iter)
        #v_lines = [ l for l in v_lines if 0 < l[0] < w - 1 ]
        #h_lines = [ l for l in h_lines if 0 < l[1] < h - 1 ]
        print(f'{v_lines=}')
        print(f'{h_lines=}')
    
        ips = []
        only_lines = np.multiply(v_mask, h_mask)
        jcs, _ = cv2.findContours(only_lines, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        for j in jcs:
            #if len(jc) <= 4:  # remove contours with less than 4 joints
            #    continue
            jx, jy, jw, jh = cv2.boundingRect(j)
            c1, c2 = (2 * jx + jw) // 2, (2 * jy + jh) // 2
            if 0 < c1 < w - 1 and 0 < c2 < h - 1:
                ips.append((c1, c2))
    
    
        #res = np.where(only_lines == 1)
        #ips = list(zip(res[1], res[0]))
        print(f'{ips=}')
        #imgcat(Image.fromarray(only_lines*255))
    
        pix_factors = [1, 2, 4, 8, 16, 32]
        deltas = [ 0, 1, 2, 3, 4 ]

        four_corner_ips = []
        for delta in deltas:
            for pix_factor in pix_factors:
                four_corner_ips = []
                for ip in ips:
                    ext_count = get_ext_count(ip, img_mask, ext_thresh, delta, pix_factor, self.cwidth)
                    print(f'{ip=}, {ext_count}')
                    if ext_count == 4:
                        four_corner_ips.append(ip)
    
                if len(four_corner_ips) == 0:
                    break

                if len(four_corner_ips) == 1:
                    show_point(four_corner_ips[0], img_mask_g)
                    return four_corner_ips[0]

                if len(four_corner_ips) > 1:
                    continue

        if SHOW_IMG:
            imgcat(Image.fromarray(img_mask_g))
        if len(four_corner_ips) == 0:
            raise Exception(f'no intersection points found - {len(four_corner_ips)}')

        if len(four_corner_ips) > 1:
            raise Exception(f'multiple intersection points found - {len(four_corner_ips)}')

    def locate_corners(self, img_hsv, corner_overrides):
    
        w = img_hsv.shape[1]
        h = img_hsv.shape[0]
        cw = round(self.corner_ratio * w)
        ch = round(self.corner_ratio * h)
        y = h - 1 - ch
        x = w - 1 - cw
    
        print(f'main img dim: {w=}, {h=}')
        # take the four corners
        corner_boxes = []
        corner_boxes.append(((0, 0), (cw, ch)))
        corner_boxes.append(((0, y), (cw, ch)))
        corner_boxes.append(((x, y), (cw, ch)))
        corner_boxes.append(((x, 0), (cw, ch)))
    
        directions = [(+1,+1), (+1,-1), (-1,-1), (-1,+1)]
        anchor_angles = [45, -45, -135, 135]
        # get intersection points
        points = []
        for i, corner_box in enumerate(corner_boxes):
            corner_override = corner_overrides[i]
            if corner_override is not None:
                points.append(corner_override)
                continue
            bx, by = corner_box[0]
            bw, bh = corner_box[1]
            c_img = img_hsv[by:by+bh, bx:bx+bw]
            print(f'{corner_box=}')
            ipoint = self.get_intersection_point_new(c_img, directions[i], anchor_angles[i])
            ipoint = bx + ipoint[0], by + ipoint[1]
            points.append(ipoint)
        return points


    def process_map_area(self, map_bbox, map_poly_points):
        mapbox_file = self.workdir.joinpath('mapbox.jpg')
        corners_file = self.workdir.joinpath('corners.json')
        if corners_file.exists():
            print('corners file exists.. shortcircuiting')
            return

        full_img = self.get_full_img()
        full_img_hsv = cv2.cvtColor(full_img, cv2.COLOR_BGR2HSV)
        map_img_hsv = crop_img(full_img_hsv, map_bbox)
        
        corner_overrides_full = self.corner_overrides
        if corner_overrides_full is None:
            corner_overrides_full = [ None ] * len(map_poly_points)
        corner_overrides = [ (c[0] - map_bbox[0], c[1] - map_bbox[1]) if c is not None else None for c in corner_overrides_full ]
        map_poly_points = [ (p[0] - map_bbox[0], p[1] - map_bbox[1]) for p in map_poly_points ]

        corners = self.locate_corners(map_img_hsv, corner_overrides)

        corners_contour = np.array(corners).reshape((-1,1,2)).astype(np.int32)
        bbox = cv2.boundingRect(corners_contour)
        print(f'{bbox=}')
        print(f'{corners=}')
        map_img_hsv = crop_img(map_img_hsv, bbox)
        print('writing the main mapbox file')
        self.map_img = cv2.cvtColor(map_img_hsv, cv2.COLOR_HSV2BGR)
        cv2.imwrite(str(mapbox_file), self.map_img)
        corners_in_box = [ (c[0] - bbox[0], c[1] - bbox[1]) for c in corners ]
        print(f'{corners_in_box=}')
        self.mapbox_corners = corners_in_box
        with open(corners_file, 'w') as f:
            json.dump(corners_in_box, f, indent = 4)


    def split_image(self):
        print('splitting shrunk image')
        main_splits = self.get_shrunk_splits()
        self.process_map_area(main_splits['map'], main_splits['map_poly_points'])

    def rotate(self):
        rotated_info_file = self.workdir.joinpath('rotated_info.txt')
        if rotated_info_file.exists():
            print('already rotated.. skipping rotation')
            return
        map_bbox, map_min_rect, _ = self.get_maparea()
        print(map_min_rect)
        _, _, angle = map_min_rect
        if angle > 45:
            angle = angle - 90

        if abs(angle) < self.auto_rotate_thresh:
            print(f'not rotated because angle: {angle}')
            rotated_info_file.write_text(f'{angle}, not_rotated')
            return

        img = self.get_full_img()
        print(f'rotating image by {angle}')
        img_rotated = imutils.rotate_bound(img, -angle)
        rotated_file = self.workdir.joinpath('full.rotated.jpg')
        cv2.imwrite(str(rotated_file), img_rotated)
        rotated_info_file.write_text(f'{angle}, rotated')
        self.workdir.joinpath('small.jpg').unlink()
        self.small_img = None
        self.full_img = None


    def get_maparea(self):
        img = self.get_shrunk_img()
        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        img_mask = get_color_mask(img_hsv, self.band_color)
        img_mask_g = img_mask.astype(np.uint8)*255
        if self.shrunk_map_area_corners is not None:
            map_contour = np.array(self.shrunk_map_area_corners).reshape((-1,1,2)).astype(np.int32)
        else:
            start = time.time()
            if self.collar_erode > 0:
                el1 = cv2.getStructuringElement(cv2.MORPH_RECT, (self.collar_erode, self.collar_erode))
                #el2 = cv2.getStructuringElement(cv2.MORPH_RECT, (self.collar_erode, 1))
                img_mask_g = cv2.erode(img_mask_g, el1)
                #img_mask_g = cv2.erode(img_mask_g, el2)
            elif self.collar_erode < 0:
                el1 = cv2.getStructuringElement(cv2.MORPH_RECT, (-self.collar_erode, -self.collar_erode))
                img_mask_g = cv2.dilate(img_mask_g, el1)
            #if SHOW_IMG:
            #    imgcat(Image.fromarray(img_mask_g))
            print(f'getting {self.band_color} contours for whole image')
            contours, hierarchy = cv2.findContours(
                img_mask_g, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE
            )
            ctuples = list(zip(list(contours), list(hierarchy[0])))
            end = time.time()
            print(f'black contours took {end - start} secs')
            show_contours(img_mask_g, [x[0] for x in ctuples])
            if self.use_bbox_area:

                def get_bbox_area(ctuple):
                    bbox = cv2.boundingRect(ctuple[0])
                    return bbox[2] * bbox[3]

                ctuples_s = sorted(ctuples, key=get_bbox_area, reverse=True)
                map_contour = ctuples_s[0][0]
            else:
                ctuples_s = sorted(ctuples, key=lambda x: cv2.contourArea(x[0]), reverse=True)
                #print(ctuples_s[0])
                map_inner_contour_idx = ctuples_s[0][1][2]
                map_contour = ctuples[map_inner_contour_idx][0]
                #map_contour = ctuples_s[0][0]
        map_bbox = cv2.boundingRect(map_contour)
        map_min_rect = cv2.minAreaRect(map_contour)
        print(f'{map_bbox=}')
        print(f'{map_min_rect=}')
        map_area = map_bbox[2] * map_bbox[3]
        h, w = img.shape[:2]
        total_area = w * h
        if total_area / map_area > 2:
            show_contours(img_mask_g, [map_contour])
            raise Exception(f'map area less than expected, {map_area=}, {total_area=}')
    
        show_contours(img_mask_g, [map_contour])
        return map_bbox, map_min_rect, map_contour


    def get_shrunk_splits(self):
        shrunk_splits_file = self.workdir / 'shrunk_splits.json'
        if shrunk_splits_file.exists():
            shrunk_splits = json.loads(shrunk_splits_file.read_text())
            return shrunk_splits

        map_bbox, _, map_contour = self.get_maparea()
        perimeter_len = cv2.arcLength(map_contour, True)
        print(f'{perimeter_len=}')
        epsilon = self.poly_approx_factor * perimeter_len
        map_poly = cv2.approxPolyDP(map_contour, epsilon, True)
        map_poly_points = [ list(p[0]) for p in map_poly ]
        print(f'{map_poly_points=}')
 
        small_img = self.get_shrunk_img()
        h, w = small_img.shape[:2]
        bboxes = [ map_bbox ]

        full_img = self.get_full_img()
        fh, fw = full_img.shape[:2]
        rh, rw = float(fh)/float(h), float(fw)/float(w)

        full_bboxes = [ scale_bbox(bbox, rw, rh) for bbox in bboxes ]
        map_poly_points_scaled = [ scale_point(p, rw, rh) for p in map_poly_points ]

        bbox_dict = {
            'map': full_bboxes[0],
            'map_poly_points': map_poly_points_scaled,
        }

        shrunk_splits_file.write_text(json.dumps(bbox_dict))

        return bbox_dict

    def get_sheet_ibox(self):
        l = self.left_lon #noqa
        t = self.top_lat
        return [[l,t], [l, t - 0.25], [l + 0.25, t - 0.25], [l + 0.25, t], [l,t]] 

    def get_utm_zone(self):
        l = self.left_lon #noqa
        t = self.top_lat
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


    def remove_line(self, line, map_img):
        h, w = map_img.shape[:2]

        #line_buf_ratio = 6.0 / 12980.0
        #blur_buf_ratio = 30.0 / 12980.0
        #blur_kern_ratio = 19.0 / 12980.0
        line_buf_ratio = 6.0 / 6500.0
        blur_buf_ratio = 30.0 / 6500.0
        blur_kern_ratio = 19.0 / 6500.0
        repeat = 1

        line_buf = round(line_buf_ratio * w)
        blur_buf = round(blur_buf_ratio * w)
        blur_kern = round(blur_kern_ratio * w)
        if blur_kern % 2 == 0:
            blur_kern += 1

        limits = Polygon([(w,0), (w,h), (0,h), (0,0), (w,0)])

        ls = LineString(line)
        line_poly = ls.buffer(line_buf, resolution=1, cap_style=CAP_STYLE.flat).intersection(limits)
        blur_poly = ls.buffer(blur_buf, resolution=1, cap_style=CAP_STYLE.flat).intersection(limits)
        bb = blur_poly.bounds
        bb = [ round(x) for x in bb ]
        # restrict to a small img strip to make things less costly
        img_strip = map_img[bb[1]:bb[3], bb[0]:bb[2]]
        sh, sw = img_strip.shape[:2]
        #cv2.imwrite('temp.jpg', img_strip)

        line_poly_t = translate(line_poly, xoff=-bb[0], yoff=-bb[1])
        mask = np.zeros(img_strip.shape[:2], dtype=np.uint8)
        poly_coords = np.array([ [int(x[0]), int(x[1])] for x in line_poly_t.exterior.coords ])
        cv2.fillPoly(mask, pts=[poly_coords], color=1)

        #img_blurred = cv2.medianBlur(img_strip, blur_kern)
        pad = int(blur_kern/2)
        img_strip_padded = cv2.copyMakeBorder(img_strip, pad, pad, pad, pad, cv2.BORDER_REFLECT_101)

        img_blurred_padded = cv2.medianBlur(img_strip_padded, blur_kern)
        for i in range(repeat):
            img_blurred_padded = cv2.medianBlur(img_blurred_padded, blur_kern)
        img_blurred = img_blurred_padded[pad:pad+sh, pad:pad+sw]
        #cv2.imwrite('temp.jpg', img_blurred)

        img_strip[mask == 1] = img_blurred[mask == 1]

    def get_utm_proj(self):
        utm_zone = self.get_utm_zone()
        if utm_zone.endswith('46N'):
            return '+proj=utm +zone=46 +a=6377276.345 +rf=300.8017 +towgs84=246.632,784.833,276.923,0,0,0,0 +units=m +no_defs'
        if utm_zone.endswith('47N'):
            return '+proj=utm +zone=47 +a=6377276.345 +rf=300.8017 +towgs84=246.632,784.833,276.923,0,0,0,0 +units=m +no_defs'

        raise Exception(f'unexpected utm zone {utm_zone}')

    def locate_grid_lines_using_coords(self):
        GRID_SPACING = 1000
        corners = self.get_corners()
        ibox = self.get_sheet_ibox()

        if len(ibox) - 1 != len(corners):
            raise Exception(f'{len(ibox) - 1=} != {len(corners)=}')

        proj_utm = self.get_utm_proj()
        corners_proj = [ lonlat_to_easting_northing(p[0], p[1], proj_utm) for p in ibox ] 

        # Define the four edges of the rectangle
        edges = [
            (corners_proj[0], corners_proj[1]),  # Edge 1-2
            (corners_proj[1], corners_proj[2]),  # Edge 2-3
            (corners_proj[2], corners_proj[3]),  # Edge 3-4
            (corners_proj[3], corners_proj[0])   # Edge 4-1
        ]

        edges_pix = [
            (corners[0], corners[1]),  # Edge 1-2
            (corners[1], corners[2]),  # Edge 2-3
            (corners[2], corners[3]),  # Edge 3-4
            (corners[3], corners[0])   # Edge 4-1
        ]
        
        grid_points = {}
        all_grid_points = []
        
        for edge_idx, (start, end) in enumerate(edges):
            edge_name = f"Edge_{edge_idx+1}"
            edge_pix = edges_pix[edge_idx]
            grid_points[edge_name] = []
            
            x1, y1 = start
            x2, y2 = end
            
            # Find grid crossings along this edge
            edge_grid_points = find_grid_crossings_on_line(
                x1, y1, x2, y2, GRID_SPACING
            )
            
            # Convert back to lat/lon for each grid point
            for x_grid, y_grid in edge_grid_points:
                
                row_col = convert_to_row_col(edge_pix, (start, end), (x_grid, y_grid))
                p_info = {
                    'easting_northing': (x_grid, y_grid),
                    'row_col': row_col
                }
                grid_points[edge_name].append(p_info)
                all_grid_points.append(p_info) 

        grid_lines = []
        done = set()
        for idx, p_info in enumerate(all_grid_points):
            if idx in done:
                continue

            x, y = p_info['easting_northing']
            for o_idx, o_p_info in enumerate(all_grid_points):
                if o_idx <= idx:
                    continue

                o_x, o_y = o_p_info['easting_northing']

                x_matched = False
                if abs(o_x - x) < 1e-6:
                    x_matched = True

                y_matched = False
                if abs(o_y - y) < 1e-6:
                    y_matched = True

                if (x_matched and not y_matched) or (y_matched and not x_matched):
                    done.add(idx)
                    done.add(o_idx)
                    grid_lines.append([p_info['row_col'], o_p_info['row_col']])
                    break

        #if len(done) != len(all_grid_points):
        #    raise GridLinesException(f'{len(done)=} {len(all_grid_points)=}')

        return grid_lines

    def remove_lines(self):
        nogrid_file = self.workdir.joinpath('nogrid.jpg')
        if nogrid_file.exists():
            print(f'{nogrid_file} file exists.. skipping')
            return

        if self.grid_lines is not None:
            grid_lines = self.grid_lines
        else:
            grid_lines = self.locate_grid_lines_using_coords()
        #print(f'{grid_lines=}')
        #if len(grid_lines) == 0:
        #    return

        map_img = self.get_map_img()
        print('dropping grid lines')
        for line in grid_lines:
            self.remove_line(line, map_img)

        cv2.imwrite(str(nogrid_file), map_img)

    def georeference_mapbox(self):
        #mapbox_file = self.workdir.joinpath('mapbox.jpg')
        nogrid_file = self.workdir.joinpath('nogrid.jpg')
        georef_file = self.workdir.joinpath('georef.tif')
        final_file = self.workdir.joinpath('final.tif')
        if georef_file.exists() or final_file.exists():
            print(f'{georef_file} or {final_file} exists.. skipping')
            return

        ibox = self.get_sheet_ibox()

        corners = self.get_corners()

        #src_crs = CRS.from_proj4(crs_proj)
        #epsg = '7015'

        if len(ibox) - 1 != len(corners):
            raise Exception(f'{len(ibox) - 1=} != {len(corners)=}')

        gcp_str = ''
        for idx, c in enumerate(corners):
            i = ibox[idx]
            gcp_str += f' -gcp {c[0]} {c[1]} {i[0]} {i[1]}'
        perf_options = '--config GDAL_CACHEMAX 128 --config GDAL_NUM_THREADS ALL_CPUS'
        translate_cmd = f'gdal_translate {perf_options} {gcp_str} -a_srs "{self.crs_proj}" -of GTiff {str(nogrid_file)} {str(georef_file)}' 
        run_external(translate_cmd)

    def create_cutline(self, ibox, file):
        with open(file, 'w') as f:
            cutline_data = {
                "type": "FeatureCollection",
                "name": "CUTLINE",
                "features": [{
                    "type": "Feature",
                    "properties": {},
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [ibox]
                    }
                }]
            }
            json.dump(cutline_data, f, indent=4)


    def warp_mapbox(self):
        cutline_file = self.workdir.joinpath('cutline.geojson')
        georef_file = self.workdir.joinpath('georef.tif')
        final_file = self.workdir.joinpath('final.tif')
        if final_file.exists():
            print(f'{final_file} exists.. skipping')
            return

        sheet_ibox = self.get_sheet_ibox()

        def warp_file(box, cline_file, f_file, jpeg_quality):
            img_quality_config = {
                'COMPRESS': 'JPEG',
                #'PHOTOMETRIC': 'YCBCR',
                'JPEG_QUALITY': f'{jpeg_quality}'
            }

            #if not USE_4326:
            #    box = [ transformer.transform(*c) for c in box ]

            self.create_cutline(box, cline_file)
            cutline_options = f'-cutline {str(cline_file)} -cutline_srs "{self.crs_proj}" -crop_to_cutline --config GDALWARP_IGNORE_BAD_CUTLINE YES -wo CUTLINE_ALL_TOUCHED=TRUE'

            warp_quality_config = img_quality_config.copy()
            warp_quality_config.update({'TILED': 'YES'})
            warp_quality_options = ' '.join([ f'-co {k}={v}' for k,v in warp_quality_config.items() ])
            reproj_options = '-tps -ts 0 6500 -r bilinear -t_srs "EPSG:3857"' 
            #nodata_options = '-dstnodata 0'
            nodata_options = '-dstalpha'
            perf_options = '-multi -wo NUM_THREADS=ALL_CPUS --config GDAL_CACHEMAX 1024 -wm 1024' 
            warp_cmd = f'gdalwarp -overwrite {perf_options} {nodata_options} {reproj_options} {warp_quality_options} {cutline_options} {str(georef_file)} {str(f_file)}'
            run_external(warp_cmd)

            
            #addo_quality_options = ' '.join([ f'--config {k}_OVERVIEW {v}' for k,v in img_quality_config.items() ])
            #addo_cmd = f'export GDAL_NUM_THREADS=ALL_CPUS; gdaladdo {addo_quality_options} -r average {str(final_file)} 2 4 8 16 32'
            #run_external(addo_cmd)

        warp_file(sheet_ibox, cutline_file, final_file, self.warp_jpeg_export_quality)
        # delete the georef file as it can get too big
        #georef_file.unlink()

    def export_internal(self, filename, out_filename, jpeg_export_quality):
        if Path(out_filename).exists():
            print(f'{out_filename} exists.. skipping export')
            return
        creation_opts = f'-co TILED=YES -co COMPRESS=JPEG -co JPEG_QUALITY={jpeg_export_quality} -co PHOTOMETRIC=YCBCR' 
        mask_options = '--config GDAL_TIFF_INTERNAL_MASK YES  -b 1 -b 2 -b 3 -mask 4'
        perf_options = '--config GDAL_CACHEMAX 512'
        cmd = f'gdal_translate {perf_options} {mask_options} {creation_opts} {filename} {out_filename}'
        run_external(cmd)

    def get_export_file(self):
        export_dir = Path('export/gtiffs')
        sheet_no = self.get_id()
        return export_dir / f'{sheet_no}.tif'

    def export(self):
        filename = str(self.workdir.joinpath('final.tif'))
        export_file = self.get_export_file()
        ensure_dir(export_file.parent)
        self.export_internal(filename, str(export_file), self.jpeg_export_quality)
        shutil.rmtree(self.workdir)


    def process(self):
        export_file = self.get_export_file()
        if export_file.exists():
            return True

        ensure_dir(self.workdir)
        self.rotate()
        self.split_image()
        self.remove_lines()
        #raise Exception('temp')
        self.georeference_mapbox()
        self.warp_mapbox()
        self.export()

        return True


def process_files(data_dir="data/raw", output_dir="data/inter"):
    
    failed_file = Path('failed.txt')
    failed_files = set()
    if failed_file.exists():
        txt = failed_file.read_text()
        failed = txt.split('\n')
        failed = [ f.strip() for f in failed if f.strip() != '' ]
        failed_files.update(failed)

    # Convert to Path object
    data_dir = Path(data_dir)
    
    from_list_file = os.environ.get('FROM_LIST', None)
    if from_list_file is not None:
        fnames = Path(from_list_file).read_text().split('\n')
        image_files = [ Path(f'data/raw/{f.strip()}') for f in fnames if f.strip() != '']
    else:
        # Find all jpg files
        print(f"Finding jpg files in {data_dir}")
        image_files = list(data_dir.glob("**/*.jpg"))
    print(f"Found {len(image_files)} jpg files")
    
    processed_count = 0
    
    special_cases_file = Path(__file__).parent / 'special_cases.json'
    special_cases = json.loads(special_cases_file.read_text())
    # Process each file
    total = len(image_files)
    done = 0
    failed = 0
    for file_path in image_files:
        #if file_path.name in failed_files:
        #    continue
        print(f'=========== {done=}/{failed=}/{total=}: processing {file_path.name} ===============')
        if file_path.name in [
            "85 J]09 Thandwe District Myanmar (2004) 1894_09.jpg",
            "96 F]10 and F]11 Kawthoung District Myanmar (2008)1097_10 and 11.jpg",
            "86 F]04 and 08 Yangon Southern District Myanmar (2007)1393 _ 01&05 & 1493_04&08.jpg",
            #"94 K]02 and 03 Pa-An District Myanmar (2007) 1798_02+03.jpg",
            #"96 G]14 Kawthoung District Myanmar (2007) 0997_14.jpg",
            #"84 J]08 Gangaw District Myanmar (2004) 2294_08 (84_J_08).jpg",
            #"94 D]03 Pyapon District Myanmar (2004) 1696_03.jpg",
            #"94 I]13 Mongsat District Myanmar (2007) 1998_13.jpg",
            #"84 J]03 Falam District Myanmar (2004) 2294_03.jpg",
            #"95 O]01 Dawei District (2008) 1399_01.jpg",
        ]:
            continue
        extra = special_cases.get(file_path.name, {})
        processor = MyanmarSurveyProcessor(file_path, extra)
        try:
            processor.process()
            processed_count += 1
        except GridLinesException as ex:
            print(ex)
            with open(failed_file, 'a') as f:
                f.write(file_path.name)
                f.write('\n')
        except Exception as ex:
            print(ex)
            failed += 1
            raise
        done += 1  
    
    return processed_count


if __name__ == "__main__":
    #proj_str = '+proj=longlat +a=6377276.345 +rf=300.8017 +towgs84=246.632,784.833,276.923,0,0,0,0 +units=m +no_defs'
    #proj_str = '+proj=utm +zone=47 +south +a=6377276.345 +rf=300.8017 +towgs84=246.632,784.833,276.923,0,0,0,0 +units=m +no_defs'
    #out = lonlat_to_easting_northing(96.5, 17.5, proj_str)
    #print(out)
    #exit(0)
    # Process images
    processed_count = process_files()
    
    print(f"Successfully processed {processed_count} images")
