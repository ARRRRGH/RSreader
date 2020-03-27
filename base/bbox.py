from shapely.geometry import Point

import rasterio as rio
import geopandas as gpd
from shapely.geometry import box, Point, mapping, MultiPolygon
import shapely as shp


class BBox(object):
    def __init__(self, bbox, crs=None, res=None):
        if type(bbox) == tuple and len(bbox) == 4:
            shape = [box(*bbox)]
        elif type(bbox) == shp.geometry.polygon.Polygon:
            shape = [bbox]
        elif type(bbox) == shp.geometry.MultiPolygon:
            shape = [p for p in bbox]
        else:
            raise ValueError('bbox must be a quadruple or a shapely.geometry.polygon '
                             'or shapely.geometry.MultiplePolygon')

        self._df = gpd.GeoDataFrame({'geometry': shape}, crs=crs)
        self._df = self._df.reset_index(drop=True)

        if res is None:
            self._res = 1, 1
        else:
            self._res = res

    def set_resolution(self, res, crs=None):
        if crs is None:
            crs = self._df.crs
        self._df.to_crs(crs=crs)
        self._res = res

    def get_resolution(self, crs=None):
        if crs is None:
            return self._res
        else:
            return self._project_resolution(crs)

    def _project_resolution(self, crs):
        left, bottom, right, top = self.get_bounds()

        mid_point_coords = (left + right) // 2, (top + bottom) // 2
        refx_point_coords = mid_point_coords[0] + self._res[0], mid_point_coords[1]
        refy_point_coords = mid_point_coords[0], mid_point_coords[1] + self._res[1]
        pts = [Point(coords) for coords in (mid_point_coords, refx_point_coords, refy_point_coords)]

        df = gpd.GeoDataFrame({'geometry': pts}, crs=crs)
        df = df.to_crs(crs=crs)
        pts = df['geometry']

        return abs(pts[1].coords[0][0] - pts[0].coords[0][0]), abs(pts[2].coords[0][1] - pts[0].coords[0][1])

    def _get(self, crs=None):
        if crs is not None:
            return self._df.to_crs(crs=crs)
        return self._df

    def get_bounds(self, crs=None):
        return MultiPolygon([p for p in self._get(crs=crs)['geometry']]).bounds

    def get_rasterio_coords(self, crs=None):
        # https://automating-gis-processes.github.io/CSC18/lessons/L6/clipping-raster.html
        if crs is not None:
            df = self._df.to_crs(crs=crs)
        else:
            df = self._df
        # return [f['geometry'] for f in json.loads(_df.to_json())['features']]
        return [mapping(geom) for geom in df['geometry']]

    def to_xlim(self, crs=None):
        bounds = self.get_bounds(crs)
        return bounds[0], bounds[2]

    def to_ylim(self, crs=None):
        bounds = self.get_bounds(crs)
        return bounds[1], bounds[3]

    @staticmethod
    def from_rasterio_bbox(bbox, crs):
        return BBox((bbox.left, bbox.bottom, bbox.right, bbox.top), crs)

    @staticmethod
    def from_tif(path):
        with rio.open(path) as fil:
            bbox = fil.bounds
            bbox = BBox.from_rasterio_bbox(bbox, fil.crs)
            bbox.set_resolution(fil.res)
        return bbox