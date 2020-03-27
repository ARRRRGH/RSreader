#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  3 22:32:26 2019

@author: jim
"""

import os
import numpy as np

import rasterio as rio
from rasterio.mask import mask

import xarray as xr
import uuid
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT

from tqdm import tqdm

from base.bbox import BBox


def rasterio_to_xarray(arr, meta, tmp_dir='.', fil_name=None, chunks=None, out=False, *args, **kwargs):
    if fil_name is None:
        fil_name = str(uuid.uuid4())
    tmp_path = os.path.join(tmp_dir, '%s' % fil_name)

    with rio.open(tmp_path, 'w', **meta) as fil2:
        fil2.write(arr)
    ret = xr.open_rasterio(tmp_path, chunks=chunks)

    if chunks is None and not out:
        os.remove(tmp_path)

    return ret, tmp_path


class _Reader(object):
    def __init__(self, path, bbox=None, time=None, *args, **kwargs):
        self.path = path
        self.bbox = bbox
        self.time = time

    def _which_bbox(self, bbox):
        if bbox is None:
            bbox = self.bbox
        return bbox

    def _which_time(self, time):
        if time is None:
            time = self.time
        return time

    def query(self, time=None, bbox=None, n_jobs=2, crs=None, *args, **kwargs):
        pass


class _RasterReader(_Reader):
    def __init__(self, path, bbox=None, time=None, *args, **kwargs):
        _Reader.__init__(self, path, bbox=bbox, time=time)

    def read(self, paths=None, bbox=None, align=False, crs=None, chunks=None,
             out=False, out_dir='./out', mute=False, *args, **kwargs):

        bbox = self._which_bbox(bbox)

        # single file read
        if paths is None:
            paths = self.path
        if type(paths) is str:
            paths = [paths]

        out_xarrs = []
        out_bboxs = []

        # take bbox of first image if align but no bbox supplied
        if align and bbox is None:
            bbox = BBox.from_tif(paths[0])

        if bbox is not None:
            for i, path in tqdm(enumerate(paths), disable=mute):

                # path arithmetic
                query_dir = os.path.join(out_dir, 'query_out')
                is_query_dir_new = os.path.exists(query_dir)

                os.makedirs(query_dir, exist_ok=True)

                fil_name, ext = os.path.splitext(os.path.basename(path))
                fil_name = os.path.join(query_dir, fil_name + '_cropped' + ext)

                # crop tif and save to tmp file
                ret, tmp_path, fil_crs = _RasterReader._crop_tif(path, bbox=bbox, chunks=chunks, out=True,
                                                                 fil_name=fil_name, *args, **kwargs)

                crs = crs if crs is not None else fil_crs

                # warp image
                if crs != fil_crs:

                    # path arithmetic
                    fil, ext = os.path.splitext(os.path.basename(fil_name))
                    fil_name = os.path.join(os.path.dirname(fil_name), fil + '_warped' + ext)

                    ret, tmp_path2 = _RasterReader._warp_tif(tmp_path, bbox=bbox, crs=crs, chunks=chunks,
                                                             fil_name=fil_name, out=out, *args, **kwargs)

                    if os.path.exists(tmp_path):
                        os.remove(tmp_path)

                    tmp_path = tmp_path2

                if 'crs' in ret.attrs:
                    ret.attrs['crs'] = dict(rio.crs.CRS.from_string(ret.crs))

                ret.attrs['path'] = tmp_path

                if not out:
                    try:
                        os.remove(tmp_path)
                    except (FileNotFoundError, TypeError):
                        pass

                    if is_query_dir_new:
                        os.removedirs(query_dir)

                out_xarrs.append(ret)
                out_bboxs.append(bbox)

        else:
            for path in tqdm(paths, disable=mute):
                with rio.open(path, 'r') as fil:
                    ret = xr.open_rasterio(fil, chunks=chunks)

                # make bbox that is returned
                out_bbox = BBox.from_tif(path)

                ret.attrs['crs'] = dict(rio.crs.CRS.from_string(ret.crs))
                ret.attrs['path'] = path

                out_xarrs.append(ret)
                out_bboxs.append(out_bbox)

        return out_xarrs, out_bboxs

    @staticmethod
    def _crop_tif(path, bbox, tmp_dir='.', fil_name=None, simple=False, *args, **kwargs):

        if not simple:
            with rio.open(path) as fil:
                coords = bbox.get_rasterio_coords(fil.crs.data)
                out_img, out_transform = mask(dataset=fil, shapes=coords, crop=True)
                out_meta = fil.meta.copy()
                crs = fil.crs

                out_meta.update({"driver": "GTiff",
                                 "height": out_img.shape[1],
                                 "width": out_img.shape[2],
                                 "transform": out_transform,
                                 "count": fil.count,
                                 "dtype": out_img.dtype})

            out, tmp_path = rasterio_to_xarray(out_img, out_meta, tmp_dir=tmp_dir,
                                               fil_name=fil_name, *args, **kwargs)

        else:
            with rio.open(path) as fil:
                out = xr.DataArray(fil.read(window=bbox))
                tmp_path = None
                crs = fil.crs

        return out, tmp_path, crs

    @staticmethod
    def _warp_tif(path, bbox, crs, tmp_dir='.', fil_name=None, resampling_method='cubic', *args, **kwargs):
        left, bottom, right, top = bbox.get_bounds(crs=crs)
        res = bbox.get_resolution(crs)

        width = (right - left) // res[0]
        height = (top - bottom) // res[1]
        dst_transform = rio.transform.from_origin(west=left, north=top, xsize=res[0], ysize=res[1])

        vrt_options = {
            'resampling': Resampling[resampling_method],
            'crs': crs,
            'transform': dst_transform,
            'height': height,
            'width': width,
        }

        with rio.open(path) as src:
            with WarpedVRT(src, **vrt_options) as vrt:
                # At this point 'vrt' is a full dataset with dimensions,
                # CRS, and spatial extent matching 'vrt_options'.
                dta = vrt.read()
                # # Read all data into memory.
                # #
                vrt_meta = vrt.meta.copy()

                vrt_meta.update({"driver": "GTiff",
                                 "height": dta.shape[1],
                                 "width": dta.shape[2],
                                 "transform": dst_transform,
                                 "count": vrt.count})

                xarr, tmp_path = rasterio_to_xarray(dta, vrt_meta, tmp_dir=tmp_dir,
                                                    fil_name=fil_name, *args, **kwargs)

        return xarr, tmp_path

    def query(self, time=None, bbox=None, n_jobs=2, crs=None, align=False, *args, **kwargs):
        ret, bbox = self.read(time=time, bbox=bbox, n_jobs=n_jobs, crs=None, align=align, *args, **kwargs)

        # if crs is set, change coordinates
        # fixme: incorrect transformation ?
        # if crs is not None and not align:
        #     ret = [hp.xarray_to_crs(r, crs) for r in ret]

        if len(ret) == 1:
            return ret[0], bbox[0]
        return ret, bbox


class _TimeRasterReader(_RasterReader):
    """
    _TimeRasterReader handles data sets of rasters with a time label. Time information extraction takes place in
    _create_path_dict.
    """

    def __init__(self, dirpath, bbox=None, time=None, *args, **kwargs):
        _RasterReader.__init__(self, path=dirpath, bbox=bbox, time=time, *args, **kwargs)

        self._path_dict = self._create_path_dict()
        self.min_time = min(self._path_dict.values())
        self.max_time = max(self._path_dict.values())

    def query(self, bbox=None, time=None, align=False, crs=None, *args, **kwargs):
        paths, times = self._prepare_query(time=time)
        arrs, bboxs = self.read(paths, bbox=bbox, align=align, crs=crs, *args, **kwargs)

        ret = xr.concat(arrs, 'time')
        ret.coords['time'] = ('time', np.array(times))
        ret = ret.sortby('time')

        str_times = np.array(times, dtype=np.datetime64).astype(str).astype('<U13')
        ret.attrs['path'] = dict([(t, a.attrs['path']) for t, a in zip(str_times, arrs)])

        return ret, bboxs

    def _prepare_query(self, time=None, *args, **kwargs):
        time = self._which_time(time)

        if time is None:
            pathes_times = list(self._path_dict.items())
        else:
            start, end = time
            if start is None:
                start = self.min_time
            if end is None:
                end = self.max_time
            pathes_times = list((path, time) for path, time in self._path_dict.items() if start <= time <= end)

        if len(pathes_times) == 0:
            return None, None

        paths, times = zip(*pathes_times)
        return paths, times

    def _create_path_dict(self):
        pass
