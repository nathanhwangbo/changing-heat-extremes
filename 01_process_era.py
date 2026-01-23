#############################
# the ERA data I downloaded on the full .25 degree grid
# Let's put this on a 1 degree grid, to match karen's ERA summer mask
#############################


# import glob
# import xarray as xr
# import xarray_regrid
# import re

# ########################################################
# # grabbing an old 1x1 degree file to use as reference grid
# #########################################################
# era_1x1_filelist = glob.glob("D:data\\ERA5\\t2m_x_1x1\\*.nc")
# era_1x1 = xr.open_dataset(era_1x1_filelist[0])
# era_1x1 = era_1x1.sel(lat=slice(-60, 80))  # matching karen's doy mask


# #####################################
# # looping over files to convert each year
# ####################################

# era_raw_filelist = glob.glob("D:data\\ERA5\\t2m_x_daily\\*.nc")


# for file in era_raw_filelist:
#     year_regex = re.search(r"(\d{4})", file)
#     year = year_regex.group(1)

#     print(f"working on {year}")

#     # match names of the coarse version (time, lon, lat)
#     ds_fine = (
#         xr.open_dataset(file)
#         .rename({"valid_time": "time", "latitude": "lat", "longitude": "lon"})
#         .drop_vars("number")
#     )

#     # using the xarray_regrid utility
#     ds_coarse = ds_fine.regrid.linear(era_1x1)
#     ds_coarse.to_netcdf(f"D:data\\ERA5\\t2m_x_1x1_2025\\t2m_x_daily_1x1_{year}.nc")


##############################################################3
# here's karen's version, if we wanna switch.
# from vortex:  modified from /kmckinnon/summer_extremes/summer_extremes/scripts/regrid_era5_1x1.py
################################################################3


import xesmf as xe
import xarray as xr
from glob import glob
import os
import numpy as np
# from subprocess import check_call

# era5_dir = '/home/data/ERA5/day'
# varnames = ['t2m', 't2m_n', 't2m_x']

era5_dir = "D:data\\ERA5\\t2m_x_daily\\"
varnames = "t2m_x"

lat1x1 = np.arange(-89.5, 90)
lon1x1 = np.arange(0.5, 360)

for varname in varnames:
    files = glob("D:data\\ERA5\\t2m_x_daily\\*.nc")

    era5_dir_1x1 = "D:data\\ERA5\\t2m_x_1x1\\"
    # # make 1x1 dir if not already present
    # era5_dir_1x1 = "%s/%s/1x1" % (era5_dir, varname)
    # cmd = 'mkdir -p %s' % era5_dir_1x1
    # check_call(cmd.split())

    for f in files:
        f_new = f.replace(".nc", "_1x1.nc").split("\\")[-1]
        f_new = "%s%s" % (era5_dir_1x1, f_new)

        if os.path.isfile(f_new):
            continue
        else:
            print(f)
            wgt_file = "%s/xe_weights_1x1.nc" % (era5_dir)
            if os.path.isfile(wgt_file):
                reuse_weights = True
            else:
                reuse_weights = False

            da = xr.open_dataarray(f)

            da = da.rename({"latitude": "lat", "longitude": "lon"})
            da = da.sortby("lat")

            regridder = xe.Regridder(
                {"lat": da.lat, "lon": da.lon},
                {"lat": lat1x1, "lon": lon1x1},
                "bilinear",
                periodic=True,
                reuse_weights=reuse_weights,
                filename=wgt_file,
            )

            da = regridder(da)
            da.to_netcdf(f_new)

## regrid landmask if needed
# land_file = "/home/data/ERA5/fx/era5_lsmask.nc"
# f_new = land_file.replace(".nc", "_1x1.nc")
# if not os.path.isfile(f_new):
#     da = xr.open_dataarray(land_file)
#     da = da.rename({"latitude": "lat", "longitude": "lon"})
#     da = da.sortby("lat")

#     regridder = xe.Regridder(
#         {"lat": da.lat, "lon": da.lon},
#         {"lat": lat1x1, "lon": lon1x1},
#         "bilinear",
#         periodic=True,
#         reuse_weights=reuse_weights,
#         filename=wgt_file,
#     )

#     da = regridder(da)
#     da.to_netcdf(f_new)
