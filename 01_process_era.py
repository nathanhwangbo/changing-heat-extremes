#############################
# the ERA data I downloaded on the full .25 degree grid
# Let's put this on a 1 degree grid, to match karen's ERA summer mask
#############################


import glob
import xarray as xr
import xarray_regrid
import re

########################################################
# grabbing an old 1x1 degree file to use as reference grid
#########################################################
era_1x1_filelist = glob.glob("D:data\\ERA5\\t2m_x_1x1\\*.nc")
era_1x1 = xr.open_dataset(era_1x1_filelist[0])
era_1x1 = era_1x1.sel(lat=slice(-60, 80))  # matching karen's doy mask


#####################################
# looping over files to convert each year
####################################

era_raw_filelist = glob.glob("D:data\\ERA5\\t2m_x_daily\\*.nc")


for file in era_raw_filelist:
    year_regex = re.search(r"(\d{4})", file)
    year = year_regex.group(1)

    print(f"working on {year}")

    # match names of the coarse version (time, lon, lat)
    ds_fine = (
        xr.open_dataset(file)
        .rename({"valid_time": "time", "latitude": "lat", "longitude": "lon"})
        .drop_vars("number")
    )

    # using the xarray_regrid utility
    ds_coarse = ds_fine.regrid.linear(era_1x1)
    ds_coarse.to_netcdf(f"D:data\\ERA5\\t2m_x_1x1_2025\\t2m_x_daily_1x1_{year}.nc")
