"""
Calculate climatological variance and skewness,
    (used in 2*_.py scripts)

input: ERA daily maximum temperatures on a 1x1 grid (output of 01_process_era.py)
output: a "detrended" version of the ERA 1960-2025 data for calculating climatology

Notable choices:

- To estimate climatological variance and skew, we wanted to use as much data as possible. To this end, the following operations were performed:
    - 1. remove year $y$'s mean from each day in year $y$ for $y$ in 1960:2025
    - 2. estimate the day-of-year climatology by taking the mean across 1960:2025 and smoothing via 5 fourier basis functions
    - 3. remove this day-of-year climatology, and calculate skewness and variance
    - 4. This gets used in 2_era_moments.py
"""

from changing_heat_extremes import flags
from changing_heat_extremes import analysis_helpers as ahelpers
import xarray as xr
import glob

#######################################
# read in ERA data
#######################################


era_filelist = glob.glob("/mnt/media-drive/data/ERA5/t2m_x_1x1/*.nc")
era = (
    xr.open_mfdataset(era_filelist)
    .rename({"__xarray_dataarray_variable__": "t2m_x", "valid_time": "time"})
    .reset_coords(names="number", drop=True)
)


# fixing formatting for the hdp package
era = era.convert_calendar(calendar="noleap", use_cftime=True)
era = era.sel(lat=slice(-60, 80)).chunk(
    {"time": -1, "lat": 10, "lon": 10}
)  # matching karen's doy mask

# convert to (-180, 180) lon. specific to our use case
era = era.assign_coords(lon=(((era.lon + 180) % 360) - 180)).sortby("lon")


# add landmask
era_land = ahelpers.add_landmask(era).compute()


##############################################
# Calculate climatological variance and skewness,
#    (used in 2_era_moments.py)
#############################################

# defining the location-specific heat threshold ---------------
# using time period 1960-2025

# # note! if using jja in nh and djf in southern hemisphere, then we should also include the year before in the ref period
# # bc djf 1950 requires december of 1949.
era_land_climatology_years = era_land.sel(
    time=slice(str(flags.ref_years[0] - 1), str(flags.new_years[1]))
)

# capture "global warming" at each grid cell by getting the mean at each year
era_land_yearly_mean = era_land_climatology_years.groupby("time.year").mean()
era_land_no_yearly_mean = (
    era_land_climatology_years.groupby("time.year") - era_land_yearly_mean
).reset_coords("year", drop=True)

doy_climatology = ahelpers.fourier_climatology_smoother(
    era_land_no_yearly_mean["t2m_x"], n_time=365, n_bases=5
)

# take doy anomalies
era_land_climatology_anom = (
    era_land_no_yearly_mean.groupby("time.dayofyear") - doy_climatology
).drop_vars("dayofyear")
era_land_climatology_anom["t2m_x"].attrs = {"units": "C"}  # hdp package needs units


# ! writing something pretty big (a few gb) to file!
# ahelpers.write_nc(
#     era_land_climatology_anom,
#     f"processed_data/land_anom_for_climatology_{flags.ref_years[0]}_{flags.new_years[1]}.nc",
# )
