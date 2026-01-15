##########################
# DEPRECATED!
# use 0_era_medianshift.py with use_calendar_summer = False
#########################

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import glob
import regionmask
import hdp  # pip install -e "E:\\Projects\\HDP\\"

xr.set_options(use_new_combine_kwarg_defaults=True)


######################################################
## Process ERA
######################################################

# day of year summer mask
era_summer_path = "D:data\\ERA5\\ERA5_hottest_doys_t2m_x.nc"
era_summer_mask = xr.open_dataarray(era_summer_path)
# era_land.groupby("time.dayofyear").where(era_summer_mask == 1)


# ERA daily tmax
era_filelist = glob.glob("D:data\\ERA5\\t2m_x_1x1\\*.nc")
era = xr.open_mfdataset(era_filelist).drop_vars("expver")

# fixing formatting for the hdp package
era["t2m_x"].attrs = {"units": "K"}  # hdp package needs units
# era = era.convert_calendar(calendar="standard", use_cftime=True)  # .compute()
era = era.convert_calendar(calendar="noleap", use_cftime=True)
era = era.sel(lat=slice(-60, 80)).chunk(
    {"time": -1, "lat": 10, "lon": 10}
)  # matching karen's doy mask


### landmask, and subset to summer days- -------------------
def add_landmask(ds):
    # create a landmask
    land = regionmask.defined_regions.natural_earth_v5_0_0.land_110
    landmask = land.mask(ds)  # ocean is nan, land is 0
    is_land = landmask == 0

    # also get rid of greenland
    greenland = regionmask.defined_regions.natural_earth_v5_0_0.countries_110[
        ["Greenland"]
    ]
    gl_mask = greenland.mask(ds)
    is_not_greenland = gl_mask.isnull()

    # also get rid of antarctic
    is_not_antarctic = ds["lat"] > -60
    # is_not_arctic = ds["lat"] < 60

    # apply landmask
    ds = ds.where(is_land & is_not_greenland & is_not_antarctic)

    return ds


era_land = add_landmask(era).compute()

# Note! we're taking anomalies with respect to the ENTIRE time period
# not just the reference period below!
# ref_doy_climatology = era_land.groupby("time.dayofyear").mean()


#################################################################
# following russo et al,
# include 2 months before and after JJA,
# # then identify heatwaves,
## for each heatwave, count the proportion of days the heatwave has in JJA.
## if this proportion is more than 50%, then keep the heatwave as a "JJA" heatwave
#################################################################################


## identify heatwaves: ------------------------------------------

# 3 consecutive days of temperatures above T90_d, calculated as follows:
# for each day d in MJJAS
# pull out a 30 day window centered at d, using years 1961-1990 (i.e. shoould have 30*30 days included)
# calcuate q90 of daily max temperature -> T90_d
# check if days (d, d+1, d+2) all have higher temperature than T90_d
# if so, then keep counting consecutive days that have higher temperature. This is a heatwave event.


# new approach ----
# 3 consecutive days of temperatures above T90_d, calculated as follows:
# pick out a summer season for each gridcell (either JJA or a 90 day window)
# for each day d in summer
# pull out a 30 day window centered at d, using years 1961-1990 (i.e. shoould have 30*30 days included)
# calcuate q90 of daily max temperature -> T90_d
# check if days (d, d+1, d+2) all have higher temperature than T90_d
# if so, then keep counting consecutive days that have higher temperature. This is a heatwave event.


# new new approach ----
# 3 consecutive days of temperatures above T90, calculated as follows:
# pick out a summer season for each gridcell (either JJA or a 90 day window)
# for each day d in summer
# using years 1961-1990 (i.e. shoould have 30 yrs*90 days included) calcuate q90 of daily max temperature -> T90
# check if days (d, d+1, d+2) all have higher temperature than T90
# if so, then keep counting consecutive days that have higher temperature. This is a heatwave event.
# q: are we worried about heatwaves favoring days in the middle of the 90 day period?


################################################
# calculate extremal metrics at each gridcell
###############################################

# defining the location-specific heat threshold ---------------

# using reference period 1960-1985 --------------------
# note that this reference period is fully in the "old" time period (see below)
era_land_ref = era_land.sel(time=slice("1960", "1985"))


# Note! we're taking anomalies with respect to the REFERENCE time period
ref_doy_climatology = era_land_ref.groupby("time.dayofyear").mean()
era_land_ref_anom = (
    era_land_ref.groupby("time.dayofyear") - ref_doy_climatology
).drop_vars("dayofyear")
era_land_ref_anom["t2m_x"].attrs = {"units": "C"}  # hdp package needs units
# conversion to celcius
measures_ref = hdp.measure.format_standard_measures(
    temp_datasets=[era_land_ref_anom["t2m_x"]]
)
percentiles = [0.9]

thresholds_ref = hdp.threshold.compute_thresholds(
    measures_ref, percentiles, rolling_window_size=7
)  # .compute()

# #### match the output of hdp.threshold.compute_threshold
# era_tmax_ref_summer_masked = measures_ref.groupby("time.dayofyear").where(
#     era_summer_mask == 1
# )
# threshold_ref_summer_mask = era_tmax_ref_summer_masked.quantile(0.9, dim="time")

# thresholds_ref = (
#     threshold_ref_summer_mask.expand_dims(doy=era_summer_mask.dayofyear.values)
#     .expand_dims(percentile=percentiles)
#     .rename({"t2m_x": "t2m_x_threshold"})
#     .transpose("lat", "lon", "doy", "percentile")
# ).compute()
# thresholds_ref["t2m_x_threshold"].attrs["baseline_variable"] = "t2m_x"
# thresholds_ref["t2m_x_threshold"].attrs["hdp_type"] = "threshold"
# thresholds_ref["t2m_x_threshold"].attrs["baseline_calendar"] = "noleap"
# thresholds_ref = thresholds_ref.drop_vars("quantile")


## heatwave is defined as 3 consec days
definitions = [[3, 0, 0]]


# --------------------------------------------------

# time period 1, 1950-1985, following russo and domeisen 2023
era_land_old = era_land.sel(time=slice("1950", "1985"))
era_land_old_anom = era_land_old.groupby("time.dayofyear") - ref_doy_climatology

# time period 2, 1986-2021, following russo and domeisen 2023
era_land_new = era_land.sel(time=slice("1986", "2021"))
era_land_new_anom = era_land_new.groupby("time.dayofyear") - ref_doy_climatology

###########
# synthetic, medianshift
#############

use_median_map = True

if use_median_map:
    # shifting the median for each grid cell
    old_medians = era_land_old_anom["t2m_x"].median(dim=["time"])
    new_medians = era_land_new_anom["t2m_x"].median(dim=["time"])
else:
    # shifting the median, single median across time and space
    # .median doesn't work withoutt specifying a subset of dims
    old_medians = era_land_old_anom["t2m_x"].quantile(q=0.5)
    new_medians = era_land_new_anom["t2m_x"].quantile(q=0.5)


era_land_synth_new = era_land_old_anom - old_medians + new_medians
era_land_synth_new = era_land_synth_new.assign_coords(
    time=era_land_new_anom.time
)  # pretend its the new time period

# combine back. this is comparable to era_land_all below
era_land_synth = xr.concat(
    [era_land_old_anom, era_land_synth_new], dim="time"
).drop_vars("dayofyear")
era_land_synth["t2m_x"].attrs = {"units": "C"}  # hdp package needs units
# use c when dealing with anomalies (bc anomalies are the same in either units)


### quick plot of the median-shift -----------------------------
# import cartopy.crs as ccrs
# import hvplot.xarray
# median_shift = (new_medians - old_medians).compute()
# median_shift = median_shift.assign_coords(
#     lon=(((median_shift.lon + 180) % 360) - 180)
# ).sortby("lon")

# fig_median_shift = median_shift.hvplot(
#     projection=ccrs.PlateCarree(),
#     coastline=True,
#     cmap=rdbu_discrete,
#     clim=(-2, 2),
#     title="summer tmax median shift\n(1986-2021) - (1950-1985)",
#     clabel="deg C",
# ).opts(fontscale=1.5)


# calculate heatwave metrics
measures_synth = hdp.measure.format_standard_measures(
    temp_datasets=[era_land_synth["t2m_x"]]
).chunk({"time": -1, "lat": 10, "lon": 10})

thresholds_ref = thresholds_ref.chunk({"lat": 10, "lon": 10})
metrics_dataset_synth = hdp.metric.compute_group_metrics(
    measures_synth, thresholds_ref, definitions, use_doy=True, doy_mask=era_summer_mask
)
# metrics_dataset_synth.load()
metrics_synth_land = add_landmask(metrics_dataset_synth)
# metrics_synth_land.to_netcdf("era_hw_metrics_1950_2021_synth_doy_anom.nc")

############################
# observations (not synthetic)
# time period ALL 1950-2021, to match russo (but with slighty different threshold period--------------------------------
###########################
era_land_all = era_land.sel(
    time=slice("1949", "2021")
)  # need to start in 1949 to capture 1950, bc hemisphere processing

era_land_all_anom = (
    era_land_all.groupby("time.dayofyear") - ref_doy_climatology
).drop_vars("dayofyear")

era_land_all_anom["t2m_x"].attrs = {"units": "C"}  # hdp package needs units
# use c when dealing with anomalies (bc anomalies are the same in either units)

# calculate heatwave metrics on time period.
measures_all = hdp.measure.format_standard_measures(
    temp_datasets=[era_land_all_anom["t2m_x"]]
).chunk({"time": -1, "lat": 10, "lon": 10})
metrics_dataset_all = hdp.metric.compute_group_metrics(
    measures_all, thresholds_ref, definitions, use_doy=True, doy_mask=era_summer_mask
)
# metrics_dataset_all.compute()

metrics_all_land = add_landmask(metrics_dataset_all)
# metrics_all_land.to_netcdf("era_hw_metrics_1950_2021_doy_anom.nc")
