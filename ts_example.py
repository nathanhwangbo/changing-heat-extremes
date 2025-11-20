########################################################

# This is Figure 1 of the paper:
#    a time series example of the analysis pipeline
#    meant to illustrate how heatwaves are defined, and how we calculate metrics.
##########################################################

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import hvplot.xarray
import colorcet as cc
import matplotlib as mpl
import tastymap
import regionmask
import holoviews as hv
import glob
from xarray_einstats import stats  # wrapper around apply_ufunc for moments
import pandas as pd
import hvplot.pandas
from holoviews import opts


xr.set_options(use_new_combine_kwarg_defaults=True)

# hvplot.extension("matplotlib")
# hvplot.extension("bokeh")

rdbu_discrete = tastymap.cook_tmap("RdYlBu_r", num_colors=12).cmap
reds_discrete = tastymap.cook_tmap("cet_CET_L18", num_colors=11)[
    1:10
].cmap  # get rid of white
blues_discrete = tastymap.cook_tmap("blues", num_colors=10).cmap


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


##############################################################################
# Calculate mean differences (1986-2021) - (1950-1985) for heatwave metrics
##############################################################################

use_calendar_summer = True  # if true, use JJA as summer. else use dayofyear mask
if use_calendar_summer:
    hw_all = (
        xr.open_dataset("era_hw_metrics_1950_2021_anom.nc")
        .sel(percentile=0.9, definition="3-0-0")
        .drop_vars(["percentile", "definition"])
    )
else:
    hw_all = (
        xr.open_dataset("era_hw_metrics_1950_2021_anom_doy.nc")
        .sel(percentile=0.9, definition="3-0-0")
        .drop_vars(["percentile", "definition"])
    )

# compute deltas
hw_old = hw_all.sel(time=slice("1950", "1985"))
hw_new = hw_all.sel(time=slice("1986", "2021"))
hw_mean_diff = hw_new.mean(dim="time") - hw_old.mean(dim="time")


#######################################################################
# Calculate mean differences (1986-2021) - (1950-1985) for temperature
#######################################################################

# anomalies relative to 1950-1985, calculated in 0_era_medianshift.py
era_anom_path = "era_land_anom.nc"
era_land_anom = xr.open_dataset(era_anom_path)

# compute deltas-------------------------------------------
era_land_old = era_land_anom.sel(time=slice("1950", "1985"))
era_land_new = era_land_anom.sel(time=slice("1986", "2021"))
tmax_mean_diff = (era_land_new.mean(dim="time") - era_land_old.mean(dim="time")).rename(
    {"t2m_x": "t2m_x_mean_diff"}
)


##############################################
# Calculate climatological (1950-1985) moments
# NOTE! these are moments of the *doy anomalies* wrt to (1950-1985), i.e. mean 0 over this period
##############################################

clim_skew = stats.skew(era_land_old["t2m_x"], dims=["time"]).rename("t2m_x_skew")
clim_kurt = stats.kurtosis(era_land_old["t2m_x"], dims=["time"]).rename("t2m_x_kurt")
clim_var = era_land_old["t2m_x"].var(dim="time").rename("t2m_x_var")
clim_ar1 = xr.corr(
    era_land_old["t2m_x"], era_land_old["t2m_x"].shift(time=1), dim="time"
).rename("t2m_x_ar1")

climatology_stats = xr.merge([clim_skew, clim_kurt, clim_var, clim_ar1])


##############################################
# combine maps into 1 xr.dataset
##############################################

combined_ds = xr.merge([tmax_mean_diff, climatology_stats, hw_mean_diff], join="exact")


#############################################
# now pull out a single gridcell for a case study
#  I'm going to pull out los angeles (because that's where I live :))
#############################################
