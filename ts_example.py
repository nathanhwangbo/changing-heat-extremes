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
import panel as pn

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


#######################################
# read in ERA data
#######################################

era_filelist = glob.glob("D:data\\ERA5\\t2m_x_1x1_2025\\*.nc")
era = xr.open_mfdataset(era_filelist).rename({"t2m": "t2m_x"})


# fixing formatting for the hdp package
era["t2m_x"].attrs = {"units": "K"}  # hdp package needs units
# era = era.convert_calendar(calendar="standard", use_cftime=True)  # .compute()
era = era.convert_calendar(calendar="noleap", use_cftime=True)
era = era.sel(lat=slice(-60, 80)).chunk(
    {"time": -1, "lat": 10, "lon": 10}
)  # matching karen's doy mask

# convert to (-180, 180) lon. specific to our use case
era = era.assign_coords(lon=(((era.lon + 180) % 360) - 180)).sortby("lon")

# add landmask
era_land = add_landmask(era).compute()


# grab thresholds from 0_era_meanshift.py ---------------------------------------
thresholds_ref = xr.open_dataarray("thresholds_ref.nc").sel(percentile=0.9)


##############################################################################
# Calculate mean differences (1986-2021) - (1950-1985) for heatwave metrics
##############################################################################

ref_years = [1960, 1990]  # the time period the thresholds are calculated over
new_years = [1995, 2025]  # the time period we're gonna compare to

use_calendar_summer = True  # if true, use JJA as summer. else use dayofyear mask
if use_calendar_summer:
    hw_all = (
        xr.open_dataset(f"era_hw_metrics_{ref_years[0]}_{new_years[1]}_anom.nc")
        .sel(percentile=0.9, definition="3-0-0")
        .drop_vars(["percentile", "definition"])
    )

else:
    hw_all = (
        xr.open_dataset(f"era_hw_metrics_{ref_years[0]}_{new_years[1]}_anom_doy.nc")
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

la_lat = 34
la_lon = -118

la_tmax_ref_da = era_land.sel(lat=la_lat, lon=la_lon, method="nearest").sel(
    time=slice("1960", "1990")
)
la_tmax_anom_ref_da = era_land_anom.sel(lat=la_lat, lon=la_lon, method="nearest").sel(
    time=slice("1960", "1990")
)

thresholds_la = thresholds_ref.sel(lat=la_lat, lon=la_lon, method="nearest")
# la_summary_ds = combined_ds.sel(lat=la_lat, lon=la_lon, method="nearest")
hw_la = hw_all.sel(lat=la_lat, lon=la_lon, method="nearest")

# raw tmax time series -------------------
fig_tmax = la_tmax_ref_da.hvplot(
    title="Daily Maximum Temperature in Los Angeles, 1960-1990",
    xlabel="",
    ylabel="Daily Max T (K)",
)

# anomalies --------------------------
fig_tmax_anom = la_tmax_anom_ref_da.hvplot(
    title="Removing the day of year climatology (across 1960-1990)",
    xlabel="",
    ylabel="Daily Max T Anomaly (K)",
)


# q90 threshold, for june 15 ------------------------

# pull out days for threshold, arbitrarily choosing june 15
# 7 days centered at june 15 is june 12 - june 18
# june 12 is day 163 in a no-leap calendar
june12_18 = la_tmax_anom_ref_da.where(
    la_tmax_anom_ref_da["time.dayofyear"].isin(np.arange(163, 169 + 1)), drop=True
)
june15_threshold = (
    june12_18["t2m_x"].quantile(0.9).values
)  # approx equal to thresholds_la.sel(doy=165).values, up to smoothing

# there are 7 days * 31 years = 217 values in this histogram
fig_june15_threshold = (
    june12_18.hvplot.hist(
        normed=True,
        legend=False,
        title="Q90 threshold for June 15 \nUsing 7 day window * 31 years = 217 days",
        xlabel="Daily Max T Anomaly",
    )
    * june12_18.hvplot.density(filled=False, legend=False)
    * hv.VLine(x=june15_threshold)
)
fig_june15_threshold.opts(opts.VLine(color="red"))


# time series of (smoothed) thresholds --------------------------------

fig_threshold_ts = thresholds_la.hvplot(
    color="red",
    title="Q90 threshold for each day",
    xlabel="Day of Year",
    ylabel="Daily Max T Anomaly",
)


# showing the threshold
fig_1991_anom = (
    era_land_anom.sel(lat=la_lat, lon=la_lon, method="nearest")
    .sel(time=slice("1991", "1991"))
    .assign_coords({"time": np.arange(0, 365)})
    .hvplot(
        title="1991 daily max temperature anomaly\nalongside 90 threshold for all days",
        xlabel="",
        ylabel="Daily Max T Anomaly",
    )
)

# get all of the metrics for this year
# I should check these -- esp check if hwd subtracts the 3 days
# note that a heatwave is at least 3 consecutive days above
hw_1991 = hw_la.sel(time=slice("1991", "1991"))
hwf_1991 = hw_1991["t2m_x.t2m_x_threshold.HWF"].values[0].round(1)
hwd_1991 = hw_1991["t2m_x.t2m_x_threshold.HWD"].values[0].round(1)
sumheat_1991 = hw_1991["t2m_x.t2m_x_threshold.sumHeat"].values[0].round(1)


# <20 is a way to add a bunch of spaces
text_1991 = hv.Text(
    200, -13, f"HWF= {hwf_1991:<20} HWD= {hwd_1991:<20} sumHeat= {sumheat_1991:<20}"
)
fig_1991 = fig_1991_anom * fig_threshold_ts * text_1991
fig_1991


# show time series of the metrics

fig_hwf_la = hw_la["t2m_x.t2m_x_threshold.HWF"].hvplot(
    label="HWF", ylabel="Days", alpha=0.8, xlabel=""
)
fig_hwd_la = hw_la["t2m_x.t2m_x_threshold.HWD"].hvplot(
    label="HWD", ylabel="Days", alpha=0.8, xlabel=""
)
fig_sumheat_la = hw_la["t2m_x.t2m_x_threshold.sumHeat"].hvplot(
    label="sumHeat", ylabel="K anomaly", alpha=0.8, xlabel=""
)

fig_hwf_la = fig_hwf_la.redim(**{"t2m_x.t2m_x_threshold.HWF": "Days"})
fig_hwd_la = fig_hwd_la.redim(**{"t2m_x.t2m_x_threshold.HWD": "Days"})


# fig_hw_la = (fig_hwf_la * fig_hwd_la * fig_sumheat_la).opts(multi_y=True)
fig_hw_la = (fig_hwf_la * fig_hwd_la * fig_sumheat_la).opts(
    multi_y=True, legend_position="top_left"
)


combined_fig = (
    (
        fig_tmax
        + fig_tmax_anom
        + fig_june15_threshold
        + fig_threshold_ts
        + fig_1991
        + fig_hw_la
    )
    .opts(shared_axes=False)
    .cols(2)
)


# hvplot.save(combined_fig, "fig_ts_example.png")
