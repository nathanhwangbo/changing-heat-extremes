########################################################

# This is SI Figure 1 of the paper:
#    a time series example of the analysis pipeline
#    meant to illustrate how heatwaves are defined, and how we calculate metrics.
##########################################################

from changing_heat_extremes import analysis_helpers as ahelpers
import numpy as np
import xarray as xr
import tastymap
import glob
import hvplot.xarray  # noqa: F401
from holoviews import opts
import holoviews as hv
import hdp


title_size = 16
label_size = 14
tick_size = 10
fwidth = 400
fheight = 150


rdbu_discrete = tastymap.cook_tmap("RdYlBu_r", num_colors=12).cmap
reds_discrete = tastymap.cook_tmap("cet_CET_L18", num_colors=11)[1:10].cmap  # get rid of white
blues_discrete = tastymap.cook_tmap("blues", num_colors=10).cmap


#######################################
# read in ERA data
#######################################

era_filelist = glob.glob("/mnt/media-drive/data/ERA5/t2m_x_1x1/*.nc")
era = xr.open_mfdataset(era_filelist).rename({"__xarray_dataarray_variable__": "t2m_x", "valid_time": "time"})


# fixing formatting for the hdp package
era["t2m_x"].attrs = {"units": "K"}  # hdp package needs units
# era = era.convert_calendar(calendar="standard", use_cftime=True)  # .compute()
era = era.convert_calendar(calendar="noleap", use_cftime=True)
era = era.sel(lat=slice(-60, 80)).chunk({"time": -1, "lat": 10, "lon": 10})  # matching karen's doy mask

# convert to (-180, 180) lon. specific to our use case
era = era.assign_coords(lon=(((era.lon + 180) % 360) - 180)).sortby("lon")

# add landmask
era_land = ahelpers.add_landmask(era).compute()


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
hw_old = hw_all.sel(time=slice(str(ref_years[0]), str(ref_years[1])))
hw_new = hw_all.sel(time=slice(str(new_years[0]), str(new_years[1])))
hw_mean_diff = hw_new.mean(dim="time") - hw_old.mean(dim="time")


#######################################################################
# Calculate mean differences (new_years) - (ref_years) for temperature
#######################################################################

# anomalies calculated in 0_era_meanshift.py
# note: this dataset is standardized to have mean zero across the whole period, AND has doy climatology removed

# TODO: should split up era_land_anom calculation into multiple files. is curently 10gb
era_land_anom = xr.open_dataset("era_land_anom.nc")

# compute deltas-------------------------------------------
era_land_old = era_land_anom.sel(time=slice(str(ref_years[0]), str(ref_years[1])))
era_land_new = era_land_anom.sel(time=slice(str(new_years[0]), str(new_years[1])))
tmax_mean_diff = (era_land_new.mean(dim="time") - era_land_old.mean(dim="time")).rename({"t2m_x": "t2m_x_mean_diff"})


# ##############################################
# # Calculate climatological (1950-1985) moments
# # NOTE! these are moments of the *doy anomalies* wrt to (1950-1985), i.e. mean 0 over this period
# ##############################################

# clim_skew = stats.skew(era_land_old["t2m_x"], dims=["time"]).rename("t2m_x_skew")
# clim_kurt = stats.kurtosis(era_land_old["t2m_x"], dims=["time"]).rename("t2m_x_kurt")
# clim_var = era_land_old["t2m_x"].var(dim="time").rename("t2m_x_var")
# clim_ar1 = xr.corr(
#     era_land_old["t2m_x"], era_land_old["t2m_x"].shift(time=1), dim="time"
# ).rename("t2m_x_ar1")

# climatology_stats = xr.merge([clim_skew, clim_kurt, clim_var, clim_ar1])


##############################################
# combine maps into 1 xr.dataset
##############################################

# combined_ds = xr.merge([tmax_mean_diff, climatology_stats, hw_mean_diff], join="exact")
combined_ds = xr.merge([tmax_mean_diff, hw_mean_diff], join="exact")


#############################################
# now pull out a single gridcell for a case study
#  I'm going to pull out los angeles (because that's where I live :))
#############################################

la_lat = 34
la_lon = -118

la_tmax_ref_da = era_land.sel(lat=la_lat, lon=la_lon, method="nearest").sel(time=slice("1960", "1990"))
la_tmax_anom_ref_da = era_land_anom.sel(lat=la_lat, lon=la_lon, method="nearest").sel(time=slice("1960", "1990"))

thresholds_la = thresholds_ref.sel(lat=la_lat, lon=la_lon, method="nearest")
# la_summary_ds = combined_ds.sel(lat=la_lat, lon=la_lon, method="nearest")
hw_la = hw_all.sel(lat=la_lat, lon=la_lon, method="nearest")

# raw tmax time series -------------------
fig_tmax = la_tmax_ref_da.hvplot(
    # title="(a) Daily Maximum Temperature in Los Angeles, 1960-1990",
    title="(a) Daily Maximum Temperature",
    xlabel="",
    ylabel="Daily Max T (K)",
)

# anomalies --------------------------
fig_tmax_anom = la_tmax_anom_ref_da.hvplot(
    # title="(b) Removing the day of year climatology (across 1960-1990)",
    title="(b) Climatology removed",
    xlabel="",
    ylabel="T Anomaly (K)",
)


# q90 threshold, for june 15 ------------------------

# pull out days for threshold, arbitrarily choosing june 15
# 7 days centered at june 15 is june 12 - june 18
# june 12 is day 163 in a no-leap calendar
june12_18 = la_tmax_anom_ref_da.where(la_tmax_anom_ref_da["time.dayofyear"].isin(np.arange(163, 169 + 1)), drop=True)
june15_threshold = (
    june12_18["t2m_x"].quantile(0.9).values
)  # approx equal to thresholds_la.sel(doy=165).values, up to smoothing

# there are 7 days * 31 years = 217 values in this histogram
fig_june15_threshold = (
    june12_18.hvplot.hist(
        normed=True,
        legend=False,
        # title="(c) Q90 threshold for June 15\nUsing 7 day window * 31 years = 217 days",
        title="(c) Q90 threshold for June 15",
        xlabel="Daily Max T Anomaly (K) ",
    )
    * june12_18.hvplot.density(filled=False, legend=False)
    * hv.VLine(x=june15_threshold)
)
fig_june15_threshold.opts(opts.VLine(color="red"))


# time series of (smoothed) thresholds --------------------------------

# jun1 is day 152 in noleap calendar
# aug 31 is day 243 in noleap calendar
vspan_jja = hv.VSpan(152, 243).opts(color="gray", alpha=0.2)

fig_threshold_ts = (
    thresholds_la.hvplot(
        color="red",
        title="(d) Q90 threshold for each day",
        xlabel="Day of Year",
        ylabel="T Anomaly (K)",
    )
    * vspan_jja
)


# showing the threshold

era_la_1995 = (
    era_land_anom.sel(lat=la_lat, lon=la_lon, method="nearest")
    .sel(time=slice("1995", "1995"))
    .assign_coords({"time": np.arange(0, 365)})
)
fig_1995_anom = era_la_1995.hvplot(
    # title="(e) 1995 daily max temperature anomaly\nalongside 90 threshold for all days",
    title="(e) 1995 Daily Max T Anomaly",
    xlabel="",
    ylabel="T Anomaly (K)",
)


# manually calculate the heatwave metrics for this year ----------------------
# should match hw_la['hwf']
hot_days_la = hdp.metric.indicate_hot_days.py_func(
    era_la_1995["t2m_x"].values, thresholds_la.values, era_la_1995.time.values
)
hw_ts_la = hdp.metric.index_heatwaves(hot_days_la, 3, 0, 0)
hwf_la = hdp.metric.heatwave_frequency.py_func(hw_ts_la, np.array([[151, 243]]))
hwn_la = hdp.metric.heatwave_number(hw_ts_la, np.array([[151, 243]]))
hwd_la = hdp.metric.heatwave_number(hw_ts_la, np.array([[151, 243]]))

# identify the exact heatwave days
# 151 and 243 are the days of the year for JJA
where_is_hw = np.where(hw_ts_la != 0)[0]
where_is_hw_jja = where_is_hw[(where_is_hw >= 151) & (where_is_hw <= 243)]

# manually mark two the two heatwave events
vspan_hw1 = hv.VSpan(207, 209).opts(color="red", alpha=0.2)
vspan_hw2 = hv.VSpan(213, 215).opts(color="red", alpha=0.2)


# get all of the metrics for this year ------------
# this should match above!
hw_1995 = hw_la.sel(time=slice("1995", "1995"))
hwf_1995 = hw_1995["t2m_x.t2m_x_threshold.HWF"].values[0].round(1)
hwd_1995 = hw_1995["t2m_x.t2m_x_threshold.HWD"].values[0].round(1)
sumheat_1995 = hw_1995["t2m_x.t2m_x_threshold.sumHeat"].values[0].round(1)

# check one:
ValueError(np.equal(hwf_1995, hwf_la), "methods don't match! check hwf")


# # <20 is a way to add a bunch of spaces
# text_1995 = hv.Text(
#     200, -16, f"HWF= {hwf_1995:<20} HWD= {hwd_1995:<20} sumHeat= {sumheat_1995:<20}"
# )

text_1995 = hv.Text(
    2,
    -15,
    f"HWF= {hwf_1995}\nHWD= {hwd_1995}\nsumHeat= {sumheat_1995}",
    halign="left",
    valign="bottom",
    fontsize=tick_size,
)


fig_1995 = fig_1995_anom * fig_threshold_ts * text_1995 * vspan_jja * vspan_hw1 * vspan_hw2
fig_1995


# show time series of the metrics

fig_hwf_la = hw_la["t2m_x.t2m_x_threshold.HWF"].hvplot(label="HWF", ylabel="Days", alpha=0.8, xlabel="")
fig_hwd_la = hw_la["t2m_x.t2m_x_threshold.HWD"].hvplot(label="HWD", ylabel="Days", alpha=0.8, xlabel="")
fig_sumheat_la = hw_la["t2m_x.t2m_x_threshold.sumHeat"].hvplot(
    label="sumHeat", ylabel="degC-days", alpha=0.8, xlabel=""
)

fig_hwf_la = fig_hwf_la.redim(**{"t2m_x.t2m_x_threshold.HWF": "Days"})
fig_hwd_la = fig_hwd_la.redim(**{"t2m_x.t2m_x_threshold.HWD": "Days"})


# fig_hw_la = (fig_hwf_la * fig_hwd_la * fig_sumheat_la).opts(multi_y=True)
fig_hw_la = (fig_hwf_la * fig_hwd_la * fig_sumheat_la).opts(
    multi_y=True,
    legend_position="top_left",
    title="(f) Heatwave Metrics",
)

combined_fig = (
    (fig_tmax + fig_tmax_anom + fig_june15_threshold + fig_threshold_ts + fig_1995 + fig_hw_la)
    .opts(shared_axes=False)
    .cols(2)
)
# combined_fig.map(lambda x: x.opts(fontscale=2), [hv.Curve, hv.Histogram])


final_fig = combined_fig.map(
    lambda x: x.options(
        fontsize={
            "title": title_size,
            "labels": label_size,
            "ticks": tick_size,
            "legend": tick_size,
        },
        frame_width=fwidth,
        frame_height=fheight,
    ),
    [hv.Curve, hv.Histogram, hv.Text],
)
