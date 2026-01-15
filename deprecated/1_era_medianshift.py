###################################
# DEPRECATED IN FAVOR OF 1_era_obs_delta.py
# analyzing the output of 0_era_medianshift.py
# a heatwave is defined as a period of at least 3 days with max daily temps over a given temperature threshold
# The temperature threshold varies by day, and is defined for day d as
# the 90th percentile of tmax across a 30-day window centered at 9, using years 1961 - 1990
# i.e. the 90th percentile across a 900-day histogram.
###################################3


import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from scipy.stats.mstats import theilslopes
import hvplot.xarray
import colorcet as cc
import matplotlib as mpl
import tastymap
import regionmask
import holoviews as hv

xr.set_options(use_new_combine_kwarg_defaults=True)


# hvplot.extension("bokeh")
use_calendar_summer = False  # if true, use MJJAS as summer. else use dayofyear mask
if use_calendar_summer:
    hw_all = xr.open_dataset("era_heatwave_metrics_1950_2021_anom.nc").sel(
        percentile=0.9, definition="3-0-0"
    )
    hw_synth = xr.open_dataset("era_heatwave_metrics_1950_2021_synth_anom.nc").sel(
        percentile=0.9, definition="3-0-0"
    )


else:
    hw_all = xr.open_dataset("era_heatwave_metrics_1950_2021_doy_anom.nc").sel(
        percentile=0.9, definition="3-0-0"
    )
    hw_synth = xr.open_dataset("era_heatwave_metrics_1950_2021_synth_doy_anom.nc").sel(
        percentile=0.9, definition="3-0-0"
    )
    # hw_all = xr.open_dataset("era_heatwave_metrics_1950_2021_doy.nc").sel(
    #     percentile=0.9, definition="3-0-0"
    # )
    # hw_synth = xr.open_dataset("era_heatwave_metrics_1950_2021_synth_doy.nc").sel(
    #     percentile=0.9, definition="3-0-0"
    # )


################################
# reproduce russo et al Figure 2
################################

# convert to (-180, 180) lon. specific to our use case
hw_all = hw_all.assign_coords(lon=(((hw_all.lon + 180) % 360) - 180)).sortby("lon")

# compute cumulative heats
hw_all["t2m_x.t2m_x_threshold.sumHeat"] = (
    hw_all["t2m_x.t2m_x_threshold.AVA"] * hw_all["t2m_x.t2m_x_threshold.HWF"]
)

# russo et al uses something close to hot_r for cmap

reds_discrete = tastymap.cook_tmap("cet_CET_L18", num_colors=10).cmap
rdbu_discrete = tastymap.cook_tmap("RdYlBu_r", num_colors=12).cmap

# need to calculate annual max

# average intensity (fig 2, bottom left)
avi_max_all = hw_all["t2m_x.t2m_x_threshold.AVI"].max(dim="time")
fig_avi_max = avi_max_all.hvplot(
    projection=ccrs.PlateCarree(),
    coastline=True,
    cmap=reds_discrete,
    clim=(-15, 45),
    title="max AVI (avg intensity), 1950-2021 ",
    clabel="deg C",
).opts(fontscale=1.5)

# average intensity anomaly (fig 2, bottom right)
ava_max_all = hw_all["t2m_x.t2m_x_threshold.AVA"].max(dim="time")
fig_ava_max = ava_max_all.hvplot(
    projection=ccrs.PlateCarree(),
    coastline=True,
    cmap=reds_discrete,
    clim=(0, 10),
    title="max AVA (avg intensity anom), 1950-2021 ",
    clabel="deg C above q90(doy, 1961-1990)",
).opts(fontscale=1.5)
# .opts(fontsize={
#     'title': '150%',
#     'labels': '150%',
#     'ticks': '200%',
# })


# cumulative heat (fig 2, top right)
heatsum_all = hw_all["t2m_x.t2m_x_threshold.AVA"] * hw_all["t2m_x.t2m_x_threshold.HWF"]

heatsum_max_all = heatsum_all.max(dim="time")
fig_heatsum_max = heatsum_max_all.hvplot(
    projection=ccrs.PlateCarree(),
    coastline=True,
    cmap=reds_discrete,
    clim=(-10, 360),
    title="max heatsum, 1950-2021 ",
    clabel="cumulative deg C over a year",
).opts(fontscale=1.5)


# number of heatwave days
hwd_max_all = hw_all["t2m_x.t2m_x_threshold.HWD"].max(dim="time")
fig_hwd_max = hwd_max_all.hvplot(
    projection=ccrs.PlateCarree(),
    coastline=True,
    cmap=reds_discrete,
    clim=(-5, 50),
    title="max HWD, 1950-2021",
    clabel="days",
).opts(fontscale=1.5)

# combine
fig2_russo = (fig_hwd_max + fig_heatsum_max + fig_avi_max + fig_ava_max).cols(2)
# hvplot.save(fig2_russo, "fig2_russo.html")

####################################
# reproduce Perkins Kirkpatrick 2020 Fig 1

# differences
# we're using ERA, vs their GHCN and Berk earth
# we're using 30 day period for threhold, they use 15

###################################


def get_theilsen_slope(y_data):
    """
    y_data (np.ndarray): a single time series
    """
    # x=None to theilslopes uses np.arange(len(y_data))
    slope, _, _, _ = theilslopes(y_data, x=None)

    return slope


def get_theilsen_slope_xr(da):
    """
    wrapper around get_theilsen_slope to get trend at each gridcell
    da (xr.dataarray): a data array with dim time
    """
    trends = xr.apply_ufunc(
        get_theilsen_slope,
        da,
        input_core_dims=[["time"]],
        output_core_dims=[[]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[np.float64],
    )
    return trends


def get_hw_trends(hw_ds):
    hwf_trend = get_theilsen_slope_xr(hw_ds["t2m_x.t2m_x_threshold.HWF"]) * 10
    hwd_trend = get_theilsen_slope_xr(hw_ds["t2m_x.t2m_x_threshold.HWD"]) * 10
    avi_trend = get_theilsen_slope_xr(hw_ds["t2m_x.t2m_x_threshold.AVI"]) * 10
    sumheat_trend = get_theilsen_slope_xr(hw_ds["t2m_x.t2m_x_threshold.sumHeat"]) * 10
    hw_trends = xr.merge([hwf_trend, hwd_trend, avi_trend, sumheat_trend], join="exact")
    return hw_trends


def get_trend_fig(hw_trends, caption=""):
    """
    hw_trends is output of get_hw_trends()
    """
    # mult by 10 to get units decade
    hwf_trend = hw_trends["t2m_x.t2m_x_threshold.HWF"]

    trendmap_hwf = hwf_trend.hvplot(
        projection=ccrs.PlateCarree(),
        coastline=True,
        cmap=rdbu_discrete,
        clim=(-10.5, 10.5),
        title=f"trend in heatwave frequency, {caption}",
        clabel="days per decade",
    ).opts(fontscale=1.5, ylim=(-60, None))

    # trend in hwd

    ## old approach, for OLS
    # xr is in nanosecs, need to convert to decade
    #  1e9 [ns/s] * 60 [s/m] * 60 [m/h] * 24 [h/d] * 365 [d/y] * 10 [y/dec]  = [ns / decade]
    # ns_per_decade = 1e9 * 60 * 60 * 24 * 365 * 10
    # hwd_trend_all = (
    #     hw_ds["t2m_x.t2m_x_threshold.HWD"]
    #     .polyfit(dim="time", deg=1)
    #     .polyfit_coefficients.sel(degree=1)
    #     * ns_per_decade
    # )

    hwd_trend = hw_trends["t2m_x.t2m_x_threshold.HWD"]
    trendmap_hwd = hwd_trend.hvplot(
        projection=ccrs.PlateCarree(),
        coastline=True,
        cmap=rdbu_discrete,
        clim=(-2.5, 2.5),
        title=f"trend in heatwave duration, {caption}",
        clabel="days per decade",
    ).opts(fontscale=1.5, ylim=(-60, None))

    # trend in avi
    avi_trend = hw_trends["t2m_x.t2m_x_threshold.AVI"]
    trendmap_avi = avi_trend.hvplot(
        projection=ccrs.PlateCarree(),
        coastline=True,
        cmap=rdbu_discrete,
        clim=(-1.2, 1.2),
        title=f"trend in average intensity, {caption}",
        clabel="degC per decade",
    ).opts(fontscale=1.5, ylim=(-60, None))

    # trend in heatsum
    heatsum_trend = hw_trends["t2m_x.t2m_x_threshold.sumHeat"]
    trendmap_heatsum = heatsum_trend.hvplot(
        projection=ccrs.PlateCarree(),
        coastline=True,
        cmap=rdbu_discrete,
        clim=(-10.5, 10.5),
        title=f"trend in cumulative heat, {caption}",
        clabel="degC per decade",
    ).opts(fontscale=1.5, ylim=(-60, None))

    # combine
    fig_trend = (trendmap_hwf + trendmap_hwd + trendmap_avi + trendmap_heatsum).cols(1)
    return fig_trend


# perkins-kirkpatric uses 1950-2014 -------------------------------------

hw_1950_2014 = xr.open_dataset("era_heatwave_metrics_1950_2014.nc").sel(
    percentile=0.9, definition="3-0-0"
)
# convert to (-180, 180) lon.
hw_1950_2014 = hw_1950_2014.assign_coords(
    lon=(((hw_1950_2014.lon + 180) % 360) - 180)
).sortby("lon")

# compute cumulative heats
hw_1950_2014["t2m_x.t2m_x_threshold.sumHeat"] = (
    hw_1950_2014["t2m_x.t2m_x_threshold.AVA"]
    * hw_1950_2014["t2m_x.t2m_x_threshold.HWF"]
)


hw_trends_1950_2014 = get_hw_trends(hw_1950_2014)
fig_trend = get_trend_fig(hw_trends_1950_2014, caption="1950-2014")
# hvplot.save(fig_trend, "fig1_perkins-kirkpatrick.html")


#################
# fig1 perkins kirkpatrick, but
# using median shifted second half of data
###################

# convert to (-180, 180) lon.
hw_synth = hw_synth.assign_coords(lon=(((hw_synth.lon + 180) % 360) - 180)).sortby(
    "lon"
)

# compute cumulative heat
hw_synth["t2m_x.t2m_x_threshold.sumHeat"] = (
    hw_synth["t2m_x.t2m_x_threshold.AVA"] * hw_synth["t2m_x.t2m_x_threshold.HWF"]
)

hw_trends_synth = get_hw_trends(hw_synth)
fig_trend_synth = get_trend_fig(
    hw_trends_synth, caption="1950-2021\n1986-2021 is median-shifted 1950-1985"
)
# hvplot.save(fig_trend_synth, "fig1_perkins-kirkpatrick_synth.html")


# compare to hw_all, which isn't median-shifted ---------
hw_all["t2m_x.t2m_x_threshold.sumHeat"] = (
    hw_all["t2m_x.t2m_x_threshold.AVA"] * hw_all["t2m_x.t2m_x_threshold.HWF"]
)
hw_trends_all = get_hw_trends(hw_all)
fig_trend_all = get_trend_fig(hw_trends_all, caption="1950-2021")
# hvplot.save(fig_trend_all, "fig1_perkins-kirkpatrick_anom.html")


# take difference
hw_trends_diff = hw_trends_all - hw_trends_synth
fig_trend_diff = get_trend_fig(hw_trends_diff, caption="(a) - (b)")

# (fig_trend_all + fig_trend_synth + fig_trend_diff). #weirdly, this works in rows instaed of cols

fig_trend_comparison = (
    (fig_trend_all[0] + fig_trend_synth[0] + fig_trend_diff[0])
    + (fig_trend_all[1] + fig_trend_synth[1] + fig_trend_diff[1])
    + (fig_trend_all[2] + fig_trend_synth[2] + fig_trend_diff[2])
    + (fig_trend_all[3] + fig_trend_synth[3] + fig_trend_diff[3])
).cols(3)

# hvplot.save(fig_trend_comparison, "fig1_perkins-kirkpatrick_synth_warm.html")

##############
# case study
# it looks like west+central europe stands out a bit in the difference plot
###############

ar6_land = regionmask.defined_regions.ar6.land.mask(hw_all)
bw_discrete = tastymap.cook_tmap("cet_CET_L1_r", num_colors=2).cmap
# (ar6_land == 17).hvplot(coastline = True, cmap = bw_discrete)

# ar6 region 7 is west and central europe
hw_eu = hw_all.where(ar6_land == 17)
hw_eu_ts = hw_eu.mean(dim=["lon", "lat"])

fig_hwf_eu_ts = hw_eu_ts.hvplot(
    x="time",
    y="t2m_x.t2m_x_threshold.HWF",
    title="heatwave frequency, west eu avg",
    ylabel="Frequency (days)",
).opts(fontscale=1.5, frame_width=400, frame_height=200)
fig_hwd_eu_ts = hw_eu_ts.hvplot(
    x="time",
    y="t2m_x.t2m_x_threshold.HWD",
    title="heatwave duration, west eu avg",
    ylabel="Duration (days)",
).opts(fontscale=1.5, frame_width=400, frame_height=200)
fig_avi_eu_ts = hw_eu_ts.hvplot(
    x="time",
    y="t2m_x.t2m_x_threshold.AVI",
    title="heatwave avg intensity, west eu avg",
    ylabel="Average Intensity (C)",
).opts(fontscale=1.5, frame_width=400, frame_height=200)
fig_sumheat_eu_ts = hw_eu_ts.hvplot(
    x="time",
    y="t2m_x.t2m_x_threshold.sumHeat",
    title="heatwave cumulative heat, west eu avg",
    ylabel="Cumulative Heat (C)",
).opts(fontscale=1.5, frame_width=400, frame_height=200)


fig_eu_ts = (fig_hwf_eu_ts + fig_hwd_eu_ts + fig_avi_eu_ts + fig_sumheat_eu_ts).cols(1)
hvplot.save(fig_eu_ts, "fig_eu_ts.html")


### repeat for synthetic time series

# ar6 region 7 is west and central europe
hw_eu_synth = hw_synth.where(ar6_land == 17)
hw_eu_synth_ts = hw_eu_synth.mean(dim=["lon", "lat"])

# draw a vertical line at 1986, start of new period
t_1986 = hw_eu_synth_ts.sel(time="1986").time.values
vline_1986 = hv.VLine(t_1986).opts(color="black", line_dash="dashed")

fig_hwf_eu_synth_ts = (
    hw_eu_synth_ts.hvplot(
        x="time",
        y="t2m_x.t2m_x_threshold.HWF",
        title="heatwave frequency, west eu avg\n1986-2021 is median-shifted '50-'85",
        ylabel="Frequency (days)",
    ).opts(fontscale=1.5, frame_width=400, frame_height=200)
    * vline_1986
)
fig_hwd_eu_synth_ts = (
    hw_eu_synth_ts.hvplot(
        x="time",
        y="t2m_x.t2m_x_threshold.HWD",
        title="heatwave duration, west eu avg\n1986-2021 is median-shifted '50-'85",
        ylabel="Duration (days)",
    ).opts(fontscale=1.5, frame_width=400, frame_height=200)
    * vline_1986
)
fig_avi_eu_synth_ts = (
    hw_eu_synth_ts.hvplot(
        x="time",
        y="t2m_x.t2m_x_threshold.AVI",
        title="heatwave avg intensity, west eu avg\n1986-2021 is median-shifted '50-'85",
        ylabel="Average Intensity (C)",
    ).opts(fontscale=1.5, frame_width=400, frame_height=200)
    * vline_1986
)
fig_sumheat_eu_synth_ts = (
    hw_eu_synth_ts.hvplot(
        x="time",
        y="t2m_x.t2m_x_threshold.sumHeat",
        title="heatwave cumulative heat, west eu avg\n1986-2021 is median-shifted '50-'85",
        ylabel="Cumulative Heat (C)",
    ).opts(fontscale=1.5, frame_width=400, frame_height=200)
    * vline_1986
)

fig_eu_synth_ts = (
    fig_hwf_eu_synth_ts
    + fig_hwd_eu_synth_ts
    + fig_avi_eu_synth_ts
    + fig_sumheat_eu_synth_ts
).cols(1)
hvplot.save(fig_eu_synth_ts, "fig_eu_synth_ts.html")

# look at the diff -----
hw_eu_diff_ts = hw_eu_ts - hw_eu_synth_ts

fig_hwf_eu_diff_ts = (
    hw_eu_diff_ts.hvplot(
        x="time",
        y="t2m_x.t2m_x_threshold.HWF",
        title="heatwave frequency (a) - (b)",
        ylabel="Frequency (days)",
    ).opts(fontscale=1.5, frame_width=400, frame_height=200)
    * vline_1986
)
fig_hwd_eu_diff_ts = (
    hw_eu_diff_ts.hvplot(
        x="time",
        y="t2m_x.t2m_x_threshold.HWD",
        title="heatwave duration (a) - (b)",
        ylabel="Duration (days)",
    ).opts(fontscale=1.5, frame_width=400, frame_height=200)
    * vline_1986
)
fig_avi_eu_diff_ts = (
    hw_eu_diff_ts.hvplot(
        x="time",
        y="t2m_x.t2m_x_threshold.AVI",
        title="heatwave avg intensity (a) - (b)",
        ylabel="Average Intensity (C)",
    ).opts(fontscale=1.5, frame_width=400, frame_height=200)
    * vline_1986
)
fig_sumheat_eu_diff_ts = (
    hw_eu_diff_ts.hvplot(
        x="time",
        y="t2m_x.t2m_x_threshold.sumHeat",
        title="heatwave cumulative heat (a) - (b)",
        ylabel="Cumulative Heat (C)",
    ).opts(fontscale=1.5, frame_width=400, frame_height=200)
    * vline_1986
)


fig_eu_diff_ts = (
    fig_hwf_eu_diff_ts
    + fig_hwd_eu_diff_ts
    + fig_avi_eu_diff_ts
    + fig_sumheat_eu_diff_ts
).cols(1)
# hvplot.save(fig_eu_diff_ts, "fig_eu_diff_ts.html")


# combined figure

fig_eu_ts_comparison = (
    (fig_eu_ts[0] + fig_eu_synth_ts[0] + fig_eu_diff_ts[0])
    + (fig_eu_ts[1] + fig_eu_synth_ts[1] + fig_eu_diff_ts[1])
    + (fig_eu_ts[2] + fig_eu_synth_ts[2] + fig_eu_diff_ts[2])
    + (fig_eu_ts[3] + fig_eu_synth_ts[3] + fig_eu_diff_ts[3])
).cols(3)


hvplot.save(fig_eu_ts_comparison, "fig_eu_ts_comparison.html")

# comparing synth post - pre ----------------------------------

hw_eu_old = hw_eu_synth_ts.sel(time=slice("1950", "1985"))
hw_eu_new_synth = hw_eu_synth_ts.sel(time=slice("1986", "2021")).assign_coords(
    time=hw_eu_old.time
)  # "pretend" time is same so we can subtract

# new minus old
hw_eu_synth_diff = hw_eu_new_synth - hw_eu_old


fig_hwf_eu_synthdiff_ts = hw_eu_synth_diff.hvplot(
    x="time",
    y="t2m_x.t2m_x_threshold.HWF",
    title="heatwave frequency, new - old",
    ylabel="Frequency (days)",
).opts(fontscale=1.5, frame_width=400, frame_height=200)
fig_hwd_eu_synthdiff_ts = hw_eu_synth_diff.hvplot(
    x="time",
    y="t2m_x.t2m_x_threshold.HWD",
    title="heatwave duration, new - old",
    ylabel="Duration (days)",
).opts(fontscale=1.5, frame_width=400, frame_height=200)
fig_avi_eu_synthdiff_ts = hw_eu_synth_diff.hvplot(
    x="time",
    y="t2m_x.t2m_x_threshold.AVI",
    title="heatwave avg intensity, new - old",
    ylabel="Average Intensity (C)",
).opts(fontscale=1.5, frame_width=400, frame_height=200)
fig_sumheat_eu_synthdiff_ts = hw_eu_synth_diff.hvplot(
    x="time",
    y="t2m_x.t2m_x_threshold.sumHeat",
    title="heatwave cumulative heat, new - old",
    ylabel="Cumulative Heat (C)",
).opts(fontscale=1.5, frame_width=400, frame_height=200)


fig_eu_synthdiff_ts = (
    fig_hwf_eu_synthdiff_ts
    + fig_hwd_eu_synthdiff_ts
    + fig_avi_eu_synthdiff_ts
    + fig_sumheat_eu_synthdiff_ts
).cols(1)

fig_eu_synthdiff_comparison = (
    (fig_eu_synth_ts[0] + fig_eu_synthdiff_ts[0])
    + (fig_eu_synth_ts[1] + fig_eu_synthdiff_ts[1])
    + (fig_eu_synth_ts[2] + fig_eu_synthdiff_ts[2])
    + (fig_eu_synth_ts[3] + fig_eu_synthdiff_ts[3])
).cols(2)  # .opts(shared_axes = False)
hvplot.save(fig_eu_synthdiff_comparison, "fig_eu_synthdiff_comparison.html")

# repeat globally, and combine space + time into histograms ------------------------

hw_old = hw_synth.sel(time=slice("1950", "1985"))
hw_new_synth = hw_synth.sel(time=slice("1986", "2020")).assign_coords(
    time=hw_old.time
)  # "pretend" time is same so we can subtract

# combine old and new into one xarray object
hw_year36 = xr.concat(
    [
        hw_old.expand_dims("period").assign_coords({"period": ["old"]}),
        hw_new_synth.expand_dims("period").assign_coords({"period": ["new"]}),
    ],
    dim="period",
)

fig_synth_hwf = hw_year36.hvplot.hist(
    y="t2m_x.t2m_x_threshold.HWF",
    by="period",
    title="heatwave frequency, over all land and 36 years",
    xlabel="days",
    bins=75,
    xlim=(-5, 50),
    alpha=0.5,
    normed=True,
).opts(fontscale=1.5)  # , frame_width=400, frame_height=200)

fig_synth_hwd = hw_year36.hvplot.hist(
    y="t2m_x.t2m_x_threshold.HWD",
    by="period",
    title="heatwave duration, over all land and 36 years",
    xlabel="days",
    bins=75,
    xlim=(-5, 50),
    alpha=0.5,
    normed=True,
).opts(fontscale=1.5)  # , frame_width=400, frame_height=200)

fig_synth_avi = hw_year36.hvplot.hist(
    y="t2m_x.t2m_x_threshold.AVI",
    by="period",
    title="heatwave avg intensity, over all land and 36 years",
    xlabel="degC",
    bins=50,
    xlim=(-5, 50),
    alpha=0.5,
    normed=True,
).opts(fontscale=1.5)  # , frame_width=400, frame_height=200)

fig_synth_sumheat = hw_year36.hvplot.hist(
    y="t2m_x.t2m_x_threshold.sumHeat",
    by="period",
    title="heatwave cumulative heat\n over all land and 36 years",
    xlabel="Cumulative Heat (C)",
    bins=100,
    xlim=(-5, 100),
    alpha=0.5,
    normed=True,
).opts(fontscale=1.5)  # , frame_width=400, frame_height=200)


fig_synth_oldnew = (
    fig_synth_hwf + fig_synth_hwd + fig_synth_avi + fig_synth_sumheat
).cols(1)

# also plot the diff
hw_synth_diff = hw_new_synth - hw_old

fig_synthdiff_hwf = hw_synth_diff.hvplot.hist(
    y="t2m_x.t2m_x_threshold.HWF",
    title="heatwave frequency, over all land and 36 years\nnew-old",
    xlabel="days",
    bins=75,
    xlim=(-5, 40),
    alpha=0.5,
    normed=True,
).opts(fontscale=1.5)  # , frame_width=400, frame_height=200)

fig_synthdiff_hwd = hw_synth_diff.hvplot.hist(
    y="t2m_x.t2m_x_threshold.HWD",
    title="heatwave duration, over all land and 36 years\nnew-old",
    xlabel="days",
    bins=75,
    xlim=(-5, 30),
    alpha=0.5,
    normed=True,
).opts(fontscale=1.5)  # , frame_width=400, frame_height=200)

fig_synthdiff_avi = hw_synth_diff.hvplot.hist(
    y="t2m_x.t2m_x_threshold.AVI",
    title="heatwave avg intensity, over all land and 36 years\nnew-old",
    xlabel="degC",
    bins=75,
    xlim=(-10, 50),
    alpha=0.5,
    normed=True,
).opts(fontscale=1.5)  # , frame_width=400, frame_height=200)

fig_synthdiff_sumheat = hw_synth_diff.hvplot.hist(
    y="t2m_x.t2m_x_threshold.sumHeat",
    title="heatwave cumulative heat\n over all land and 36 years\nnew-old",
    xlabel="Cumulative Heat (C)",
    bins=100,
    xlim=(-5, 75),
    alpha=0.5,
    normed=True,
).opts(fontscale=1.5)  # , frame_width=400, frame_height=200)


fig_synthdiff = (
    fig_synthdiff_hwf + fig_synthdiff_hwd + fig_synthdiff_avi + fig_synthdiff_sumheat
).cols(1)

fig_synthdiff_comparison = (
    (
        (fig_synth_oldnew[0] + fig_synthdiff[0])
        + (fig_synth_oldnew[1] + fig_synthdiff[1])
        + (fig_synth_oldnew[2] + fig_synthdiff[2])
        + (fig_synth_oldnew[3] + fig_synthdiff[3])
    )
    .cols(2)
    .opts(shared_axes=False)
)
hvplot.save(fig_synthdiff_comparison, "fig_synthdiff_comparison_doy.html")


############################

############################
test_lon = -118
test_lat = 34
b = hw_synth.sel(lon=test_lon, lat=test_lat, method="nearest")
bb = b.sel(time=slice("1950", "1985"))
bbb = b.sel(time=slice("1986", "2021")).assign_coords(time=bb.time)
(bb - bbb)["t2m_x.t2m_x_threshold.sumHeat"]


a = hw_eu_synth_ts.sel(time=slice("1950", "1985"))
aa = hw_eu_synth_ts.sel(time=slice("1986", "2021")).assign_coords(time=a.time)

(a - aa)["t2m_x.t2m_x_threshold.sumHeat"]


##########################################################

# instead of looking at trends over time, look at simple change between the two periods
# mean(1986-2021) - mean(1950-1985)

###########################################################

# ERA
hw_old_obs = hw_all.sel(time=slice("1950", "1985"))
hw_new_obs = hw_all.sel(time=slice("1986", "2021"))
mean_diff_obs = hw_new_obs.mean(dim="time") - hw_old_obs.mean(dim="time")

hwf_delta = mean_diff_obs["t2m_x.t2m_x_threshold.HWF"]
deltamap_hwf_obs = hwf_delta.hvplot(
    projection=ccrs.PlateCarree(),
    coastline=True,
    cmap=reds_discrete,
    clim=(0, 30),
    title="delta in heatwave frequency\nmean(1986:2021) - mean(1950:1985)",
    clabel="days",
).opts(fontscale=1.5, ylim=(-60, None))

# delta in hwd
hwd_delta = mean_diff_obs["t2m_x.t2m_x_threshold.HWD"]
deltamap_hwd_obs = hwd_delta.hvplot(
    projection=ccrs.PlateCarree(),
    coastline=True,
    cmap=rdbu_discrete,
    clim=(-6, 6),
    title="delta in heatwave duration\nmean(1986:2021) - mean(1950:1985)",
    clabel="days",
).opts(fontscale=1.5, ylim=(-60, None))

# delta in avi
avi_delta = mean_diff_obs["t2m_x.t2m_x_threshold.AVI"]
deltamap_avi_obs = avi_delta.hvplot(
    projection=ccrs.PlateCarree(),
    coastline=True,
    cmap=rdbu_discrete,
    clim=(-5, 5),
    title="delta in average intensity\nmean(1986:2021) - mean(1950:1985)",
    clabel="degC anomaly",
).opts(fontscale=1.5, ylim=(-60, None))

# delta in heatsum
heatsum_delta = mean_diff_obs["t2m_x.t2m_x_threshold.sumHeat"]
deltamap_heatsum_obs = heatsum_delta.hvplot(
    projection=ccrs.PlateCarree(),
    coastline=True,
    cmap=rdbu_discrete,
    clim=(-50.5, 50.5),
    title="delta in cumulative heat\nmean(1986:2021) - mean(1950:1985)",
    clabel="degC anomaly",
).opts(fontscale=1.5, ylim=(-60, None))

# combine
fig_delta_obs = (
    deltamap_hwf_obs + deltamap_hwd_obs + deltamap_avi_obs + deltamap_heatsum_obs
).cols(1)
hvplot.save(fig_delta_obs, "fig_delta_obs_anom_mjjas.html")

#### temp
test_lon = -60
test_lat = -18
a = hw_all["t2m_x.t2m_x_threshold.AVI"].sel(
    lat=test_lat, lon=test_lon, method="nearest"
)
a.hvplot()
a.sel(time=slice("1950", "1985")).mean()
a.sel(time=slice("1986", "2021")).mean()

# synthetic --------------------------------------------------------------------
hw_old_synth = hw_synth.sel(time=slice("1950", "1985"))  # should be same as hw_old_obs
hw_new_synth = hw_synth.sel(time=slice("1986", "2021"))
mean_diff_synth = hw_new_synth.mean(dim="time") - hw_old_synth.mean(dim="time")

hwf_delta_synth = mean_diff_synth["t2m_x.t2m_x_threshold.HWF"]
deltamap_hwf_synth = hwf_delta_synth.hvplot(
    projection=ccrs.PlateCarree(),
    coastline=True,
    cmap=reds_discrete,
    clim=(0, 30),
    title="delta in heatwave frequency\nmean(synthetic 1986:2021) - mean(1950:1985)",
    clabel="days",
).opts(fontscale=1.5, ylim=(-60, None))

# delta in hwd
hwd_delta_synth = mean_diff_synth["t2m_x.t2m_x_threshold.HWD"]
deltamap_hwd_synth = hwd_delta_synth.hvplot(
    projection=ccrs.PlateCarree(),
    coastline=True,
    cmap=rdbu_discrete,
    clim=(-6, 6),
    title="delta in heatwave duration\nmean(synthetic 1986:2021) - mean(1950:1985)",
    clabel="days",
).opts(fontscale=1.5, ylim=(-60, None))

# delta in avi
avi_delta_synth = mean_diff_synth["t2m_x.t2m_x_threshold.AVI"]
deltamap_avi_synth = avi_delta_synth.hvplot(
    projection=ccrs.PlateCarree(),
    coastline=True,
    cmap=rdbu_discrete,
    clim=(-12, 12),
    title="delta in average intensity\nmean(synthetic 1986:2021) - mean(1950:1985)",
    clabel="degC",
).opts(fontscale=1.5, ylim=(-60, None))

# delta in heatsum
heatsum_delta_synth = mean_diff_synth["t2m_x.t2m_x_threshold.sumHeat"]
deltamap_heatsum_synth = heatsum_delta_synth.hvplot(
    projection=ccrs.PlateCarree(),
    coastline=True,
    cmap=rdbu_discrete,
    clim=(-50.5, 50.5),
    title="delta in cumulative heat\nmean(synthetic 1986:2021) - mean(1950:1985)",
    clabel="degC",
).opts(fontscale=1.5, ylim=(-60, None))

# combine
fig_delta_synth = (
    deltamap_hwf_synth
    + deltamap_hwd_synth
    + deltamap_avi_synth
    + deltamap_heatsum_synth
).cols(1)
hvplot.save(fig_delta_synth, "fig_delta_synth.html")

#########
# synthetic V2, shared median --------------------------------------------------------------------
#########

hw_synth_sharedmedian = xr.open_dataset(
    "era_heatwave_metrics_1950_2021_synth_doy_singlemedian.nc"
).sel(percentile=0.9, definition="3-0-0")
# convert to (-180, 180) lon.
hw_synth_sharedmedian = hw_synth_sharedmedian.assign_coords(
    lon=(((hw_synth_sharedmedian.lon + 180) % 360) - 180)
).sortby("lon")

# compute cumulative heat
hw_synth_sharedmedian["t2m_x.t2m_x_threshold.sumHeat"] = (
    hw_synth_sharedmedian["t2m_x.t2m_x_threshold.AVA"]
    * hw_synth_sharedmedian["t2m_x.t2m_x_threshold.HWF"]
)


hw_old_synth_sharedmedian = hw_synth_sharedmedian.sel(
    time=slice("1950", "1985")
)  # should be same as hw_old_obs
hw_new_synth_sharedmedian = hw_synth_sharedmedian.sel(time=slice("1986", "2021"))
median_diff_synth_sharedmedian = hw_new_synth_sharedmedian.median(
    dim="time"
) - hw_old_synth_sharedmedian.median(dim="time")

hwf_delta_synth_sharedmedian = median_diff_synth_sharedmedian[
    "t2m_x.t2m_x_threshold.HWF"
]
deltamap_hwf_synth_sharedmedian = hwf_delta_synth_sharedmedian.hvplot(
    projection=ccrs.PlateCarree(),
    coastline=True,
    cmap=reds_discrete,
    clim=(0, 30),
    title="delta in heatwave frequency\nglobal median(synthetic 1986:2021) - median(1950:1985)",
    clabel="days",
).opts(fontscale=1.5, ylim=(-60, None))

# delta in hwd
hwd_delta_synth_sharedmedian = median_diff_synth_sharedmedian[
    "t2m_x.t2m_x_threshold.HWD"
]
deltamap_hwd_synth_sharedmedian = hwd_delta_synth_sharedmedian.hvplot(
    projection=ccrs.PlateCarree(),
    coastline=True,
    cmap=rdbu_discrete,
    clim=(-6, 6),
    title="delta in heatwave duration\nglobal median(synthetic 1986:2021) - median(1950:1985)",
    clabel="days",
).opts(fontscale=1.5, ylim=(-60, None))

# delta in avi
avi_delta_synth_sharedmedian = median_diff_synth_sharedmedian[
    "t2m_x.t2m_x_threshold.AVI"
]
deltamap_avi_synth_sharedmedian = avi_delta_synth_sharedmedian.hvplot(
    projection=ccrs.PlateCarree(),
    coastline=True,
    cmap=rdbu_discrete,
    clim=(-12, 12),
    title="delta in average intensity\nglobal median(synthetic 1986:2021) - median(1950:1985)",
    clabel="degC",
).opts(fontscale=1.5, ylim=(-60, None))

# delta in heatsum
heatsum_delta_synth_sharedmedian = median_diff_synth_sharedmedian[
    "t2m_x.t2m_x_threshold.sumHeat"
]
deltamap_heatsum_synth_sharedmedian = heatsum_delta_synth_sharedmedian.hvplot(
    projection=ccrs.PlateCarree(),
    coastline=True,
    cmap=rdbu_discrete,
    clim=(-50.5, 50.5),
    title="delta in cumulative heat\nglobal median(synthetic 1986:2021) - median(1950:1985)",
    clabel="degC",
).opts(fontscale=1.5, ylim=(-60, None))

# combine
fig_delta_synth_sharedmedian = (
    deltamap_hwf_synth_sharedmedian
    + deltamap_hwd_synth_sharedmedian
    + deltamap_avi_synth_sharedmedian
    + deltamap_heatsum_synth_sharedmedian
).cols(1)
hvplot.save(fig_delta_synth_sharedmedian, "fig_delta_synth_sharedmedian.html")
