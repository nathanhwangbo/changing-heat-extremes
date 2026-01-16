# analyzing the output of 0_era_meanshift.py


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
from holoviews import opts
import cftime
import bokeh
import string
from functools import partial


# first figure had 6 panels
# this figure has 9
scale = 9 / 6
title_size = 16 * scale
label_size = 14 * scale
tick_size = 10 * scale


fwidth = 400
fheight = 150


xr.set_options(use_new_combine_kwarg_defaults=True)

hvplot.extension("bokeh")

use_calendar_summer = True  # if true, use JJA as summer. else use dayofyear mask

ref_years = [1960, 1990]  # the time period the thresholds are calculated over
new_years = [1995, 2025]  # the time period we're gonna compare to
# ref_years = [1950, 1987]
# new_years = [1988, 2025]

if use_calendar_summer:
    suffix = "jja"  # used for labeling plots
    hw_obs = xr.open_dataset(
        f"era_hw_metrics_{ref_years[0]}_{new_years[1]}_anom.nc"
    ).sel(percentile=0.9, definition="3-0-0")
    hw_synth = xr.open_dataset(
        f"era_hw_metrics_{ref_years[0]}_{new_years[1]}_synth_anom.nc"
    ).sel(percentile=0.9, definition="3-0-0")
else:
    suffix = "doy"  # used for labeling plots
    hw_obs = xr.open_dataset(
        f"era_hw_metrics_{ref_years[0]}_{new_years[1]}_anom_doy.nc"
    ).sel(percentile=0.9, definition="3-0-0")
    hw_synth = xr.open_dataset(
        f"era_hw_metrics_{ref_years[0]}_{new_years[1]}_synth_anom_doy.nc"
    ).sel(percentile=0.9, definition="3-0-0")

################################
# reproduce russo et al Figure 2
################################

# russo et al uses something close to hot_r for cmap
# reds_discrete = tastymap.cook_tmap("cet_CET_L18", num_colors=10).cmap
reds_discrete = tastymap.cook_tmap("cet_CET_L18", num_colors=11)[
    1:11
].cmap  # get rid of white


rdbu_discrete = tastymap.cook_tmap("RdYlBu_r", num_colors=12).cmap

##########################################################

# instead of looking at trends over time, look at simple change between the two periods
# mean(1986-2021) - mean(1950-1985)

###########################################################


def get_delta_fig(
    mean_diff_ds,
    label_source,
    label_summer,
    ref_years,
    new_years,
    clim_hwf=(0, 15),
    clim_hwd=(-6, 6),
    clim_heatsum=(0, 25),
    clim_max=(-3, 3),
    cmap_hwf=reds_discrete,
    cmap_hwd=rdbu_discrete,
    cmap_heatsum=reds_discrete,
):
    """
    Creates figures looking at changes in difference in mean heatwave characteristics over two time period
    mean_diff_ds is an xarray dataset, the output of hdp.metric.compute_group_metrics, manipulated to represent the difference in metrics across two time periods
    label_source is a string (intended either "synthetic" or "observed") describing the source of the more recent time period used to calculate of mean_diff_ds
    label_summer is a string (intended either "jja" or "doy") describing how summer was defined in mean_diff_ds
    """
    hwf_delta = mean_diff_ds["t2m_x.t2m_x_threshold.HWF"]
    deltamap_hwf = hwf_delta.hvplot(
        projection=ccrs.PlateCarree(),
        coastline=True,
        cmap=cmap_hwf,
        clim=clim_hwf,
        # title=f"delta in heatwave frequency ({label_summer})\nmean({label_source} {new_years[0]}:{new_years[1]}) - mean(obs {ref_years[0]}:{ref_years[1]})",
        title="Change in Frequency",
        clabel="days",
        xlabel="",
        ylabel="",
    ).opts(ylim=(-59, None))

    # delta in hwd
    hwd_delta = mean_diff_ds["t2m_x.t2m_x_threshold.HWD"]
    deltamap_hwd = hwd_delta.hvplot(
        projection=ccrs.PlateCarree(),
        coastline=True,
        cmap=cmap_hwd,
        clim=clim_hwd,
        # title=f"delta in heatwave duration ({label_summer})\nmean({label_source} {new_years[0]}:{new_years[1]}) - mean(obs {ref_years[0]}:{ref_years[1]})",
        title="Change in Duration",
        clabel="days",
        xlabel="",
        ylabel="",
    ).opts(ylim=(-59, None))

    # # delta in avi
    # avi_delta = mean_diff_ds["t2m_x.t2m_x_threshold.AVI"]
    # deltamap_avi = avi_delta.hvplot(
    #     projection=ccrs.PlateCarree(),
    #     coastline=True,
    #     cmap=rdbu_discrete,
    #     clim=(-1, 1),
    #     title=f"delta in average intensity ({label_summer})\nmean({label_source} {new_years[0]}:{new_years[1]}) - mean(obs {ref_years[0]}:{ref_years[1]})",
    #     clabel="degC anomaly",
    #     xlabel="",
    #     ylabel="",
    # ).opts(fontscale=2.5, ylim=(-60, None))

    # delta in heatsum
    heatsum_delta = mean_diff_ds["t2m_x.t2m_x_threshold.sumHeat"]
    deltamap_heatsum = heatsum_delta.hvplot(
        projection=ccrs.PlateCarree(),
        coastline=True,
        cmap=cmap_heatsum,
        clim=clim_heatsum,
        # title=f"delta in cumulative heat ({label_summer})\nmean({label_source} {new_years[0]}:{new_years[1]}) - mean(obs {ref_years[0]}:{ref_years[1]})",
        title="Change in Cumulative Heat",
        clabel="degC-days",
        xlabel="",
        ylabel="",
    ).opts(ylim=(-59, None))

    # # delta in max seasonal temp
    # max_delta = mean_diff_ds["t2m_x.t2m_x_threshold.MAX"]
    # deltamap_max = max_delta.hvplot(
    #     projection=ccrs.PlateCarree(),
    #     coastline=True,
    #     cmap=rdbu_discrete,
    #     clim=clim_max,
    #     # title=f"delta in seasonal max ({label_summer})\nmean({label_source} {new_years[0]}:{new_years[1]}) - mean(obs {ref_years[0]}:{ref_years[1]})",
    #     title="Change in seasonal max",
    #     clabel="degC anomaly",
    #     xlabel="",
    #     ylabel="",
    # ).opts(fontscale=2.5, ylim=(-60, None))

    # combine
    fig_delta = (deltamap_hwf + deltamap_hwd + deltamap_heatsum).cols(1)

    return fig_delta


# ERA observed --------------------------------------
hw_old_obs = hw_obs.sel(time=slice(str(ref_years[0]), str(ref_years[1])))
hw_new_obs = hw_obs.sel(time=slice(str(new_years[0]), str(new_years[1])))
mean_diff_obs = hw_new_obs.mean(dim="time") - hw_old_obs.mean(dim="time")
fig_delta_obs = get_delta_fig(
    mean_diff_obs,
    label_source="obs",
    label_summer=suffix,
    ref_years=ref_years,
    new_years=new_years,
    # cmap_hwf=reds_discrete_odd,
)
# manually fix the tickers on hwf
hwf_ticks = np.linspace(0, 15, 11)[::2]
# manually fix the tickers on hwf
fig_delta_obs[0].map(
    lambda x: x.opts(
        colorbar_opts={"ticker": bokeh.models.FixedTicker(ticks=hwf_ticks)}
    ),
    hv.Image,
)

# make sure order matches get_delta_fig!
var_list = ["HWF", "HWD", "sumHeat"]  # , "MAX"]

# add in some text
fig_delta_obs = hv.Layout(
    [
        (
            fig_delta_obs[i]
            * hv.Text(
                -180 + 220,
                -60 + 10,
                f"Global Mean={str(mean_diff_obs[f't2m_x.t2m_x_threshold.{var_list[i]}'].mean().values.round(2))}",
                fontsize=label_size - 2,
            )
        ).opts(ylim=(-59, None))
        for i in range(len(var_list))
    ]
)

# hvplot.save(fig_delta_obs, f"fig_delta_obs_anom_{suffix}_ref{ref_years[0]}_{ref_years[1]}.html")


# ERA synthetic second half -------------------------------------
hw_old_synth = hw_synth.sel(time=slice(str(ref_years[0]), str(ref_years[1])))
hw_new_synth = hw_synth.sel(time=slice(str(new_years[0]), str(new_years[1])))
mean_diff_synth = hw_new_synth.mean(dim="time") - hw_old_synth.mean(dim="time")
fig_delta_synth_init = get_delta_fig(
    mean_diff_synth,
    label_source="synth",
    label_summer=suffix,
    ref_years=ref_years,
    new_years=new_years,
    # cmap_hwf=reds_discrete_odd,
)
# manually fix the tickers on hwf
fig_delta_synth_init[0].map(
    lambda x: x.opts(
        colorbar_opts={
            "ticker": bokeh.models.FixedTicker(ticks=hwf_ticks),
        }
    ),
    hv.Image,
)


### also calculate the correlations for each, and add as labels ---
cor_obs_synth = xr.combine_by_coords(
    [
        xr.corr(
            mean_diff_obs[var],
            mean_diff_synth[var],
        )
        for var in mean_diff_obs.data_vars
    ]
)


fig_delta_synth = hv.Layout(
    [
        (
            fig_delta_synth_init[i]
            * hv.Text(
                -180 + 35,
                -60 + 10,
                f"r={str(cor_obs_synth[f't2m_x.t2m_x_threshold.{var_list[i]}'].values.round(2))}",
                fontsize=label_size - 2,
            )
            * hv.Text(
                -180 + 220,
                -60 + 10,
                f"Global Mean={str(mean_diff_synth[f't2m_x.t2m_x_threshold.{var_list[i]}'].mean().values.round(2))}",
                fontsize=label_size - 2,
            )
        ).opts(ylim=(-59, None))
        for i in range(len(var_list))
    ]
)
# hvplot.save(fig_delta_synth, f"fig_delta_synth_anom_{suffix}_ref{ref_years[0]}_{ref_years[1]}.html")

# difference between observed and synthetic -------------------
obs_minus_synth = mean_diff_obs - mean_diff_synth
fig_obs_minus_synth_init = get_delta_fig(
    obs_minus_synth,
    label_source="obs - synth",
    label_summer=suffix,
    ref_years=ref_years,
    new_years=new_years,
    clim_hwf=(-10, 10),
    clim_hwd=(-3, 3),
    clim_heatsum=(-30, 30),
    clim_max=(-2, 2),
    cmap_hwf=rdbu_discrete,
    cmap_hwd=rdbu_discrete,
    cmap_heatsum=rdbu_discrete,
)

# add in mean absolute error over the map
mean_abs_diff = abs(obs_minus_synth).mean()

fig_obs_minus_synth = hv.Layout(
    [
        (
            fig_obs_minus_synth_init[i]
            * hv.Text(
                -180 + 52,
                -60 + 10,
                f"MAE={str(mean_abs_diff[f't2m_x.t2m_x_threshold.{var_list[i]}'].values.round(2))}",
                fontsize=label_size - 2,
            )
        ).opts(ylim=(-59, None), xlim=(-180, 180))
        for i in range(len(var_list))
    ]
)


# manually update the labels on this one
fig_obs_minus_synth[0].opts(title=f"obs - synth ({suffix})")
fig_obs_minus_synth[1].opts(title=f"obs - synth ({suffix})")
fig_obs_minus_synth[2].opts(title=f"obs - synth ({suffix})")
# fig_obs_minus_synth[3].opts(title=f"obs - synth ({suffix})")


# stitch all together into a single figure -------------------------
fig1 = fig_delta_obs[0] + fig_delta_synth[0] + fig_obs_minus_synth[0]
for i in np.arange(1, len(fig_delta_obs)).tolist():
    fig1 += fig_delta_obs[i] + fig_delta_synth[i] + fig_obs_minus_synth[i]


# add subplot labels
def subplot_label_hook(plot, element, sub_label=""):
    """add subplot labels (a, b, c...)"""
    # Access the underlying Bokeh figure
    fig = plot.state

    original_title = fig.title.text
    fig.title.text = f"{sub_label} {original_title}"


# iterate over the subplots and add the label to the title.
updated_fig1list = []
# weird ordering bc I want to go vertical instead of horizontal
# letter ordering = string.ascii_lowercase[i]
letter_ordering = ["a", "d", "g", "b", "e", "h", "c", "f", "i"]
for i, subplot in enumerate(fig1):
    new_label = f"({letter_ordering[i]})"  # this sets the format to (a), (b), ..
    updated_subplot = subplot.opts(
        hooks=[partial(subplot_label_hook, sub_label=new_label)]
    )
    updated_fig1list.append(updated_subplot)

fig1_updated = hv.Layout(updated_fig1list)

fig1_updated.cols(3).opts(shared_axes=False)

####################
# Final figure! ----
####################

# update of all maps here.
fig1_updated.map(
    lambda x: x.opts(
        xticks=0,
        yticks=0,
        xlabel="",
        xlim=(-180, 180),
    ),
    hv.Image,
).map(lambda x: x.opts(xlabel=""), hv.Text)

fig1_final = fig1_updated.map(
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
    [hv.Image, hv.Text],
)


# hvplot.save(fig1_final, f"figures\\fig_meanshift_{suffix}_ref{ref_years[0]}_{ref_years[1]}.png")
