# analyzing the output of 0_era_medianshift.py


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
    clim_heatsum=(0, 50.5),
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
        title=f"Change in Heatwave Frequency",
        clabel="days",
        xlabel="",
        ylabel="",
    ).opts(fontscale=2.5, ylim=(-60, None))

    # delta in hwd
    hwd_delta = mean_diff_ds["t2m_x.t2m_x_threshold.HWD"]
    deltamap_hwd = hwd_delta.hvplot(
        projection=ccrs.PlateCarree(),
        coastline=True,
        cmap=cmap_hwd,
        clim=clim_hwd,
        # title=f"delta in heatwave duration ({label_summer})\nmean({label_source} {new_years[0]}:{new_years[1]}) - mean(obs {ref_years[0]}:{ref_years[1]})",
        title="Change in Heatwave Duration",
        clabel="days",
        xlabel="",
        ylabel="",
    ).opts(fontscale=2.5, ylim=(-60, None))

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
        clabel="degC anomaly",
        xlabel="",
        ylabel="",
    ).opts(fontscale=2.5, ylim=(-60, None))

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
)

# also calculate the correlations for each, and add as labels
cor_obs_synth = xr.combine_by_coords(
    [
        xr.corr(
            mean_diff_obs[var],
            mean_diff_synth[var],
        )
        for var in mean_diff_obs.data_vars
    ]
)

# make sure order matches get_delta_fig!
var_list = ["HWF", "HWD", "sumHeat"]

fig_delta_synth = hv.Layout(
    [
        (
            fig_delta_synth_init[i]
            * hv.Text(
                -180 + 35,
                -60 + 10,
                f"r = {str(cor_obs_synth[f't2m_x.t2m_x_threshold.{var_list[i]}'].values.round(2))}",
                fontsize=30,
            )
        ).opts(fontscale=2.5, ylim=(-60, None))
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
                -180 + 50,
                -60 + 10,
                f"MAE = {str(mean_abs_diff[f't2m_x.t2m_x_threshold.{var_list[i]}'].values.round(2))}",
                fontsize=30,
            )
        ).opts(fontscale=2.5, ylim=(-60, None))
        for i in range(len(var_list))
    ]
)


# manually update the labels on this one
fig_obs_minus_synth[0].opts(title=f"obs - synth ({suffix})")
fig_obs_minus_synth[1].opts(title=f"obs - synth ({suffix})")
fig_obs_minus_synth[2].opts(title=f"obs - synth ({suffix})")
# fig_obs_minus_synth[3].opts(title=f"obs - synth ({suffix})", xlabel="a")


# stitch all together into a single figure -------------------------
fig1 = fig_delta_obs[0] + fig_delta_synth[0] + fig_obs_minus_synth[0]
for i in [1, 2]:
    fig1 += fig_delta_obs[i] + fig_delta_synth[i] + fig_obs_minus_synth[i]

fig1.cols(3).opts(shared_axes=False)

# update of all maps here.
fig1.map(lambda x: x.opts(xticks=0, yticks=0, xlabel=""), hv.Image).map(
    lambda x: x.opts(xlabel=""), hv.Text
)

# hvplot.save(fig1, f'fig_medianshift_{suffix}_ref{ref_years[0]}_{ref_years[1]}.png')
# hvplot.save(fig1, f'fig_medianshift_{suffix}_ref{ref_years[0]}_{ref_years[1]}.html')

# size_opts = dict(frame_width=700, frame_height=400)
# fig1.cols(3).opts(opts.Overlay(**size_opts))


# #############################
# # deep dives
# #############################

# time_1985 = cftime.DatetimeNoLeap(
#     1985, 1, 1, 0, 0, 0, 0, has_year_zero=True
# )  # need this for the tick mark

# # comparing the jja and doy versions, the following areas look different:
# # (lon,lat) = (6, 30) # ~sudan
# # (lon,lat) = (-50, -9) # ~ brazil


# # sudan ----------------------
# sudan_lon = 30
# sudan_lat = 6


# hw_sudan = hw_obs.sel(lon=sudan_lon, lat=sudan_lat, method="nearest")
# vars_of_interest = ["HWF", "HWD", "sumHeat"]
# hw_sudan_list = [
#     (
#         hw_sudan[f"t2m_x.t2m_x_threshold.{var}"].hvplot(
#             title=f"{var}, sudan (lon, lat) = ({sudan_lon}, {sudan_lat}) ({suffix})"
#         )
#         * hv.VLine(time_1985)
#     ).opts(opts.VLine(color="gray"))
#     for var in vars_of_interest
# ]
# fig_hw_sudan = hv.Layout(hw_sudan_list).cols(1)

# # brazil -----------------------------
# brazil_lon = -50
# brazil_lat = -9
# hw_brazil = hw_obs.sel(lon=brazil_lon, lat=brazil_lat, method="nearest")
# hw_brazil_list = [
#     (
#         hw_brazil[f"t2m_x.t2m_x_threshold.{var}"].hvplot(
#             title=f"{var}, brazil (lon, lat) = ({brazil_lon}, {brazil_lat}) ({suffix})"
#         )
#         * hv.VLine(time_1985)
#     ).opts(opts.VLine(color="gray"))
#     for var in vars_of_interest
# ]
# fig_hw_brazil = hv.Layout(hw_brazil_list).cols(1)


# fig_hw_casestudy = (
#     fig_hw_brazil[0]
#     + fig_hw_sudan[0]
#     + fig_hw_brazil[1]
#     + fig_hw_sudan[1]
#     + fig_hw_brazil[2]
#     + fig_hw_sudan[2]
#     + fig_hw_brazil[3]
#     + fig_hw_sudan[3]
# ).cols(2)
# # hvplot.save(fig_hw_casestudy, f"fig_hw_casestudy_{suffix}.html")
