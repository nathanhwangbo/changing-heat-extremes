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
use_calendar_summer = False  # if true, use JJA as summer. else use dayofyear mask
if use_calendar_summer:
    suffix = "jja"  # used for labeling plots
    hw_obs = xr.open_dataset("era_hw_metrics_1950_2021_anom.nc").sel(
        percentile=0.9, definition="3-0-0"
    )
    hw_synth = xr.open_dataset("era_hw_metrics_1950_2021_synth_anom.nc").sel(
        percentile=0.9, definition="3-0-0"
    )
else:
    suffix = "doy"  # used for labeling plots
    hw_obs = xr.open_dataset("era_hw_metrics_1950_2021_anom_doy.nc").sel(
        percentile=0.9, definition="3-0-0"
    )
    hw_synth = xr.open_dataset("era_hw_metrics_1950_2021_synth_anom_doy.nc").sel(
        percentile=0.9, definition="3-0-0"
    )

################################
# reproduce russo et al Figure 2
################################

# russo et al uses something close to hot_r for cmap
reds_discrete = tastymap.cook_tmap("cet_CET_L18", num_colors=10).cmap
rdbu_discrete = tastymap.cook_tmap("RdYlBu_r", num_colors=12).cmap

##########################################################

# instead of looking at trends over time, look at simple change between the two periods
# mean(1986-2021) - mean(1950-1985)

###########################################################


def get_delta_fig(mean_diff_ds, label_source, label_summer):
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
        cmap=reds_discrete,
        clim=(0, 15),
        title=f"delta in heatwave frequency ({label_summer})\nmean({label_source} 1986:2021) - mean(obs 1950:1985)",
        clabel="days",
        xlabel="",
        ylabel="",
    ).opts(fontscale=2.5, ylim=(-60, None))

    # delta in hwd
    hwd_delta = mean_diff_ds["t2m_x.t2m_x_threshold.HWD"]
    deltamap_hwd = hwd_delta.hvplot(
        projection=ccrs.PlateCarree(),
        coastline=True,
        cmap=rdbu_discrete,
        clim=(-6, 6),
        title=f"delta in heatwave duration ({label_summer})\nmean({label_source} 1986:2021) - mean(obs 1950:1985)",
        clabel="days",
        xlabel="",
        ylabel="",
    ).opts(fontscale=2.5, ylim=(-60, None))

    # delta in avi
    avi_delta = mean_diff_ds["t2m_x.t2m_x_threshold.AVI"]
    deltamap_avi = avi_delta.hvplot(
        projection=ccrs.PlateCarree(),
        coastline=True,
        cmap=rdbu_discrete,
        clim=(-1, 1),
        title=f"delta in average intensity ({label_summer})\nmean({label_source} 1986:2021) - mean(obs 1950:1985)",
        clabel="degC anomaly",
        xlabel="",
        ylabel="",
    ).opts(fontscale=2.5, ylim=(-60, None))

    # delta in heatsum
    heatsum_delta = mean_diff_ds["t2m_x.t2m_x_threshold.sumHeat"]
    deltamap_heatsum = heatsum_delta.hvplot(
        projection=ccrs.PlateCarree(),
        coastline=True,
        cmap=rdbu_discrete,
        clim=(-50.5, 50.5),
        title=f"delta in cumulative heat ({label_summer})\nmean({label_source} 1986:2021) - mean(obs 1950:1985)",
        clabel="degC anomaly",
        xlabel="",
        ylabel="",
    ).opts(fontscale=2.5, ylim=(-60, None))

    # combine
    fig_delta = (deltamap_hwf + deltamap_hwd + deltamap_avi + deltamap_heatsum).cols(1)

    return fig_delta


# ERA observed --------------------------------------
hw_old_obs = hw_obs.sel(time=slice("1950", "1985"))
hw_new_obs = hw_obs.sel(time=slice("1986", "2021"))
mean_diff_obs = hw_new_obs.mean(dim="time") - hw_old_obs.mean(dim="time")
fig_delta_obs = get_delta_fig(mean_diff_obs, label_source="obs", label_summer=suffix)
# hvplot.save(fig_delta_obs, f"fig_delta_obs_anom_{suffix}.html")


# ERA synthetic second half -------------------------------------
hw_old_synth = hw_synth.sel(time=slice("1950", "1985"))
hw_new_synth = hw_synth.sel(time=slice("1986", "2021"))
mean_diff_synth = hw_new_synth.mean(dim="time") - hw_old_synth.mean(dim="time")
fig_delta_synth_init = get_delta_fig(
    mean_diff_synth, label_source="synth", label_summer=suffix
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
var_list = ["HWF", "HWD", "AVI", "sumHeat"]

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
# hvplot.save(fig_delta_synth, f"fig_delta_synth_anom_{suffix}.html")

# difference between observed and synthetic -------------------
obs_minus_synth = mean_diff_obs - mean_diff_synth
fig_obs_minus_synth = get_delta_fig(
    obs_minus_synth, label_source="obs - synth", label_summer=suffix
)
# manually update the labels on this one
fig_obs_minus_synth[0].opts(title=f"obs - synth ({suffix})", xlabel="a")
fig_obs_minus_synth[1].opts(title=f"obs - synth ({suffix})", xlabel="a")
fig_obs_minus_synth[2].opts(title=f"obs - synth ({suffix})", xlabel="a")
fig_obs_minus_synth[3].opts(title=f"obs - synth ({suffix})", xlabel="a")


# stitch all together into a single figure -------------------------
fig1 = fig_delta_obs[0] + fig_delta_synth[0] + fig_obs_minus_synth[0]
for i in [1, 2, 3]:
    fig1 += fig_delta_obs[i] + fig_delta_synth[i] + fig_obs_minus_synth[i]

fig1.cols(3)

size_opts = dict(frame_width=700, frame_height=400)
# fig1.cols(3).opts(opts.Overlay(**size_opts))

# hvplot.save(fig1, f'fig_medianshift_{suffix}.html')

# import panel as pn
# layout = pn.Row(
#     pn.panel(fig_delta_obs[0], align='end'),
#     pn.panel(fig_delta_synth[0], align='end'),
#     pn.panel(fig_obs_minus_synth[0], align='end'),
#     sizing_mode="fixed", width=400, height=400, styles={"border": "1px solid black"}
# ).servable()
# # hvplot.save(layout, 'tmp.html')


#############################
# deep dives
#############################

time_1985 = cftime.DatetimeNoLeap(
    1985, 1, 1, 0, 0, 0, 0, has_year_zero=True
)  # need this for the tick mark

# comparing the jja and doy versions, the following areas look different:
# (lon,lat) = (6, 30) # ~sudan
# (lon,lat) = (-50, -9) # ~ brazil


# sudan ----------------------
sudan_lon = 30
sudan_lat = 6


hw_sudan = hw_obs.sel(lon=sudan_lon, lat=sudan_lat, method="nearest")
vars_of_interest = ["HWF", "HWD", "AVI", "sumHeat"]
hw_sudan_list = [
    (
        hw_sudan[f"t2m_x.t2m_x_threshold.{var}"].hvplot(
            title=f"{var}, sudan (lon, lat) = ({sudan_lon}, {sudan_lat}) ({suffix})"
        )
        * hv.VLine(time_1985)
    ).opts(opts.VLine(color="gray"))
    for var in vars_of_interest
]
fig_hw_sudan = hv.Layout(hw_sudan_list).cols(1)

# brazil -----------------------------
brazil_lon = -50
brazil_lat = -9
hw_brazil = hw_obs.sel(lon=brazil_lon, lat=brazil_lat, method="nearest")
vars_of_interest = ["HWF", "HWD", "AVI", "sumHeat"]
hw_brazil_list = [
    (
        hw_brazil[f"t2m_x.t2m_x_threshold.{var}"].hvplot(
            title=f"{var}, brazil (lon, lat) = ({brazil_lon}, {brazil_lat}) ({suffix})"
        )
        * hv.VLine(time_1985)
    ).opts(opts.VLine(color="gray"))
    for var in vars_of_interest
]
fig_hw_brazil = hv.Layout(hw_brazil_list).cols(1)


fig_hw_casestudy = (
    fig_hw_brazil[0]
    + fig_hw_sudan[0]
    + fig_hw_brazil[1]
    + fig_hw_sudan[1]
    + fig_hw_brazil[2]
    + fig_hw_sudan[2]
    + fig_hw_brazil[3]
    + fig_hw_sudan[3]
).cols(2)
# hvplot.save(fig_hw_casestudy, f"fig_hw_casestudy_{suffix}.html")
