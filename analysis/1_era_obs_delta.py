# analyzing the output of 0_era_meanshift.py

from changing_heat_extremes import flags
from changing_heat_extremes import plot_helpers as phelpers
# from changing_heat_extremes import analysis_helpers as ahelpers

import numpy as np
import xarray as xr
import cartopy.crs as ccrs
import hvplot.xarray  # noqa: F401
import holoviews as hv
from pathlib import Path

hvplot.extension(phelpers.backend_hv)

fig_dir = Path("figures")
data_dir = Path("processed_data")


fig_kwargs = dict(
    xlabel="",
    ylabel="",
    fig_inches=(phelpers.width_default, phelpers.height_wide),
    xaxis=None,
    yaxis=None,
    **phelpers.global_kwargs,
)

layout_kwargs = dict(sublabel_format="", tight=True, tight_padding=8)


hw_obs = xr.open_dataset(
    data_dir
    / f"hw_metrics_{flags.ref_years[0]}_{flags.new_years[1]}_anom{flags.label}.nc"
).sel(
    percentile=flags.percentile_threshold,
    definition="-".join(map(str, flags.hw_def[0])),
)
hw_synth = xr.open_dataset(
    data_dir
    / f"hw_metrics_{flags.ref_years[0]}_{flags.new_years[1]}_synth_anom{flags.label}.nc"
).sel(
    percentile=flags.percentile_threshold,
    definition="-".join(map(str, flags.hw_def[0])),
)


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
):
    """
    Creates figures looking at changes in difference in mean heatwave characteristics over two time period
    mean_diff_ds is an xarray dataset, the output of hdp.metric.compute_group_metrics, manipulated to represent the difference in metrics across two time periods
    label_source is a string (intended either "synthetic" or "observed") describing the source of the more recent time period used to calculate of mean_diff_ds
    label_summer is a string (intended either "jja" or "doy") describing how summer was defined in mean_diff_ds
    """
    hwf_delta = mean_diff_ds["t2m_x.t2m_x_threshold.HWF"]
    cmap_hwf = phelpers.cbar_helper_hv(0, 15, cmap=phelpers.cmap_red)
    deltamap_hwf = hwf_delta.hvplot(
        projection=ccrs.PlateCarree(),
        coastline=True,
        # title=f"delta in heatwave frequency ({label_summer})\nmean({label_source} {flags.new_years[0]}:{flags.new_years[1]}) - mean(obs {flags.ref_years[0]}:{flags.ref_years[1]})",
        title="Change in Frequency",
        clabel="days",
        xlabel="",
        ylabel="",
    ).opts(ylim=(-59, None), hooks=[cmap_hwf])

    # delta in hwd
    hwd_delta = mean_diff_ds["t2m_x.t2m_x_threshold.HWD"]
    cmap_hwd = phelpers.cbar_helper_hv(-6, 6, cmap=phelpers.cmap_rdbu)

    deltamap_hwd = hwd_delta.hvplot(
        projection=ccrs.PlateCarree(),
        coastline=True,
        # title=f"delta in heatwave duration ({label_summer})\nmean({label_source} {flags.new_years[0]}:{flags.new_years[1]}) - mean(obs {flags.ref_years[0]}:{flags.ref_years[1]})",
        title="Change in Duration",
        clabel="days",
        xlabel="",
        ylabel="",
    ).opts(ylim=(-59, None), hooks=[cmap_hwd])

    # delta in heatsum
    heatsum_delta = mean_diff_ds["t2m_x.t2m_x_threshold.sumHeat"]
    cmap_heatsum = phelpers.cbar_helper_hv(0, 25, cmap=phelpers.cmap_red)

    deltamap_heatsum = heatsum_delta.hvplot(
        projection=ccrs.PlateCarree(),
        coastline=True,
        # title=f"delta in cumulative heat ({label_summer})\nmean({label_source} {flags.new_years[0]}:{flags.new_years[1]}) - mean(obs {flags.ref_years[0]}:{flags.ref_years[1]})",
        title="Change in Cumulative Heat",
        clabel="°C-days",
        xlabel="",
        ylabel="",
    ).opts(ylim=(-59, None), hooks=[cmap_heatsum])

    # combine
    figlist_delta = [deltamap_hwf, deltamap_hwd, deltamap_heatsum]

    return figlist_delta


# ERA observed --------------------------------------
hw_old_obs = hw_obs.sel(time=slice(str(flags.ref_years[0]), str(flags.ref_years[1])))
hw_new_obs = hw_obs.sel(time=slice(str(flags.new_years[0]), str(flags.new_years[1])))
mean_diff_obs = hw_new_obs.mean(dim="time") - hw_old_obs.mean(dim="time")
figlist_delta_obs = get_delta_fig(
    mean_diff_obs,
    label_source="obs",
    label_summer=flags.label,
    ref_years=flags.ref_years,
    new_years=flags.new_years,
    # cmap_hwf=reds_discrete_odd,
)


# # manually fix the tickers on hwf
# hwf_ticks = np.linspace(0, 15, 11)[::2]
# # manually fix the tickers on hwf
# fig_delta_obs[0].map(
#     lambda x: x.opts(colorbar_opts={"ticker": bokeh.models.FixedTicker(ticks=hwf_ticks)}),
#     hv.Image,
# )

# make sure order matches get_delta_fig!
var_list = ["HWF", "HWD", "sumHeat"]  # , "MAX"]

# add in some text
fig_delta_obs = hv.Layout(
    [
        (
            figlist_delta_obs[i]
            * hv.Text(
                -180 + 220,
                -60 + 11,
                f"Global Mean={str(mean_diff_obs[f't2m_x.t2m_x_threshold.{var_list[i]}'].mean().values.round(2))}",
                fontsize=phelpers.label_size - 2,
            ).opts(ylim=(-59, 80), xlim=(-180, 180), **fig_kwargs)
        )
        for i in range(len(var_list))
    ]
).opts(**layout_kwargs)

# hvplot.save(fig_delta_obs, f"fig_delta_obs_anom_{flags.label}_ref{flags.ref_years[0]}_{flags.ref_years[1]}.html")


# ERA synthetic second half -------------------------------------
hw_old_synth = hw_synth.sel(
    time=slice(str(flags.ref_years[0]), str(flags.ref_years[1]))
)
hw_new_synth = hw_synth.sel(
    time=slice(str(flags.new_years[0]), str(flags.new_years[1]))
)
mean_diff_synth = hw_new_synth.mean(dim="time") - hw_old_synth.mean(dim="time")
fig_delta_synth_init = get_delta_fig(
    mean_diff_synth,
    label_source="synth",
    label_summer=flags.label,
    ref_years=flags.ref_years,
    new_years=flags.new_years,
    # cmap_hwf=reds_discrete_odd,
)
# # manually fix the tickers on hwf
# fig_delta_synth_init[0].map(
#     lambda x: x.opts(
#         colorbar_opts={
#             "ticker": bokeh.models.FixedTicker(ticks=hwf_ticks),
#         }
#     ),
#     hv.Image,
# )


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
                -60 + 11,
                f"r={str(cor_obs_synth[f't2m_x.t2m_x_threshold.{var_list[i]}'].values.round(2))}",
                fontsize=phelpers.label_size - 2,
            )
            * hv.Text(
                -180 + 220,
                -60 + 11,
                f"Global Mean={str(mean_diff_synth[f't2m_x.t2m_x_threshold.{var_list[i]}'].mean().values.round(2))}",
                fontsize=phelpers.label_size - 2,
            )
        ).opts(ylim=(-59, None))
        for i in range(len(var_list))
    ]
)
# hvplot.save(fig_delta_synth, f"fig_delta_synth_anom_{flags.label}_ref{flags.ref_years[0]}_{flags.ref_years[1]}.html")

# difference between observed and synthetic -------------------
obs_minus_synth = mean_diff_obs - mean_diff_synth
fig_obs_minus_synth_init = get_delta_fig(
    obs_minus_synth,
    label_source="obs - synth",
    label_summer=flags.label,
    ref_years=flags.ref_years,
    new_years=flags.new_years,
    clim_hwf=(-10, 10),
    clim_hwd=(-3, 3),
    clim_heatsum=(-30, 30),
    clim_max=(-2, 2),
    cmap_hwf=phelpers.rdbu_discrete,
    cmap_hwd=phelpers.rdbu_discrete,
    cmap_heatsum=phelpers.rdbu_discrete,
)

# add in mean absolute error over the map
mean_abs_diff = abs(obs_minus_synth).mean()

fig_obs_minus_synth = hv.Layout(
    [
        (
            fig_obs_minus_synth_init[i]
            * hv.Text(
                -180 + 52,
                -60 + 11,
                f"MAE={str(mean_abs_diff[f't2m_x.t2m_x_threshold.{var_list[i]}'].values.round(2))}",
                fontsize=phelpers.label_size - 2,
            )
        ).opts(ylim=(-59, None), xlim=(-180, 180))
        for i in range(len(var_list))
    ]
)


# manually update the labels on this one
fig_obs_minus_synth[0].opts(title="obs - synth")
fig_obs_minus_synth[1].opts(title="obs - synth")
fig_obs_minus_synth[2].opts(title="obs - synth")
# fig_obs_minus_synth[3].opts(title=f"obs - synth ({flags.label})")


# stitch all together into a single figure -------------------------
fig1 = fig_delta_obs[0] + fig_delta_synth[0] + fig_obs_minus_synth[0]
for i in np.arange(1, len(fig_delta_obs)).tolist():
    fig1 += fig_delta_obs[i] + fig_delta_synth[i] + fig_obs_minus_synth[i]


# iterate over the subplots and add the label to the title. --

# # weird ordering bc I want to go vertical instead of horizontal
# letter_ordering = ["a", "d", "g", "b", "e", "h", "c", "f", "i"]
# fig1_updated = phelpers.add_subplot_labels(fig1, labels=letter_ordering)
fig1_updated = fig1.cols(3).opts(shared_axes=False)

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

# fig1_final = fig1_updated.map(
#     lambda x: x.options(frame_height=phelpers.fheight_wide, **phelpers.global_kwargs),
#     [hv.Image, hv.Text],
# )

# hvplot.save(fig1_final, fig_dir / f"fig_meanshift_{flags.label}_ref{flags.ref_years[0]}_{flags.ref_years[1]}.png")
