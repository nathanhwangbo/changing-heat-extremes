"""
generate scatterplots for a fixed 2 degree shift, with points of
- climatology moments on the x axis
- change in heatwave metrics on the y axis

inputs:
- moments_ds.nc from 2a_get_moments.py
- heatwave metrics for the synthetic 2degree shift experiment from 0_era_meanshift.py

outputs:
- fig_2deg.png
"""

from changing_heat_extremes import flags
from changing_heat_extremes import plot_helpers as phelpers
import xarray as xr
import numpy as np
import holoviews as hv
from holoviews import opts
import statsmodels.api as sm
import hvplot.xarray  # noqa: F401
import hvplot.pandas  # noqa: F401
from pathlib import Path

fig_dir = Path("figures")
data_dir = Path("processed_data")


scale = 1  # in this case,
title_size = 16 * scale
label_size = 14 * scale
tick_size = 10 * scale
fwidth = 400
fheight = 150

# have 400 * 150 = 60000 to play with
fwidth_qbins = 300
fheight_qbins = 200

###########################################3
# read in data from 2a_get_moments
###########################################

combined_ds = xr.open_dataset(data_dir / f"moments_ds_{flags.label}.nc")
combined_df = combined_ds.to_dataframe().dropna(how="all")

# 2deg --------------------------------

deg2_obs_df = combined_df.loc[(combined_df["t2m_x_mean_diff"] >= 1.75) & (combined_df["t2m_x_mean_diff"] <= 2.25)]

fig_skew_2deg_obs = phelpers.get_scatter(
    deg2_obs_df,
    x_var="t2m_x_skew",
    x_label="skewness",
    deg=2,
    alpha_pt=0.1,
    label_curve="observed",
    ylim_hwf=(0, 35),
    ylim_hwd=(0, 14),
    ylim_sumheat=(10, 50),
    ref_years=flags.ref_years,
    new_years=flags.new_years,
)
# hvplot.save(fig_skew_2deg_obs, 'fig_skew_2deg_obs.html')

fig_var_2deg_obs = phelpers.get_scatter(
    deg2_obs_df,
    x_var="t2m_x_var",
    x_label="variance",
    deg=2,
    size=15,
    alpha_pt=0.1,
    label_curve="observed",
    ylim_hwf=(0, 35),
    ylim_hwd=(0, 14),
    ylim_sumheat=(10, 50),
    ref_years=flags.ref_years,
    new_years=flags.new_years,
)
# hvplot.save(fig_var_2deg_obs, 'fig_var_2deg_obs.html')

fig_ar1_2deg_obs = phelpers.get_scatter(
    deg2_obs_df,
    x_var="t2m_x_ar1",
    x_label="AR(1)",
    deg=2,
    size=15,
    alpha_pt=0.1,
    label_curve="observed",
    ylim_hwf=(0, 35),
    ylim_hwd=(0, 14),
    ylim_sumheat=(10, 50),
    ref_years=flags.ref_years,
    new_years=flags.new_years,
)
# hvplot.save(fig_ar1_2deg_obs, 'fig_ar1_2deg_obs.html')


# 2 degree, synthetic ---------------------------------------------------


hw_synth_2deg = (
    xr.open_dataset(data_dir / f"hw_metrics_{flags.ref_years[0]}_{flags.new_years[1]}_synth_2deg_anom{flags.label}.nc")
    .sel(percentile=flags.percentile_threshold, definition="3-0-0")
    .drop_vars(["percentile", "definition"])
)

hw_old_2deg = hw_synth_2deg.sel(time=slice(str(flags.ref_years[0]), str(flags.ref_years[1])))
hw_new_2deg = hw_synth_2deg.sel(time=slice(str(flags.new_years[0]), str(flags.new_years[1])))
hw_mean_diff_2deg = hw_new_2deg.mean(dim="time") - hw_old_2deg.mean(dim="time")

# pull out just the climatology variables
climatology_stats = combined_ds[["t2m_x_skew", "t2m_x_kurt", "t2m_x_var", "t2m_x_ar1"]]
combined_synth_2deg_ds = xr.merge([climatology_stats, hw_mean_diff_2deg], join="exact")
combined_synth_2deg_df = combined_synth_2deg_ds.to_dataframe().dropna(how="all")  # this just drops ocean gridcells


fig_var_2deg_synth = phelpers.get_scatter(
    combined_synth_2deg_df,
    "t2m_x_var",
    "variance",
    deg=2,
    color_pt="blue",
    color_line="blue",
    alpha_pt=0.04,
    label_curve="synthetic",
    ylim_hwf=(0, 35),
    ylim_hwd=(0, 14),
    ylim_sumheat=(10, 50),
    ref_years=flags.ref_years,
    new_years=flags.new_years,
)
fig_var_2deg_synth.map(lambda x: x.opts(xlim=(-1, 70)), hv.Curve)

# temp ---------------------------------------------------
combined_synth_2deg_df["t2m_x_sd"] = np.sqrt(combined_synth_2deg_df["t2m_x_var"])
fig_sd_2deg_synth = phelpers.get_scatter(
    combined_synth_2deg_df,
    "t2m_x_sd",
    "sd",
    deg=2,
    color_pt="blue",
    color_line="blue",
    alpha_pt=0.04,
    label_curve="synthetic",
    ylim_hwf=(0, 35),
    ylim_hwd=(0, 14),
    ylim_sumheat=(10, 50),
    ref_years=flags.ref_years,
    new_years=flags.new_years,
)
# end temp ------------------------------


fig_skew_2deg_synth = phelpers.get_scatter(
    combined_synth_2deg_df,
    x_var="t2m_x_skew",
    x_label="skewness",
    deg=2,
    color_pt="blue",
    color_line="blue",
    alpha_pt=0.04,
    label_curve="synthetic",
    ylim_hwf=(0, 35),
    ylim_hwd=(0, 14),
    ylim_sumheat=(10, 50),
    ref_years=flags.ref_years,
    new_years=flags.new_years,
)
fig_skew_2deg_synth.map(lambda x: x.opts(xlim=(-1, 0.75)), hv.Curve)


fig_ar1_2deg_synth = phelpers.get_scatter(
    combined_synth_2deg_df,
    x_var="t2m_x_ar1",
    x_label="AR(1)",
    deg=2,
    color_pt="blue",
    color_line="blue",
    alpha_pt=0.04,
    label_curve="synthetic",
    ylim_hwf=(0, 35),
    ylim_hwd=(0, 14),
    ylim_sumheat=(10, 50),
    ref_years=flags.ref_years,
    new_years=flags.new_years,
)
fig_ar1_2deg_synth.map(lambda x: x.opts(xlim=(0.55, 0.9)), hv.Curve)


# combine figures
# see the original title in _obs for better descriptions

fig_var_hwf_2deg = (fig_var_2deg_synth[0] * fig_var_2deg_obs[0]).opts(
    legend_position="top_right", title="Change in HWF"
)
fig_var_hwd_2deg = (fig_var_2deg_synth[1] * fig_var_2deg_obs[1]).opts(
    legend_position="top_right", title="Change in HWD"
)
fig_var_heatsum_2deg = (fig_var_2deg_synth[2] * fig_var_2deg_obs[2]).opts(
    legend_position="top_right", title="Change in sumHeat"
)

fig_skew_hwf_2deg = (fig_skew_2deg_synth[0] * fig_skew_2deg_obs[0]).opts(
    legend_position="top_right", title="Change in HWF"
)
fig_skew_hwd_2deg = (fig_skew_2deg_synth[1] * fig_skew_2deg_obs[1]).opts(
    legend_position="top_right", title="Change in HWD"
)
fig_skew_heatsum_2deg = (fig_skew_2deg_synth[2] * fig_skew_2deg_obs[2]).opts(
    legend_position="top_right", title="Change in sumHeat"
)

fig_ar1_hwf_2deg = (fig_ar1_2deg_synth[0] * fig_ar1_2deg_obs[0]).opts(
    legend_position="top_right", title="Change in HWF"
)
fig_ar1_hwd_2deg = (fig_ar1_2deg_synth[1] * fig_ar1_2deg_obs[1]).opts(
    legend_position="top_right", title="Change in HWD"
)
fig_ar1_heatsum_2deg = (fig_ar1_2deg_synth[2] * fig_ar1_2deg_obs[2]).opts(
    legend_position="top_right", title="Change in sumHeat"
)

fig_2deg = (
    fig_var_hwf_2deg
    + fig_skew_hwf_2deg
    + fig_var_hwd_2deg
    + fig_skew_hwd_2deg
    + fig_var_heatsum_2deg
    + fig_skew_heatsum_2deg
).cols(2)

# weird ordering bc I want to go vertical instead of horizontal
letter_ordering = ["a", "d", "b", "e", "c", "f"]
updated_fig_2deg_list = phelpers.add_subplot_labels(fig_2deg, letter_ordering)
fig_2deg_updated = hv.Layout(updated_fig_2deg_list).cols(2)


######################
# fig_2deg ----
######################


fig_2deg_final = fig_2deg_updated.map(
    lambda x: x.options(
        fontsize={
            "title": title_size,
            "labels": label_size,
            "ticks": tick_size,
            "legend": tick_size,
        },
    )
).opts(
    opts.Curve(frame_width=fwidth_qbins, frame_height=fheight_qbins),
)
# hvplot.save(fig_2deg_final, fig_dir / 'fig_2deg_{flags.label}.png')

##########################################
## misc analyses referenced in the paper
##########################################

# computing linear regression coefficients for comparisons ----


skew_hwf_2deg_synth_fitted = sm.nonparametric.lowess(
    exog=combined_synth_2deg_df["t2m_x_skew"],
    endog=combined_synth_2deg_df["t2m_x.t2m_x_threshold.HWF"],
    frac=2 / 3,
)
# get gradients
skew_hwf_2deg_synth_grad = np.gradient(skew_hwf_2deg_synth_fitted[:, 1], skew_hwf_2deg_synth_fitted[:, 0])

# compare to linear fit
np.polyfit(
    combined_synth_2deg_df["t2m_x_skew"],
    combined_synth_2deg_df["t2m_x.t2m_x_threshold.HWF"],
    deg=1,
)
np.polyfit(deg2_obs_df["t2m_x_skew"], deg2_obs_df["t2m_x.t2m_x_threshold.HWF"], deg=1)


np.polyfit(
    combined_synth_2deg_df["t2m_x_skew"],
    combined_synth_2deg_df["t2m_x.t2m_x_threshold.HWD"],
    deg=1,
)
np.polyfit(deg2_obs_df["t2m_x_skew"], deg2_obs_df["t2m_x.t2m_x_threshold.HWD"], deg=1)

np.polyfit(
    combined_synth_2deg_df["t2m_x_skew"],
    combined_synth_2deg_df["t2m_x.t2m_x_threshold.sumHeat"],
    deg=1,
)
np.polyfit(deg2_obs_df["t2m_x_skew"], deg2_obs_df["t2m_x.t2m_x_threshold.sumHeat"], deg=1)

hv.Curve(zip(skew_hwf_2deg_synth_fitted[:, 0], skew_hwf_2deg_synth_grad))
hv.Curve(zip(skew_hwf_2deg_synth_fitted[:, 0], skew_hwf_2deg_synth_fitted[:, 1]))


np.polyfit(
    combined_synth_2deg_df["t2m_x_skew"],
    combined_synth_2deg_df["t2m_x.t2m_x_threshold.HWD"],
    deg=1,
)
np.polyfit(deg2_obs_df["t2m_x_skew"], deg2_obs_df["t2m_x.t2m_x_threshold.HWD"], deg=1)

#  ------------------------------
