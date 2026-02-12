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
import polars as pl
import polars.selectors as cs
import numpy as np
import holoviews as hv
import statsmodels.api as sm
import hvplot.xarray  # noqa: F401
import hvplot.polars  # noqa: F401
from pathlib import Path

hvplot.extension(phelpers.backend_hv)


fig_dir = Path("figures")
data_dir = Path("processed_data")

fig_kwargs = dict(
    fig_inches=(phelpers.width_default, phelpers.height_wide),
    **phelpers.global_kwargs,
)

layout_kwargs = dict(sublabel_format="", tight=True, tight_padding=4)


def fig_scatter(df, bin_var, bin_id_var, hw_metric, label, color):

    binned_df = (
        df.sort(bin_var)
        .group_by(bin_id_var)
        .agg(
            [
                pl.col(f"t2m_x.t2m_x_threshold.{hw_metric}")
                .mean()
                .alias(f"{hw_metric}_mean"),
                pl.col(f"t2m_x.t2m_x_threshold.{hw_metric}")
                .std()
                .alias(f"{hw_metric}_std"),
            ]
        )
    )
    # scatter = df.hvplot.scatter(
    #     x=bin_var, y=f"t2m_x.t2m_x_threshold.{hw_metric}", alpha=0.05, c=color
    # )
    # means = binned_df.hvplot.scatter(
    #     x="bin", y=f"{hw_metric}_mean", size=100, color=color
    # )
    sds = hv.ErrorBars(
        binned_df.select(
            [bin_id_var, f"{hw_metric}_mean", f"{hw_metric}_std"]
        ).to_numpy(),
        label=label,
    ).opts(alpha=0.7, capsize=3, edgecolor=color)

    lines = (
        binned_df.sort(bin_id_var)
        .hvplot.line(
            x=bin_id_var, y=f"{hw_metric}_mean", color=color, marker=".", ms=10
        )
        .opts(alpha=0.7)
    )

    # fig = scatter * means * sds
    fig = lines * sds
    return fig


###########################################3
# read in data from 2a_get_moments
###########################################

combined_ds = xr.open_dataset(data_dir / f"moments_ds_{flags.label}.nc")
combined_df = pl.from_pandas(
    combined_ds.to_dataframe(), include_index=True
).drop_nulls()

# 2deg --------------------------------


# variance --------------


deg2_obs_df = combined_df.filter(
    (pl.col("t2m_x_mean_diff") >= 1.75) & (pl.col("t2m_x_mean_diff") <= 2.25)
)

n_bins = 20
bins_var = np.linspace(
    deg2_obs_df["t2m_x_var"].min(), deg2_obs_df["t2m_x_var"].max(), n_bins
)
midpoints_var = ((bins_var[:-1] + bins_var[1:]) / 2).round(1).astype(str)

bins_skew = np.linspace(
    deg2_obs_df["t2m_x_skew"].min(), deg2_obs_df["t2m_x_skew"].max(), n_bins
)
midpoints_skew = ((bins_skew[:-1] + bins_skew[1:]) / 2).round(1).astype(str)


deg2_obs_df = deg2_obs_df.with_columns(
    var_bin_id=pl.col("t2m_x_var")
    .cut(breaks=bins_var[1:-1], labels=midpoints_var)
    .cast(pl.String)
    .cast(pl.Float64),
    skew_bin_id=pl.col("t2m_x_skew")
    .cut(breaks=bins_skew[1:-1], labels=midpoints_skew)
    .cast(pl.String)
    .cast(pl.Float64),
)

fig_var_hwf_obs = fig_scatter(
    deg2_obs_df, "t2m_x_var", "var_bin_id", "HWF", "Observed", "red"
).opts(hv.opts.Scatter(alpha=0.2))
fig_var_hwd_obs = fig_scatter(
    deg2_obs_df, "t2m_x_var", "var_bin_id", "HWD", "Observed", "red"
).opts(hv.opts.Scatter(alpha=0.2))
fig_var_sumheat_obs = fig_scatter(
    deg2_obs_df, "t2m_x_var", "var_bin_id", "sumHeat", "Observed", "red"
).opts(hv.opts.Scatter(alpha=0.2))


# skewness --------------
n_bins = 20
bin_size_skew = (
    deg2_obs_df["t2m_x_skew"].max() - deg2_obs_df["t2m_x_skew"].min()
) / n_bins

fig_skew_hwf_obs = fig_scatter(
    deg2_obs_df, "t2m_x_skew", "skew_bin_id", "HWF", "Observed", "red"
).opts(hv.opts.Scatter(alpha=0.05))
fig_skew_hwd_obs = fig_scatter(
    deg2_obs_df, "t2m_x_skew", "skew_bin_id", "HWD", "Observed", "red"
).opts(hv.opts.Scatter(alpha=0.05))
fig_skew_sumheat_obs = fig_scatter(
    deg2_obs_df, "t2m_x_skew", "skew_bin_id", "sumHeat", "Observed", "red"
).opts(hv.opts.Scatter(alpha=0.05))


# 2 degree, synthetic ---------------------------------------------------


hw_synth_2deg = (
    xr.open_dataset(
        data_dir
        / f"hw_metrics_{flags.ref_years[0]}_{flags.new_years[1]}_synth_2deg_anom{flags.label}.nc"
    )
    .sel(percentile=flags.percentile_threshold, definition="3-0-0")
    .drop_vars(["percentile", "definition"])
)

hw_old_2deg = hw_synth_2deg.sel(
    time=slice(str(flags.ref_years[0]), str(flags.ref_years[1]))
)
hw_new_2deg = hw_synth_2deg.sel(
    time=slice(str(flags.new_years[0]), str(flags.new_years[1]))
)
hw_mean_diff_2deg = hw_new_2deg.mean(dim="time") - hw_old_2deg.mean(dim="time")

# pull out just the climatology variables
climatology_stats = combined_ds[["t2m_x_skew", "t2m_x_kurt", "t2m_x_var", "t2m_x_ar1"]]
combined_synth_2deg_ds = xr.merge([climatology_stats, hw_mean_diff_2deg], join="exact")
# combined_synth_2deg_df = combined_synth_2deg_ds.to_dataframe().dropna(how="all")  # this just drops ocean gridcells
combined_synth_2deg_df = pl.from_pandas(
    combined_synth_2deg_ds.to_dataframe(), include_index=True
).drop_nulls()

combined_synth_2deg_df = combined_synth_2deg_df.with_columns(
    var_bin_id=pl.col("t2m_x_var")
    .cut(breaks=bins_var[1:-1], labels=midpoints_var)
    .cast(pl.String)
    .cast(pl.Float64),
    skew_bin_id=pl.col("t2m_x_skew")
    .cut(breaks=bins_skew[1:-1], labels=midpoints_skew)
    .cast(pl.String)
    .cast(pl.Float64),
)


# variance -----------------------
fig_var_hwf_synth = fig_scatter(
    combined_synth_2deg_df, "t2m_x_var", "var_bin_id", "HWF", "Synthetic", "blue"
).opts(hv.opts.Scatter(alpha=0.01))
fig_var_hwd_synth = fig_scatter(
    combined_synth_2deg_df, "t2m_x_var", "var_bin_id", "HWD", "Synthetic", "blue"
).opts(hv.opts.Scatter(alpha=0.01))
fig_var_sumheat_synth = fig_scatter(
    combined_synth_2deg_df, "t2m_x_var", "var_bin_id", "sumHeat", "Synthetic", "blue"
).opts(hv.opts.Scatter(alpha=0.01))

# skewness ----------------------
fig_skew_hwf_synth = fig_scatter(
    combined_synth_2deg_df, "t2m_x_skew", "skew_bin_id", "HWF", "Synthetic", "blue"
).opts(hv.opts.Scatter(alpha=0.01))
fig_skew_hwd_synth = fig_scatter(
    combined_synth_2deg_df, "t2m_x_skew", "skew_bin_id", "HWD", "Synthetic", "blue"
).opts(hv.opts.Scatter(alpha=0.01))
fig_skew_sumheat_synth = fig_scatter(
    combined_synth_2deg_df, "t2m_x_skew", "skew_bin_id", "sumHeat", "Synthetic", "blue"
).opts(hv.opts.Scatter(alpha=0.01))


###########
# combine
##########
fig_var_hwf = (fig_var_hwf_synth * fig_var_hwf_obs).opts(
    xlabel="Climatological Variance (C²)",
    ylabel="Change in HWF (days)",
    xlim=(0, 60),
    ylim=(0, 50),
    show_legend=False,
)
fig_var_hwd = (fig_var_hwd_synth * fig_var_hwd_obs).opts(
    xlabel="Climatological Variance (C²)",
    ylabel="Change in HWD (days)",
    xlim=(0, 60),
    ylim=(0, 20),
    show_legend=False,
)
fig_var_sumheat = (fig_var_sumheat_synth * fig_var_sumheat_obs).opts(
    xlabel="Climatological Variance (C²)",
    ylabel="Change in sumHeat (C-days)",
    xlim=(0, 60),
    ylim=(0, 75),
)

fig_skew_hwf = (fig_skew_hwf_synth * fig_skew_hwf_obs).opts(
    xlabel="Climatological Skew",
    ylabel="Change in HWF (days)",
    xlim=(-1, 0.75),
    ylim=(0, 50),
    show_legend=False,
)
fig_skew_hwd = (fig_skew_hwd_synth * fig_skew_hwd_obs).opts(
    xlabel="Climatological Skew",
    ylabel="Change in HWD (days)",
    xlim=(-1, 0.75),
    ylim=(0, 20),
    show_legend=False,
)
fig_skew_sumheat = (fig_skew_sumheat_synth * fig_skew_sumheat_obs).opts(
    xlabel="Climatological Skew",
    ylabel="Change in sumHeat (C-days)",
    xlim=(-1, 0.75),
    ylim=(0, 75),
    show_legend=False,
)

fig_scatter = (
    (
        fig_var_hwf
        + fig_var_hwd
        + fig_var_sumheat
        + fig_skew_hwf
        + fig_skew_hwd
        + fig_skew_sumheat
    )
    .cols(3)
    .opts(shared_axes=False, **layout_kwargs)
)
fig_scatter
# hvplot.save(fig_scatter, phelpers.fig_dir / "fig_2deg_new.png")
