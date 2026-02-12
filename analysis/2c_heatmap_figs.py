"""
generate heatmap figures, with deciles bins of
- climatology moments on the x axis
- change in heatwave metrics on the y axis

inputs:
- moments_ds.nc from 2a_get_moments

outputs:
- fig_qbins.png
"""

from changing_heat_extremes import flags
from changing_heat_extremes import plot_helpers as phelpers
import xarray as xr
import polars as pl
import polars.selectors as cs
import numpy as np
import holoviews as hv
import hvplot.xarray  # noqa: F401
import hvplot.polars  # noqa: F401
from pathlib import Path

hvplot.extension(phelpers.backend_hv)

data_dir = Path("processed_data")

fig_kwargs = dict(
    fig_inches=(phelpers.width_default, phelpers.height_wide),
    **phelpers.global_kwargs,
)

layout_kwargs = dict(sublabel_format="", tight=True, tight_padding=4)

###########################################3
# read in data from 2a_get_moments
###########################################

combined_ds = xr.open_dataset(data_dir / f"moments_ds_{flags.label}.nc")
combined_df = pl.from_pandas(
    combined_ds.to_dataframe(), include_index=True
).drop_nulls()


# combined_ds.plot.scatter(
#     x="t2m_x_mean_diff", y="t2m_x_skew", hue="t2m_x.t2m_x_threshold.HWF", s=10
# )


##########################################
# if the above has too many points
# let's combine points into decile bins
#########################################

# exclude 0 beacuse pl.qcut is left-closed by default
decile_list = np.linspace(0.1, 1, 10)
vars_to_bin = ["t2m_x_mean_diff", "t2m_x_var", "t2m_x_skew"]

q_df = combined_df.clone()
for var in vars_to_bin:
    qbins = (
        combined_df[var]
        .qcut(quantiles=decile_list, include_breaks=True)
        .to_frame()
        .unnest(var)
        # .select("breakpoint")
        .rename({"category": f"{var}_cat", "breakpoint": f"{var}_q"})
    )
    q_df = q_df.hstack(qbins)


### variance ----------------------------------------

hw_cols = [name for name in q_df.columns if "threshold" in name]
qvar_df = (
    q_df.group_by(["t2m_x_mean_diff_q", "t2m_x_var_q"])
    .agg(pl.col(hw_cols).median(), pl.len())
    .with_columns(cs.numeric().round(1))
)

# make the ordering match what hv.quadmesh is expecting
qvar_ds = (
    (qvar_df.to_pandas().set_index(["t2m_x_mean_diff_q", "t2m_x_var_q"]).to_xarray())
    .sortby(["t2m_x_mean_diff_q", "t2m_x_var_q"])
    .transpose("t2m_x_var_q", "t2m_x_mean_diff_q")
)

# get the boundaries (i.e. including the min)
mean_diff_qs = np.concatenate(
    ([q_df["t2m_x_mean_diff"].min()], qvar_ds["t2m_x_mean_diff_q"].values)
).round(1)
var_qs = np.concatenate(
    ([q_df["t2m_x_var"].min()], qvar_ds["t2m_x_var_q"].values)
).round(1)


# # mark which boxes have fewer than 100 gridcells
# small_n_var = qvar_df.filter(pl.col("len") < 100).select(
#     ["t2m_x_mean_diff_q", "t2m_x_var_q"]
# )

# # map the box edges to the center
# mean_diff_midpoints = mean_diff_qs[:-1] + np.diff(mean_diff_qs) / 2
# var_midpoints = var_qs[:-1] + np.diff(var_qs) / 2
# x_coords_sorted = np.sort(qvar_ds["t2m_x_mean_diff_q"].values)
# y_coords_sorted = np.sort(qvar_ds["t2m_x_var_q"].values)
# x_map = dict(zip(x_coords_sorted, mean_diff_midpoints))
# y_map = dict(zip(y_coords_sorted, var_midpoints))
# small_n_var = small_n_var.with_columns([
#     pl.col("t2m_x_mean_diff_q").replace(x_map).alias("x_center"),
#     pl.col("t2m_x_var_q").replace(y_map).alias("y_center")
# ])

# markers_var = hv.Points(small_n_var, kdims=["x_center", "y_center"]).opts(
#         color="black",
#         marker="x",
#         s=10,
#     )

# make the plot
# hwf_q95 =qvar_ds["t2m_x.t2m_x_threshold.HWF"].quantile(0.95).values
cbar_hwf = phelpers.cbar_helper_hv(1, 20, num_bins=11, cmap="YlOrRd")
fig_var_hwf = hv.QuadMesh(
    (mean_diff_qs, var_qs, qvar_ds["t2m_x.t2m_x_threshold.HWF"])
).opts(
    edgecolors="white",
    xticks=mean_diff_qs[1::2],
    yticks=var_qs[::2],
    colorbar=True,
    xlim=(
        mean_diff_qs[1] - 0.3,
        mean_diff_qs[-2] + 0.3,
    ),  # cut off the huge first and last bin (-0.8 -> 3.8)
    title="(a) Change in HWF",
    xlabel="",
    ylabel="Climatological Variance (C²)",
    clabel="Days",
    hooks=[cbar_hwf],
    cbar_extend="both",
    **fig_kwargs,
)

# duration
hwd_q95 = qvar_ds["t2m_x.t2m_x_threshold.HWD"].quantile(0.95).values
cbar_hwd = phelpers.cbar_helper_hv(0, 7, cmap="YlOrRd")
fig_var_hwd = hv.QuadMesh(
    (mean_diff_qs, var_qs, qvar_ds["t2m_x.t2m_x_threshold.HWD"])
).opts(
    edgecolors="white",
    xticks=mean_diff_qs[1::2],
    yticks=0,
    colorbar=True,
    xlim=(
        mean_diff_qs[1] - 0.3,
        mean_diff_qs[-2] + 0.3,
    ),  # cut off the huge first and last bin (-0.8 -> 3.8)
    title="(b) Change in HWD",
    xlabel="",
    ylabel="",
    clabel="Days",
    hooks=[cbar_hwd],
    cbar_extend="both",
    **fig_kwargs,
)

# sumheat
hwf_q95 = qvar_ds["t2m_x.t2m_x_threshold.sumHeat"].quantile(0.95).values
cbar_sumheat = phelpers.cbar_helper_hv(1, 35, cmap="YlOrRd")
fig_var_sumheat = hv.QuadMesh(
    (mean_diff_qs, var_qs, qvar_ds["t2m_x.t2m_x_threshold.sumHeat"])
).opts(
    edgecolors="white",
    xticks=mean_diff_qs[1::2],
    yticks=0,
    colorbar=True,
    xlim=(
        mean_diff_qs[1] - 0.3,
        mean_diff_qs[-2] + 0.3,
    ),  # cut off the huge first and last bin (-0.8 -> 3.8)
    title="(b) Change in sumHeat",
    xlabel="",
    ylabel="",
    clabel="C",
    hooks=[cbar_sumheat],
    cbar_extend="both",
    **fig_kwargs,
)


figlist_var_qbins = [fig_var_hwf, fig_var_hwd, fig_var_sumheat]
fig_layout_var_qbins = hv.Layout(figlist_var_qbins).cols(3).opts(**layout_kwargs)
# hvplot.save(fig_layout_var_qbins, "fig_qbins_var.svg")

### skewness ----------------------------------------
qskew_df = (
    q_df.group_by(["t2m_x_mean_diff_q", "t2m_x_skew_q"])
    .agg(pl.col(hw_cols).median(), pl.len())
    .with_columns(cs.numeric().round(2))
)

# make the ordering match what hv.quadmesh is expecting
qskew_ds = (
    (qskew_df.to_pandas().set_index(["t2m_x_mean_diff_q", "t2m_x_skew_q"]).to_xarray())
    .sortby(["t2m_x_mean_diff_q", "t2m_x_skew_q"])
    .transpose("t2m_x_skew_q", "t2m_x_mean_diff_q")
)

# get the boundaries (i.e. including the min)
mean_diff_qs = np.concatenate(
    ([q_df["t2m_x_mean_diff"].min()], qskew_ds["t2m_x_mean_diff_q"].values)
).round(1)
skew_qs = np.concatenate(
    ([q_df["t2m_x_skew"].min()], qskew_ds["t2m_x_skew_q"].values)
).round(2)

fig_skew_hwf = hv.QuadMesh(
    (mean_diff_qs, skew_qs, qskew_ds["t2m_x.t2m_x_threshold.HWF"])
).opts(
    edgecolors="white",
    xticks=mean_diff_qs[1::2],
    yticks=skew_qs[1::2],
    colorbar=True,
    xlim=(
        mean_diff_qs[1] - 0.3,
        mean_diff_qs[-2] + 0.3,
    ),  # cut off the huge first and last bin (-0.8 -> 3.8)
    ylim=(
        skew_qs[1] - 0.3,
        skew_qs[-2] + 0.3,
    ),  # cut off the huge first and last bin (-1.57 -> 1.35)
    title="(d) Change in HWF",
    xlabel="Change in Tx (C)",
    ylabel="Climatological Skew",
    clabel="Days",
    hooks=[cbar_hwf],
    cbar_extend="both",
    **fig_kwargs,
)

# duration
fig_skew_hwd = hv.QuadMesh(
    (mean_diff_qs, skew_qs, qskew_ds["t2m_x.t2m_x_threshold.HWD"])
).opts(
    edgecolors="white",
    xticks=mean_diff_qs[1::2],
    yticks=0,
    colorbar=True,
    xlim=(
        mean_diff_qs[1] - 0.3,
        mean_diff_qs[-2] + 0.3,
    ),  # cut off the huge first and last bin (-0.8 -> 3.8)
    ylim=(
        skew_qs[1] - 0.3,
        skew_qs[-2] + 0.3,
    ),  # cut off the huge first and last bin (-1.57 -> 1.35)
    title="(e) Change in HWD",
    xlabel="Change in Tx (C)",
    ylabel="",
    clabel="Days",
    hooks=[cbar_hwd],
    cbar_extend="both",
    **fig_kwargs,
)

# sumheat
fig_skew_sumheat = hv.QuadMesh(
    (mean_diff_qs, skew_qs, qskew_ds["t2m_x.t2m_x_threshold.sumHeat"])
).opts(
    edgecolors="white",
    xticks=mean_diff_qs[1::2],
    yticks=0,
    colorbar=True,
    xlim=(
        mean_diff_qs[1] - 0.3,
        mean_diff_qs[-2] + 0.3,
    ),  # cut off the huge first and last bin (-0.8 -> 3.8)
    ylim=(
        skew_qs[1] - 0.3,
        skew_qs[-2] + 0.3,
    ),  # cut off the huge first and last bin (-1.57 -> 1.35)
    title="(f) Change in sumHeat",
    xlabel="Change in Tx (C)",
    ylabel="",
    clabel="C",
    hooks=[cbar_sumheat],
    cbar_extend="both",
    **fig_kwargs,
)


figlist_skew_qbins = [fig_skew_hwf, fig_skew_hwd, fig_skew_sumheat]
fig_layout_skew_qbins = hv.Layout(figlist_skew_qbins).cols(3).opts(**layout_kwargs)

##########################3
# fig_qbins ----
###########################3

fig_qbins_final = (
    hv.Layout(figlist_var_qbins + figlist_skew_qbins).cols(3).opts(**layout_kwargs)
)
# hvplot.save(fig_qbins_final, phelpers.fig_dir / f"fig_qbins_{flags.label}.png")


###################################
# supplemental analyses used in the paper
####################################

# Figure of gridcell counts -----------------
cbar_count = phelpers.cbar_helper_hv(0, 200, cmap="YlOrRd")


fig_var_count = hv.QuadMesh((mean_diff_qs, var_qs, qvar_ds["len"])).opts(
    edgecolors="white",
    xticks=mean_diff_qs[1::2],
    yticks=var_qs[::2],
    colorbar=False,
    xlim=(
        mean_diff_qs[1] - 0.3,
        mean_diff_qs[-2] + 0.3,
    ),  # cut off the huge first and last bin (-0.8 -> 3.8)
    title="Variance",
    xlabel="Change in Tx (C)",
    ylabel="Climatological Variance (C²)",
    hooks=[cbar_count],
    cbar_extend="both",
    **fig_kwargs,
)

fig_skew_count = hv.QuadMesh((mean_diff_qs, skew_qs, qskew_ds["len"])).opts(
    edgecolors="white",
    xticks=mean_diff_qs[1::2],
    yticks=skew_qs[1::2],
    colorbar=True,
    xlim=(
        mean_diff_qs[1] - 0.3,
        mean_diff_qs[-2] + 0.3,
    ),  # cut off the huge first and last bin (-0.8 -> 3.8)
    ylim=(
        skew_qs[1] - 0.3,
        skew_qs[-2] + 0.3,
    ),  # cut off the huge first and last bin (-1.57 -> 1.35)
    title="Skewness",
    xlabel="Change in Tx (C)",
    ylabel="Climatological Skew",
    clabel="Number of Gridcells",
    hooks=[cbar_count],
    cbar_extend="both",
    **fig_kwargs,
)

fig_counts = (fig_var_count + fig_skew_count).opts(**layout_kwargs)

# hvplot.save(fig_counts, phelpers.fig_dir / "supplemental" / "fig_qbin_counts.svg")
