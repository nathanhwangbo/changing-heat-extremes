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
import pandas as pd
import holoviews as hv
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

# combined_ds.plot.scatter(
#     x="t2m_x_mean_diff", y="t2m_x_skew", hue="t2m_x.t2m_x_threshold.HWF", s=10
# )


##########################################
# if the above has too many points
# let's combine points into quantile bins
#########################################

n_qbins = 10
combined_df["tmax_diff_qbins"] = pd.qcut(combined_df["t2m_x_mean_diff"], q=n_qbins, precision=1)
combined_df["skew_qbins"] = pd.qcut(combined_df["t2m_x_skew"], q=n_qbins, precision=1)
combined_df["var_qbins"] = pd.qcut(combined_df["t2m_x_var"], q=n_qbins, precision=1)
combined_df["ar1_qbins"] = pd.qcut(
    combined_df["t2m_x_ar1"],
    q=n_qbins,
    precision=1,
)

# fixing the first variance bin looking like this : (0.16999999999999998, 0.58]
var_qbin_cats = combined_df["var_qbins"].cat.categories.to_list()
new_first_interval_var = pd.Interval(round(var_qbin_cats[0].left, 2), round(var_qbin_cats[0].right, 2), closed="right")
var_qbin_cats[0] = new_first_interval_var
combined_df["var_qbins"] = combined_df["var_qbins"].cat.rename_categories(var_qbin_cats)


# fixing the first bin looking like this : (0.16999999999999998, 0.58]
ar1_qbin_cats = combined_df["ar1_qbins"].cat.categories.to_list()
new_first_interval = pd.Interval(round(ar1_qbin_cats[0].left, 2), round(ar1_qbin_cats[0].right, 2), closed="right")
ar1_qbin_cats[0] = new_first_interval
combined_df["ar1_qbins"] = combined_df["ar1_qbins"].cat.rename_categories(ar1_qbin_cats)
# ---------- end manual bin fix

# combined_df.reset_index()[['lon', 'lat']].value_counts() # there are 14728 land gridcells
combined_df = combined_df.reset_index().sort_values(by=["t2m_x_mean_diff", "t2m_x_skew"])


### variance ----------------------------------------

figlist_var_qbins = phelpers.get_heatmap(
    combined_df,
    y_name_var="var_qbins",
    y_name_label="variance",
    use_qbins=True,
    ref_years=flags.ref_years,
    new_years=flags.new_years,
)

fig_layout_var_qbins = hv.Layout(figlist_var_qbins[0:3]).cols(1)
# hvplot.save(fig_layout_var_qbins, "fig_qbins_var.html")

### skewness ----------------------------------------

figlist_skewness_qbins = phelpers.get_heatmap(
    combined_df,
    y_name_var="skew_qbins",
    y_name_label="skewness",
    use_qbins=True,
    ref_years=flags.ref_years,
    new_years=flags.new_years,
)
fig_layout_skewness_qbins = hv.Layout(figlist_skewness_qbins[0:3]).cols(1)

# mpl_render = hv.renderer('matplotlib')
# mpl_skewness_qbins = mpl_render.get_plot(fig_layout_skewness_qbins)

# hvplot.save(fig_layout_skewness_qbins, "fig_qbins_skew.html")

### ar1 ----------------------------------------
figlist_ar1_qbins = phelpers.get_heatmap(
    combined_df,
    y_name_var="ar1_qbins",
    y_name_label="AR(1)",
    use_qbins=True,
    ref_years=flags.ref_years,
    new_years=flags.new_years,
)
fig_layout_ar1_qbins = hv.Layout(figlist_ar1_qbins[0:3]).cols(1)
# hvplot.save(fig_layout_ar1_qbins, "fig_qbins_ar1.html")


##########################3
# fig_qbins ----
###########################3

# fig_qbins = hv.Layout(figlist_var_qbins[0:3] + figlist_skewness_qbins[0:3]).cols(2)

figlist_qbins = [
    figlist_var_qbins[0],
    figlist_skewness_qbins[0],
    figlist_var_qbins[1],
    figlist_skewness_qbins[1],
    figlist_var_qbins[2],
    figlist_skewness_qbins[2],
]


# weird ordering bc I want to go vertical instead of horizontal
# letter ordering = string.ascii_lowercase
letter_ordering = ["a", "d", "b", "e", "c", "f"]
figlist_qbins = phelpers.add_subplot_labels(figlist_qbins, letter_ordering)
fig_qbins = hv.Layout(figlist_qbins).cols(2)


fig_qbins_final = fig_qbins.map(
    lambda x: x.options(
        fontscale=1,
        fontsize={
            "title": title_size,
            "labels": label_size,
            "ticks": tick_size,
            "legend": tick_size,
        },
    )
).opts(hv.opts.HeatMap(frame_width=fwidth_qbins, frame_height=fheight_qbins))

# hvplot.save(fig_qbins_final, "figures\\fig_qbins.png")
