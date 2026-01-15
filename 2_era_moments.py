import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import hvplot.xarray
import colorcet as cc
import matplotlib as mpl
import tastymap
import regionmask
import glob
from xarray_einstats import stats  # wrapper around apply_ufunc for moments
import pandas as pd
import holoviews as hv
import hvplot.pandas
from holoviews import opts
import statsmodels.api as sm
import matplotlib.colors as mcolors
import string
from functools import partial

xr.set_options(use_new_combine_kwarg_defaults=True)


def subplot_label_hook(plot, element, sub_label=""):
    """add subplot labels (a, b, c...)"""
    # Access the underlying Bokeh figure
    fig = plot.state

    original_title = fig.title.text
    fig.title.text = f"{sub_label} {original_title}"


scale = 1  # in this case,
title_size = 16 * scale
label_size = 14 * scale
tick_size = 10 * scale
fwidth = 400
fheight = 150

# have 400 * 150 = 60000 to play with
fwidth_qbins = 300
fheight_qbins = 200

# hvplot.extension("matplotlib")
# hvplot.extension("bokeh")

rdbu_discrete = tastymap.cook_tmap("RdYlBu_r", num_colors=12).cmap

# rdbu_hex = [mcolors.rgb2hex(rdbu_discrete(i)) for i in range(rdbu_discrete.N)]
# equivalent:
# rdbu_discrete = tastymap.cook_tmap("RdYlBu_r", num_colors=12)
# rdbu_hex = rdbu_discrete.to_model("hex")

reds_discrete = tastymap.cook_tmap("cet_CET_L18", num_colors=12)[
    1:11
].cmap  # get rid of white
# equiv:
# reds_discrete = tastymap.cook_tmap("cet_CET_L18", num_colors = 12)
# reds_discrete_no_white = tastymap.utils.subset_cmap(slice(1, 11)))
blues_discrete = tastymap.cook_tmap("blues", num_colors=10).cmap


def add_landmask(ds):
    # create a landmask
    land = regionmask.defined_regions.natural_earth_v5_0_0.land_110
    landmask = land.mask(ds)  # ocean is nan, land is 0
    is_land = landmask == 0

    # also get rid of greenland
    greenland = regionmask.defined_regions.natural_earth_v5_0_0.countries_110[
        ["Greenland"]
    ]
    gl_mask = greenland.mask(ds)
    is_not_greenland = gl_mask.isnull()

    # also get rid of antarctic
    is_not_antarctic = ds["lat"] > -60
    # is_not_arctic = ds["lat"] < 60

    # apply landmask
    ds = ds.where(is_land & is_not_greenland & is_not_antarctic)

    return ds


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
    hw_synth_1deg = (
        xr.open_dataset(
            f"era_hw_metrics_{ref_years[0]}_{new_years[1]}_synth_1deg_anom.nc"
        )
        .sel(percentile=0.9, definition="3-0-0")
        .drop_vars(["percentile", "definition"])
    )
    hw_synth_2deg = (
        xr.open_dataset(
            f"era_hw_metrics_{ref_years[0]}_{new_years[1]}_synth_2deg_anom.nc"
        )
        .sel(percentile=0.9, definition="3-0-0")
        .drop_vars(["percentile", "definition"])
    )
else:
    hw_all = (
        xr.open_dataset(f"era_hw_metrics_{ref_years[0]}_{new_years[1]}_anom_doy.nc")
        .sel(percentile=0.9, definition="3-0-0")
        .drop_vars(["percentile", "definition"])
    )
    hw_synth_1deg = (
        xr.open_dataset(
            f"era_hw_metrics_{ref_years[0]}_{new_years[1]}_synth_1deg_anom_doy.nc"
        )
        .sel(percentile=0.9, definition="3-0-0")
        .drop_vars(["percentile", "definition"])
    )
    hw_synth_2deg = (
        xr.open_dataset(
            f"era_hw_metrics_{ref_years[0]}_{new_years[1]}_synth_2deg_anom_doy.nc"
        )
        .sel(percentile=0.9, definition="3-0-0")
        .drop_vars(["percentile", "definition"])
    )

# compute deltas
hw_old = hw_all.sel(time=slice(str(ref_years[0]), str(ref_years[1])))
hw_new = hw_all.sel(time=slice(str(new_years[0]), str(new_years[1])))
hw_mean_diff = hw_new.mean(dim="time") - hw_old.mean(dim="time")


#######################################################################
# Calculate mean differences (new period) - (old period) for temperature
#######################################################################

# anomalies relative to ref_years, calculated in 0_era_medianshift.py
# TODO: should split up era_land_anom calculation into multiple files. is curently 10gb
era_anom_path = "era_land_anom.nc"
era_land_anom = xr.open_dataset(era_anom_path)

# compute deltas-------------------------------------------
era_land_old = era_land_anom.sel(time=slice(str(ref_years[0]), str(ref_years[1])))
era_land_new = era_land_anom.sel(time=slice(str(new_years[0]), str(new_years[1])))
tmax_mean_diff = (era_land_new.mean(dim="time") - era_land_old.mean(dim="time")).rename(
    {"t2m_x": "t2m_x_mean_diff"}
)

##############################################
# Calculate climatological (ref_years) moments
# NOTE! these are moments of the *doy anomalies* wrt to (ref_years), i.e. mean 0 over this period
##############################################

clim_skew = stats.skew(era_land_old["t2m_x"], dims=["time"]).rename("t2m_x_skew")
clim_kurt = stats.kurtosis(era_land_old["t2m_x"], dims=["time"]).rename("t2m_x_kurt")
clim_var = era_land_old["t2m_x"].var(dim="time").rename("t2m_x_var")
clim_ar1 = xr.corr(
    era_land_old["t2m_x"], era_land_old["t2m_x"].shift(time=1), dim="time"
).rename("t2m_x_ar1")

climatology_stats = xr.merge([clim_skew, clim_kurt, clim_var, clim_ar1])


##############################################
# combine maps into 1 xr.dataset
##############################################

combined_ds = xr.merge([tmax_mean_diff, climatology_stats, hw_mean_diff], join="exact")

combined_df = combined_ds.to_dataframe().dropna(how="all")

# combined_ds.plot.scatter(
#     x="t2m_x_mean_diff", y="t2m_x_skew", hue="t2m_x.t2m_x_threshold.HWF", s=10
# )


##########################################
# if the above has too many points
# let's combine points into quantile bins
#########################################

# hvplot.extension("bokeh")

n_qbins = 10
combined_df["tmax_diff_qbins"] = pd.qcut(
    combined_df["t2m_x_mean_diff"], q=n_qbins, precision=1
)
combined_df["skew_qbins"] = pd.qcut(combined_df["t2m_x_skew"], q=n_qbins, precision=1)
combined_df["var_qbins"] = pd.qcut(combined_df["t2m_x_var"], q=n_qbins, precision=1)
combined_df["ar1_qbins"] = pd.qcut(
    combined_df["t2m_x_ar1"],
    q=n_qbins,
    precision=1,
)

# fixing the first variance bin looking like this : (0.16999999999999998, 0.58]
var_qbin_cats = combined_df["var_qbins"].cat.categories.to_list()
new_first_interval_var = pd.Interval(
    round(var_qbin_cats[0].left, 2), round(var_qbin_cats[0].right, 2), closed="right"
)
var_qbin_cats[0] = new_first_interval_var
combined_df["var_qbins"] = combined_df["var_qbins"].cat.rename_categories(var_qbin_cats)


# fixing the first bin looking like this : (0.16999999999999998, 0.58]
ar1_qbin_cats = combined_df["ar1_qbins"].cat.categories.to_list()
new_first_interval = pd.Interval(
    round(ar1_qbin_cats[0].left, 2), round(ar1_qbin_cats[0].right, 2), closed="right"
)
ar1_qbin_cats[0] = new_first_interval
combined_df["ar1_qbins"] = combined_df["ar1_qbins"].cat.rename_categories(ar1_qbin_cats)
# ---------- end manual bin fix

# combined_df.reset_index()[['lon', 'lat']].value_counts() # there are 14728 land gridcells
# combined_df = combined_df.sort_values(by=["tmax_diff_qbins", "skew_qbins"])
combined_df = combined_df.reset_index().sort_values(
    by=["t2m_x_mean_diff", "t2m_x_skew"]
)

# combined_df["tmax_diff_qbins"] = combined_df["tmax_diff_qbins"].astype(str)
# combined_df["skew_qbins"] = combined_df["skew_qbins"].astype(str)
# combined_df["var_qbins"] = combined_df["var_qbins"].astype(str)
# combined_df["ar1_qbins"] = combined_df["ar1_qbins"].astype(str)

# skew_bin_ordering_dict = {
#     "skew_qbins": skew_per_tmax_diff_bins.cat.categories.astype(str),
#     "hwf_bins": hwf_bins.cat.categories.astype(str),
# }


def get_heatmap(
    combined_df, y_name_var, y_name_label, x_name_base="tmax_diff", use_qbins=True
):
    if use_qbins:
        x_name = f"{x_name_base}_qbins"
    else:
        x_name = f"{x_name_base}_bins"

    df = combined_df.copy()

    bin_ordering_dict = {
        x_name: df[x_name].cat.categories.astype(str),
        y_name_var: df[y_name_var].cat.categories.astype(str),
    }

    df[x_name] = df[x_name].astype(str)
    df[y_name_var] = df[y_name_var].astype(str)

    # count (2d histogram)
    fig_count = (
        df.hvplot.heatmap(
            x=x_name,
            y=y_name_var,
            C="t2m_x.t2m_x_threshold.HWF",  # in this plot, this isn't doing anything
            reduce_function=np.size,
            fields={
                "t2m_x.t2m_x_threshold.HWF": "count"
            },  # the tooltip displays count instead of hwf
            cmap=reds_discrete,
            title=f"gridcell count by climatological {y_name_label} and mean tmax shift\nThere are 14,728 land gridcells",
            xlabel=f"tmax mean({new_years[0]}:{new_years[1]}) - mean({ref_years[0]}:{ref_years[1]})\nanomalies wrt {ref_years[0]}-{ref_years[1]} (C)",
            ylabel=f"sample {y_name_label} over {ref_years[0]}-{ref_years[1]}",
            clabel="number of gridcells",
        )
        .opts(clim=(0, 150))
        .redim.values(  # hack to order the bins
            **bin_ordering_dict
        )
    )

    # hwf
    fig_hwf = (
        df.hvplot.heatmap(
            x=x_name,
            y=y_name_var,
            C="t2m_x.t2m_x_threshold.HWF",
            reduce_function=np.mean,
            cmap=reds_discrete,
            title="Change in HWF",
            xlabel="Change in Daily Max Anomalies (C)",
            ylabel=f"Climatological {y_name_label}",
            clabel="Days",
            # title=f"mean shift in heatwave frequency\nby climatological {y_name_label} and mean tmax shift",
            # xlabel=f"tmax mean({new_years[0]}:{new_years[1]}) - mean({ref_years[0]}:{ref_years[1]})\nanomalies wrt {ref_years[0]}-{ref_years[1]} (C)",
            # ylabel=f"sample {y_name_label} over {ref_years[0]}-{ref_years[1]}",
            # clabel=f"heatwave frequency (days)\nmean({new_years[0]}:{new_years[1]}) - mean({ref_years[0]}:{ref_years[1]})",
            clim=(1, 11),
            # min_count=10,
        )
        .redim.values(  # hack to order the bins
            **bin_ordering_dict
        )
        .opts(cmap=reds_discrete, cticks=[1, 3, 5, 7, 9, 11], xrotation=45)
    )

    # hwd
    fig_hwd = (
        df.hvplot.heatmap(
            x=x_name,
            y=y_name_var,
            C="t2m_x.t2m_x_threshold.HWD",
            reduce_function=np.mean,
            title="Change in HWD",
            xlabel="Change in Daily Max Anomalies (C)",
            ylabel=f"Climatological {y_name_label}",
            clabel="Days",
            # title=f"mean shift in heatwave duration\nby climatological {y_name_label} and mean tmax shift",
            # xlabel=f"tmax mean({new_years[0]}:{new_years[1]}) - mean({ref_years[0]}:{ref_years[1]})\nanomalies wrt {ref_years[0]}-{ref_years[1]} (C)",
            # ylabel=f"sample {y_name_label} over {ref_years[0]}-{ref_years[1]}",
            # min_count=10,
        )
        .redim.values(  # hack to order the bins
            **bin_ordering_dict
        )
        .opts(
            cmap=reds_discrete,
            cticks=[0, 1, 2, 3, 4, 5],
            clim=(0, 5),
            # clabel=f"heatwave duration (days)\nmean({new_years[0]}:{new_years[1]}) - mean({ref_years[0]}:{ref_years[1]})",
            xrotation=45,
        )
    )
    # cumulative intensity
    fig_sumheat = (
        df.hvplot.heatmap(
            x=x_name,
            y=y_name_var,
            C="t2m_x.t2m_x_threshold.sumHeat",
            reduce_function=np.mean,
            # data_aspect=1,
            cmap=reds_discrete,
            title="Change in sumHeat",
            xlabel="Change in Daily Max Anomalies (C)",
            ylabel=f"Climatological {y_name_label}",
            clabel="T Anomalies (C)",
            # title=f"mean shift in heatwave cumulative intensity\nby climatological {y_name_label} and mean tmax shift",
            # xlabel="tmax mean(1986:2021) - mean(1950:1985)\nanomalies wrt 1960-1985 (C)",
            # ylabel=f"sample {y_name_label} over 1960-1985",
            # clabel=f"heatwave cumulative intensity (degC anom)\nmean({new_years[0]}:{new_years[1]}) - mean({ref_years[0]}:{ref_years[1]})",
            clim=(1, 21),
            # min_count=10,
        )
        .redim.values(  # hack to order the bins
            **bin_ordering_dict
        )
        .opts(cticks=np.linspace(1, 21, 6), xrotation=45)
    )

    if use_qbins:
        figlist = [
            fig_hwf,
            fig_hwd,
            fig_sumheat,
            fig_count,
        ]
    else:
        figlist = [
            fig_count,
            fig_hwf,
            fig_hwd,
            fig_sumheat,
        ]
    return figlist


### variance ----------------------------------------

figlist_var_qbins = get_heatmap(
    combined_df, y_name_var="var_qbins", y_name_label="variance", use_qbins=True
)

fig_layout_var_qbins = hv.Layout(figlist_var_qbins[0:3]).cols(1)
# fig_layout_var_qbins
# hvplot.save(fig_layout_var_qbins, "fig_qbins_var.html")

### skewness ----------------------------------------


figlist_skewness_qbins = get_heatmap(
    combined_df, y_name_var="skew_qbins", y_name_label="skewness", use_qbins=True
)
fig_layout_skewness_qbins = hv.Layout(figlist_skewness_qbins[0:3]).cols(1)
# fig_layout_skewness_qbins


# mpl_render = hv.renderer('matplotlib')
# mpl_skewness_qbins = mpl_render.get_plot(fig_layout_skewness_qbins)

# hvplot.save(fig_layout_skewness_qbins, "fig_qbins_skew.html")

### ar1 ----------------------------------------
figlist_ar1_qbins = get_heatmap(
    combined_df, y_name_var="ar1_qbins", y_name_label="AR(1)", use_qbins=True
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
updated_fig_qbins_list = []
for i, subplot in enumerate(figlist_qbins):
    new_label = f"({letter_ordering[i]})"  # this sets the format to (a), (b), ..
    updated_subplot = subplot.opts(
        hooks=[partial(subplot_label_hook, sub_label=new_label)]
    )
    updated_fig_qbins_list.append(updated_subplot)

fig_qbins = hv.Layout(updated_fig_qbins_list).cols(2)

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

# hvplot.save(fig_qbins_final, "fig_qbins.png")
#########################################


def get_scatter(
    deg_df,
    x_var,
    x_label,
    deg,
    size=5,
    alpha_pt=0.02,
    ylim_hwf=(-5, 25),
    ylim_hwd=(-5, 10),
    # ylim_avi=(-2, 3),
    ylim_sumheat=(-5, 70),
    color_pt="red",
    color_line="red",
    label_curve="",
):
    fig_hwf_scatter = deg_df.hvplot.scatter(
        x=x_var,
        y="t2m_x.t2m_x_threshold.HWF",
        c="t2m_x_mean_diff",
        s=size,
        alpha=alpha_pt,
        # cmap=reds_discrete,
        color=color_pt,
    ).opts(
        xlabel=f"climatological {x_label}",
        ylabel="change in HWF (days)",
        title=f"obs, filtered to {deg - 1}.75 < mean tmax change < {deg}.25\nmean({new_years[0]}:{new_years[1]}) - mean({ref_years[0]}:{ref_years[1]})",
        # width=600,
        # height=400,
        ylim=ylim_hwf,
    )

    hwf_fitted = sm.nonparametric.lowess(
        exog=deg_df[x_var], endog=deg_df["t2m_x.t2m_x_threshold.HWF"], frac=2 / 3
    )
    # eval_x = np.linspace(deg_df[x_var].min(), deg_df[x_var].max(), num=500)
    # hwf_fitted, hwf_bottom, hwf_top = lowess_with_confidence_bounds(
    #     deg_df[x_var],
    #     deg_df["t2m_x.t2m_x_threshold.HWF"],
    #     eval_x,
    #     lowess_kw={"frac": 2 / 3},
    # )
    # fig_hwf_ci = hv.Area(
    #     x=eval_x, y=hwf_bottom, y2=hwf_top, alpha=0.3, color=color_ci#, label="Uncertainty"
    # )
    fig_hwf_fitted = hv.Curve(
        zip(hwf_fitted[:, 0], hwf_fitted[:, 1]), label=label_curve
    ).opts(color=color_line)
    # make figure
    fig_hwf = fig_hwf_scatter * fig_hwf_fitted

    # hwd ---------------------------------------------
    fig_hwd_scatter = deg_df.hvplot.scatter(
        x=x_var,
        y="t2m_x.t2m_x_threshold.HWD",
        c="t2m_x_mean_diff",
        s=size,
        alpha=alpha_pt,
        # cmap=reds_discrete,
        color=color_pt,
    ).opts(
        xlabel=f"climatological {x_label}",
        ylabel="change in HWD (days)",
        title=f"obs, filtered to {deg - 1}.75 < mean tmax change < {deg}.25\nmean({new_years[0]}:{new_years[1]}) - mean({ref_years[0]}:{ref_years[1]})",
        # width=600,
        # height=400,
        ylim=ylim_hwd,
    )
    hwd_fitted = sm.nonparametric.lowess(
        exog=deg_df[x_var], endog=deg_df["t2m_x.t2m_x_threshold.HWD"], frac=2 / 3
    )
    fig_hwd_fitted = hv.Curve(
        zip(hwd_fitted[:, 0], hwd_fitted[:, 1]), label=label_curve
    ).opts(color=color_line)
    fig_hwd = fig_hwd_scatter * fig_hwd_fitted

    # # avi ---------------------------------------------
    # fig_avi_scatter = deg_df.hvplot.scatter(
    #     x=x_var,
    #     y="t2m_x.t2m_x_threshold.AVI",
    #     c="t2m_x_mean_diff",
    #     s=size,
    #     cmap=reds_discrete,
    # ).opts(
    #     xlabel=f"climatological {x_label}",
    #     ylabel="change in average heatwave intensity (degC)",
    #     title=f"obs, filtered to {deg - 1}.95 < mean tmax change < {deg}.05\nmean({new_years[0]}:{new_years[1]}) - mean({ref_years[0]}:{ref_years[1]})",
    #     width=600,
    #     height=400,
    #     ylim=ylim_avi,
    # )
    # avi_fitted = sm.nonparametric.lowess(
    #     exog=deg_df[x_var], endog=deg_df["t2m_x.t2m_x_threshold.AVI"], frac=2 / 3
    # )
    # fig_avi_fitted = hv.Curve(zip(avi_fitted[:, 0], avi_fitted[:, 1]))
    # fig_avi = fig_avi_scatter * fig_avi_fitted

    # sumheat ---------------------------------------------
    fig_sumheat_scatter = deg_df.hvplot.scatter(
        x=x_var,
        y="t2m_x.t2m_x_threshold.sumHeat",
        c="t2m_x_mean_diff",
        s=size,
        alpha=alpha_pt,
        # cmap=reds_discrete,
        color=color_pt,
    ).opts(
        xlabel=f"climatological {x_label}",
        ylabel="change in sumHeat (C)",
        title=f"obs, filtered to {deg - 1}.75 < mean tmax change < {deg}.25\nmean({new_years[0]}:{new_years[1]}) - mean({ref_years[0]}:{ref_years[1]})",
        # ylim=(-5, 25),
        ylim=ylim_sumheat,
        # width=600,
        # height=400,
    )
    sumheat_fitted = sm.nonparametric.lowess(
        exog=deg_df[x_var], endog=deg_df["t2m_x.t2m_x_threshold.sumHeat"], frac=2 / 3
    )
    fig_sumheat_fitted = hv.Curve(
        zip(sumheat_fitted[:, 0], sumheat_fitted[:, 1]), label=label_curve
    ).opts(color=color_line)
    fig_sumheat = fig_sumheat_scatter * fig_sumheat_fitted

    figlist = [
        fig_hwf,
        fig_hwd,
        # fig_avi,
        fig_sumheat,
    ]
    return hv.Layout(figlist).cols(1)


# 2deg --------------------------------

deg2_obs_df = combined_df.loc[
    (combined_df["t2m_x_mean_diff"] >= 1.75) & (combined_df["t2m_x_mean_diff"] <= 2.25)
]

fig_skew_2deg_obs = get_scatter(
    deg2_obs_df,
    x_var="t2m_x_skew",
    x_label="skewness",
    deg=2,
    alpha_pt=0.1,
    label_curve="observed",
    ylim_hwf=(0, 35),
    ylim_hwd=(0, 14),
    ylim_sumheat=(10, 50),
)
# hvplot.save(fig_skew_2deg_obs, 'fig_skew_2deg_obs.html')

fig_var_2deg_obs = get_scatter(
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
)
# hvplot.save(fig_var_2deg_obs, 'fig_var_2deg_obs.html')

fig_ar1_2deg_obs = get_scatter(
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
)
# hvplot.save(fig_ar1_2deg_obs, 'fig_ar1_2deg_obs.html')


# 2 degree, synthetic ---------------------------------------------------
hw_old_2deg = hw_synth_2deg.sel(time=slice(str(ref_years[0]), str(ref_years[1])))
hw_new_2deg = hw_synth_2deg.sel(time=slice(str(new_years[0]), str(new_years[1])))
hw_mean_diff_2deg = hw_new_2deg.mean(dim="time") - hw_old_2deg.mean(dim="time")

combined_synth_2deg_ds = xr.merge([climatology_stats, hw_mean_diff_2deg], join="exact")
combined_synth_2deg_df = combined_synth_2deg_ds.to_dataframe().dropna(how="all")

fig_var_2deg_synth = get_scatter(
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
)
fig_var_2deg_synth.map(lambda x: x.opts(xlim=(-1, 70)), hv.Curve)

# temp ---------------------------------------------------
combined_synth_2deg_df["t2m_x_sd"] = np.sqrt(combined_synth_2deg_df["t2m_x_var"])
fig_sd_2deg_synth = get_scatter(
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
)
# end temp ------------------------------


fig_skew_2deg_synth = get_scatter(
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
)
fig_skew_2deg_synth.map(lambda x: x.opts(xlim=(-1, 0.75)), hv.Curve)


# temp ---------------------------------------------------

skew_hwf_2deg_synth_fitted = sm.nonparametric.lowess(
    exog=combined_synth_2deg_df["t2m_x_skew"],
    endog=combined_synth_2deg_df["t2m_x.t2m_x_threshold.HWF"],
    frac=2 / 3,
)
# get gradients
skew_hwf_2deg_synth_grad = np.gradient(
    skew_hwf_2deg_synth_fitted[:, 1], skew_hwf_2deg_synth_fitted[:, 0]
)

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
np.polyfit(
    deg2_obs_df["t2m_x_skew"], deg2_obs_df["t2m_x.t2m_x_threshold.sumHeat"], deg=1
)

hv.Curve(zip(skew_hwf_2deg_synth_fitted[:, 0], skew_hwf_2deg_synth_grad))

hv.Curve(zip(skew_hwf_2deg_synth_fitted[:, 0], skew_hwf_2deg_synth_fitted[:, 1]))


np.polyfit(
    combined_synth_2deg_df["t2m_x_skew"],
    combined_synth_2deg_df["t2m_x.t2m_x_threshold.HWD"],
    deg=1,
)
np.polyfit(deg2_obs_df["t2m_x_skew"], deg2_obs_df["t2m_x.t2m_x_threshold.HWD"], deg=1)


# end temp ------------------------------

fig_ar1_2deg_synth = get_scatter(
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
    # + fig_ar1_hwf_2deg
    # + fig_ar1_hwd_2deg
    # + fig_ar1_heatsum_2deg
).cols(2)

# weird ordering bc I want to go vertical instead of horizontal
# letter ordering = string.ascii_lowercase
letter_ordering = ["a", "d", "b", "e", "c", "f"]
updated_fig_2deg_list = []
for i, subplot in enumerate(fig_2deg):
    new_label = f"({letter_ordering[i]})"  # this sets the format to (a), (b), ..
    updated_subplot = subplot.opts(
        hooks=[partial(subplot_label_hook, sub_label=new_label)]
    )
    updated_fig_2deg_list.append(updated_subplot)

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
# hvplot.save(fig_2deg_final, 'fig_2deg.png')
