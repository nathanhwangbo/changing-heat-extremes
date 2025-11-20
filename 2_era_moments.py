import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import hvplot.xarray
import colorcet as cc
import matplotlib as mpl
import tastymap
import regionmask
import holoviews as hv
import glob
from xarray_einstats import stats  # wrapper around apply_ufunc for moments
import pandas as pd
import hvplot.pandas
from holoviews import opts
import statsmodels.api as sm


xr.set_options(use_new_combine_kwarg_defaults=True)

# hvplot.extension("matplotlib")
# hvplot.extension("bokeh")

rdbu_discrete = tastymap.cook_tmap("RdYlBu_r", num_colors=12).cmap
reds_discrete = tastymap.cook_tmap("cet_CET_L18", num_colors=12)[
    1:11
].cmap  # get rid of white
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

ref_years = [1980, 1999]  # the time period the thresholds are calculated over
new_years = [2006, 2025]  # the time period we're gonna compare to

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

# combined_ds.plot.scatter(
#     x="t2m_x_mean_diff", y="t2m_x_skew", hue="t2m_x.t2m_x_threshold.HWF", s=10
# )


##########################################
# the above has too many points
# let's try hexbins
#########################################

combined_df = combined_ds.to_dataframe().dropna(how="all")


def get_hexplots(y_name_var, y_name_label, x_name="t2m_x_mean_diff"):
    # count (2d histogram)
    fig_count = combined_df.hvplot.hexbin(
        x=x_name,
        y=y_name_var,
        cmap=reds_discrete,
        # clim=(0, 50), # for some reason clim inside doesn't work if we're doing count
        title=f"gridcell count by climatological {y_name_label} and mean tmax shift\nThere are 14,728 land gridcells",
        xlabel=f"tmax mean({new_years[0]}:{new_years[1]}) - mean({ref_years[0]}:{ref_years[1]})\nanomalies wrt {ref_years[0]}-{ref_years[1]} (C)",
        ylabel=f"sample {y_name_label} over {ref_years[0]}-{ref_years[1]}",
        clabel="number of gridcells",
        gridsize=10,
        # min_count=10,
    ).opts(clim=(0, 150))

    # hwf
    fig_hwf = combined_df.hvplot.hexbin(
        x=x_name,
        y=y_name_var,
        C="t2m_x.t2m_x_threshold.HWF",
        reduce_function=np.mean,
        cmap=reds_discrete,
        title=f"mean shift in heatwave frequency\nby climatological {y_name_label} and mean tmax shift",
        xlabel=f"tmax mean({new_years[0]}:{new_years[1]}) - mean({ref_years[0]}:{ref_years[1]})\nanomalies wrt {ref_years[0]}-{ref_years[1]} (C)",
        ylabel=f"sample {y_name_label} over {ref_years[0]}-{ref_years[1]}",
        clabel=f"heatwave frequency (days)\nmean({new_years[0]}:{new_years[1]}) - mean({ref_years[0]}:{ref_years[1]})",
        clim=(-4, 15),
        width=600,
        height=400,
        gridsize=10,
        # min_count=10,
    )

    # hwd
    fig_hwd = combined_df.hvplot.hexbin(
        x=x_name,
        y=y_name_var,
        C="t2m_x.t2m_x_threshold.HWD",
        reduce_function=np.mean,
        cmap=reds_discrete,
        title=f"mean shift in heatwave duration\nby climatological {y_name_label} and mean tmax shift",
        xlabel=f"tmax mean({new_years[0]}:{new_years[1]}) - mean({ref_years[0]}:{ref_years[1]})\nanomalies wrt {ref_years[0]}-{ref_years[1]} (C)",
        ylabel=f"sample {y_name_label} over {ref_years[0]}-{ref_years[1]}",
        clabel=f"heatwave duration (days)\nmean({new_years[0]}:{new_years[1]}) - mean({ref_years[0]}:{ref_years[1]})",
        clim=(-2, 8),
        width=600,
        height=400,
        gridsize=10,
        # min_count=10,
    )

    # average intensity
    fig_avi = combined_df.hvplot.hexbin(
        x=x_name,
        y=y_name_var,
        C="t2m_x.t2m_x_threshold.AVI",
        reduce_function=np.mean,
        # data_aspect=1,
        cmap=reds_discrete,
        title=f"mean shift in heatwave average intensity\nby climatological {y_name_label} and mean tmax shift",
        xlabel=f"tmax mean({new_years[0]}:{new_years[1]}) - mean({ref_years[0]}:{ref_years[1]})\nanomalies wrt {ref_years[0]}-{ref_years[1]} (C)",
        ylabel=f"sample {y_name_label} over {ref_years[0]}-{ref_years[1]}",
        clabel=f"heatwave avg intensity (degC anom)\nmean({new_years[0]}:{new_years[1]}) - mean({ref_years[0]}:{ref_years[1]})",
        clim=(0, 0.75),
        width=600,
        height=400,
        gridsize=10,
        # min_count=10,
    )

    # cumulative intensity
    fig_sumheat = combined_df.hvplot.hexbin(
        x=x_name,
        y=y_name_var,
        C="t2m_x.t2m_x_threshold.sumHeat",
        reduce_function=np.mean,
        # data_aspect=1,
        cmap=reds_discrete,
        title=f"mean shift in heatwave cumulative intensity\nby climatological {y_name_label} and mean tmax shift",
        xlabel=f"tmax mean({new_years[0]}:{new_years[1]}) - mean({ref_years[0]}:{ref_years[1]})\nanomalies wrt {ref_years[0]}-{ref_years[1]} (C)",
        ylabel=f"sample {y_name_label} over {ref_years[0]}-{ref_years[1]}",
        clabel=f"heatwave cumulative intensity (degC anom)\nmean({new_years[0]}:{new_years[1]}) - mean({ref_years[0]}:{ref_years[1]})",
        clim=(-5, 40),
        width=600,
        height=400,
        gridsize=10,
        # min_count=10,
    )

    figlist = [
        fig_count,
        fig_hwf,
        fig_hwd,
        fig_avi,
        fig_sumheat,
    ]
    return figlist


### skewness ----------------------------------------


combined_ds["t2m_x_skew"].hvplot(
    projection=ccrs.PlateCarree(),
    coastline=True,
    cmap=rdbu_discrete,
    # clim=(0, 30),
    title=f"climatological skewness ({ref_years[0]}:{ref_years[1]})",
    clabel="skewness",
).opts(fontscale=1.5)  # .opts(width=600, height=400)#
figlist_skewness = get_hexplots("t2m_x_skew", "skewness")
fig_skewness_count = figlist_skewness[0].opts(width=600, height=400)
fig_layout_skewness = hv.Layout(figlist_skewness[1:]).cols(2)
fig_layout_skewness

# size_opts = dict(width=700, height=400)
# fig_layout_skewness.opts(opts.HexTiles(**size_opts))

# hvplot.save(fig_layout_skewness, "fig_hex10_skew.html")


### variance ----------------------------------------

combined_ds["t2m_x_var"].hvplot(
    projection=ccrs.PlateCarree(),
    coastline=True,
    cmap=reds_discrete,
    # clim=(0, 30),
    title=f"climatological variance ({ref_years[0]}:{ref_years[1]})",
    clabel="variance (degC^2)",
).opts(fontscale=1.5)

figlist_var = get_hexplots("t2m_x_var", "variance")
fig_var_count = figlist_var[0].opts(xlim=(-1, 3), width=600, height=400)

fig_layout_var = hv.Layout(figlist_var[1:]).cols(2)
fig_layout_var
# hvplot.save(fig_layout_var, "fig_hex10_var.html")

### ar1 ----------------------------------------
combined_ds["t2m_x_ar1"].hvplot(
    projection=ccrs.PlateCarree(),
    coastline=True,
    cmap=reds_discrete,
    # clim=(0, 30),
    title=f"climatological AR(1) ({ref_years[0]}:{ref_years[1]})",
    clabel="AR(1)",
).opts(fontscale=1.5)


figlist_ar1 = get_hexplots("t2m_x_ar1", "AR(1)")
fig_ar1_count = figlist_ar1[0].opts(width=600, height=400)

fig_layout_ar1 = hv.Layout(figlist_ar1[1:]).cols(2)
# hvplot.save(fig_layout_ar1, "fig_hex10_ar1.html")


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
    # # plot count (i.e. 2d histogram)
    # don't need this one here bc quantiles guarantee rough uniformity
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
            title=f"mean shift in heatwave frequency\nby climatological {y_name_label} and mean tmax shift",
            xlabel=f"tmax mean({new_years[0]}:{new_years[1]}) - mean({ref_years[0]}:{ref_years[1]})\nanomalies wrt {ref_years[0]}-{ref_years[1]} (C)",
            ylabel=f"sample {y_name_label} over {ref_years[0]}-{ref_years[1]}",
            clabel=f"heatwave frequency (days)\nmean({new_years[0]}:{new_years[1]}) - mean({ref_years[0]}:{ref_years[1]})",
            clim=(1, 11),
            width=600,
            height=400,
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
            title=f"mean shift in heatwave duration\nby climatological {y_name_label} and mean tmax shift",
            xlabel=f"tmax mean({new_years[0]}:{new_years[1]}) - mean({ref_years[0]}:{ref_years[1]})\nanomalies wrt {ref_years[0]}-{ref_years[1]} (C)",
            ylabel=f"sample {y_name_label} over {ref_years[0]}-{ref_years[1]}",
            width=600,
            height=400,
            # min_count=10,
        )
        .redim.values(  # hack to order the bins
            **bin_ordering_dict
        )
        .opts(
            cmap=reds_discrete,
            cticks=[0, 1, 2, 3, 4, 5],
            clim=(0, 5),
            clabel=f"heatwave duration (days)\nmean({new_years[0]}:{new_years[1]}) - mean({ref_years[0]}:{ref_years[1]})",
            xrotation=45,
        )
    )

    # average intensity
    fig_avi = (
        df.hvplot.heatmap(
            x=x_name,
            y=y_name_var,
            C="t2m_x.t2m_x_threshold.AVI",
            reduce_function=np.mean,
            # data_aspect=1,
            cmap=reds_discrete,
            title=f"mean shift in heatwave average intensity\nby climatological {y_name_label} and mean tmax shift",
            xlabel=f"tmax mean({new_years[0]}:{new_years[1]}) - mean({ref_years[0]}:{ref_years[1]})\nanomalies wrt {ref_years[0]}-{ref_years[1]} (C)",
            ylabel=f"sample {y_name_label} over {ref_years[0]}-{ref_years[1]}",
            clabel=f"heatwave avg intensity (degC anom)\nmean({new_years[0]}:{new_years[1]}) - mean({ref_years[0]}:{ref_years[1]})",
            clim=(0, 0.5),
            width=600,
            height=400,
            # min_count=10,
        )
        .redim.values(  # hack to order the bins
            **bin_ordering_dict
        )
        .opts(xrotation=45)
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
            title=f"mean shift in heatwave cumulative intensity\nby climatological {y_name_label} and mean tmax shift",
            xlabel="tmax mean(1986:2021) - mean(1950:1985)\nanomalies wrt 1960-1985 (C)",
            ylabel=f"sample {y_name_label} over 1960-1985",
            clabel=f"heatwave cumulative intensity (degC anom)\nmean({new_years[0]}:{new_years[1]}) - mean({ref_years[0]}:{ref_years[1]})",
            clim=(1, 21),
            width=600,
            height=400,
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
            fig_avi,
            fig_sumheat,
        ]
    else:
        figlist = [
            fig_count,
            fig_hwf,
            fig_hwd,
            fig_avi,
            fig_sumheat,
        ]
    return figlist


### skewness ----------------------------------------

# hvplot.extension('matplotlib')
# hvplot.extension('bokeh')

figlist_skewness_qbins = get_heatmap(
    combined_df, y_name_var="skew_qbins", y_name_label="skewness", use_qbins=True
)
fig_layout_skewness_qbins = hv.Layout(figlist_skewness_qbins).cols(2)
fig_layout_skewness_qbins


# mpl_render = hv.renderer('matplotlib')
# mpl_skewness_qbins = mpl_render.get_plot(fig_layout_skewness_qbins)

# hvplot.save(fig_layout_skewness_qbins, "fig_qbins_skew.html")


### variance ----------------------------------------

figlist_var_qbins = get_heatmap(
    combined_df, y_name_var="var_qbins", y_name_label="variance", use_qbins=True
)

fig_layout_var_qbins = hv.Layout(figlist_var_qbins).cols(2)
fig_layout_var_qbins
# hvplot.save(fig_layout_var_qbins, "fig_qbins_var.html")

### ar1 ----------------------------------------
figlist_ar1_qbins = get_heatmap(
    combined_df, y_name_var="ar1_qbins", y_name_label="AR(1)", use_qbins=True
)
fig_layout_ar1_qbins = hv.Layout(figlist_ar1_qbins).cols(2)
# hvplot.save(fig_layout_ar1_qbins, "fig_qbins_ar1.html")

##########################################
# option 3:
# let's combine points into equal-sized bins
#########################################

n_bins = 10
combined_df["tmax_diff_bins"] = pd.cut(
    combined_df["t2m_x_mean_diff"], n_bins, precision=1
)
combined_df["skew_bins"] = pd.cut(combined_df["t2m_x_skew"], n_bins, precision=1)
combined_df["var_bins"] = pd.cut(combined_df["t2m_x_var"], n_bins, precision=1)
combined_df["ar1_bins"] = pd.cut(combined_df["t2m_x_ar1"], n_bins, precision=1)


# combined_df.reset_index()[['lon', 'lat']].value_counts() # there are 14728 land gridcells
combined_df = combined_df.sort_values(by=["tmax_diff_bins", "skew_bins"])
combined_df["tmax_diff_bins"] = combined_df["tmax_diff_bins"].astype(str)
combined_df["skew_bins"] = combined_df["skew_bins"].astype(str)
combined_df["var_bins"] = combined_df["var_bins"].astype(str)
combined_df["ar1_bins"] = combined_df["ar1_bins"].astype(str)


### skewness ----------------------------------------

figlist_skewness_bins = get_heatmap(
    combined_df, y_name_var="skew_bins", y_name_label="skewness", use_qbins=False
)
fig_skewness_count_bins = figlist_skewness_bins[0].opts(width=600, height=400)
fig_layout_skewness_bins = hv.Layout(figlist_skewness_bins[1:]).cols(2)
fig_layout_skewness_bins

# hvplot.save(fig_layout_skewness_bins, "fig_bins_skew.html")


### variance ----------------------------------------

figlist_var_bins = get_heatmap(
    combined_df, y_name_var="var_bins", y_name_label="variance", use_qbins=False
)
fig_var_count_bins = figlist_var_bins[0].opts(xlim=(-1, 3), width=600, height=400)

fig_layout_var_bins = hv.Layout(figlist_var_bins[1:]).cols(2)
fig_layout_var_bins
# hvplot.save(fig_layout_var_bins, "fig_bins_var.html")

### ar1 ----------------------------------------
figlist_ar1_bins = get_heatmap(
    combined_df, y_name_var="ar1_bins", y_name_label="AR(1)", use_qbins=False
)
fig_ar1_count = figlist_ar1_bins[0].opts(width=600, height=400)

fig_layout_ar1_bins = hv.Layout(figlist_ar1_bins[1:]).cols(2)
# hvplot.save(fig_layout_ar1_bins, "fig_bins_ar1.html")


##########################################
# option 4:
# let's collapse the x axis by compauting "per degree warming"
#########################################


def get_scatter(
    deg_df,
    x_var,
    x_label,
    deg,
    size=5,
    ylim_hwf=(-5, 25),
    ylim_hwd=(-5, 10),
    ylim_avi=(-2, 3),
    ylim_sumheat=(-5, 70),
):
    fig_hwf_scatter = deg_df.hvplot.scatter(
        x=x_var,
        y="t2m_x.t2m_x_threshold.HWF",
        c="t2m_x_mean_diff",
        s=size,
        cmap=reds_discrete,
    ).opts(
        xlabel=f"climatological {x_label}",
        ylabel="change in heatwave frequency (days)",
        title=f"obs, filtered to {deg - 1}.95 < mean tmax change < {deg}.05\nmean({new_years[0]}:{new_years[1]}) - mean({ref_years[0]}:{ref_years[1]})",
        width=600,
        height=400,
        ylim=ylim_hwf,
    )
    hwf_fitted = sm.nonparametric.lowess(
        exog=deg_df[x_var], endog=deg_df["t2m_x.t2m_x_threshold.HWF"], frac=2 / 3
    )
    fig_hwf_fitted = hv.Curve(zip(hwf_fitted[:, 0], hwf_fitted[:, 1]))
    fig_hwf = fig_hwf_scatter * fig_hwf_fitted

    # hwd ---------------------------------------------
    fig_hwd_scatter = deg_df.hvplot.scatter(
        x=x_var,
        y="t2m_x.t2m_x_threshold.HWD",
        c="t2m_x_mean_diff",
        s=size,
        cmap=reds_discrete,
    ).opts(
        xlabel=f"climatological {x_label}",
        ylabel="change in heatwave duration (days)",
        title=f"obs, filtered to {deg - 1}.95 < mean tmax change < {deg}.05\nmean({new_years[0]}:{new_years[1]}) - mean({ref_years[0]}:{ref_years[1]})",
        width=600,
        height=400,
        ylim=ylim_hwd,
    )
    hwd_fitted = sm.nonparametric.lowess(
        exog=deg_df[x_var], endog=deg_df["t2m_x.t2m_x_threshold.HWD"], frac=2 / 3
    )
    fig_hwd_fitted = hv.Curve(zip(hwd_fitted[:, 0], hwd_fitted[:, 1]))
    fig_hwd = fig_hwd_scatter * fig_hwd_fitted

    # avi ---------------------------------------------
    fig_avi_scatter = deg_df.hvplot.scatter(
        x=x_var,
        y="t2m_x.t2m_x_threshold.AVI",
        c="t2m_x_mean_diff",
        s=size,
        cmap=reds_discrete,
    ).opts(
        xlabel=f"climatological {x_label}",
        ylabel="change in average heatwave intensity (degC)",
        title=f"obs, filtered to {deg - 1}.95 < mean tmax change < {deg}.05\nmean({new_years[0]}:{new_years[1]}) - mean({ref_years[0]}:{ref_years[1]})",
        width=600,
        height=400,
        ylim=ylim_avi,
    )
    avi_fitted = sm.nonparametric.lowess(
        exog=deg_df[x_var], endog=deg_df["t2m_x.t2m_x_threshold.AVI"], frac=2 / 3
    )
    fig_avi_fitted = hv.Curve(zip(avi_fitted[:, 0], avi_fitted[:, 1]))
    fig_avi = fig_avi_scatter * fig_avi_fitted

    # sumheat ---------------------------------------------
    fig_sumheat_scatter = deg_df.hvplot.scatter(
        x=x_var,
        y="t2m_x.t2m_x_threshold.sumHeat",
        c="t2m_x_mean_diff",
        s=size,
        cmap=reds_discrete,
    ).opts(
        xlabel=f"climatological {x_label}",
        ylabel="change in cumulative heat (degC)",
        title=f"obs, filtered to {deg - 1}.95 < mean tmax change < {deg}.05\nmean({new_years[0]}:{new_years[1]}) - mean({ref_years[0]}:{ref_years[1]})",
        # ylim=(-5, 25),
        width=600,
        height=400,
    )
    sumheat_fitted = sm.nonparametric.lowess(
        exog=deg_df[x_var], endog=deg_df["t2m_x.t2m_x_threshold.sumHeat"], frac=2 / 3
    )
    fig_sumheat_fitted = hv.Curve(zip(sumheat_fitted[:, 0], sumheat_fitted[:, 1]))
    fig_sumheat = fig_sumheat_scatter * fig_sumheat_fitted

    figlist = [
        fig_hwf,
        fig_hwd,
        fig_avi,
        fig_sumheat,
    ]
    return hv.Layout(figlist).cols(2)


deg1_obs_df = combined_df.loc[
    (combined_df["t2m_x_mean_diff"] >= 0.95) & (combined_df["t2m_x_mean_diff"] <= 1.05)
]


fig_var_1deg_obs = get_scatter(
    deg1_obs_df, x_var="t2m_x_var", x_label="variance", deg=1
)
# hvplot.save(fig_var_1deg_obs, 'fig_var_1deg_obs.html')

fig_skew_1deg_obs = get_scatter(
    deg1_obs_df, x_var="t2m_x_skew", x_label="skewness", deg=1
)
# hvplot.save(fig_skew_1deg_obs, 'fig_skew_1deg_obs.html')

fig_ar1_1deg_obs = get_scatter(deg1_obs_df, x_var="t2m_x_ar1", x_label="AR(1)", deg=1)
# hvplot.save(fig_ar1_1deg_obs, 'fig_ar1_1deg_obs.html')


# 2deg --------------------------------

deg2_obs_df = combined_df.loc[
    (combined_df["t2m_x_mean_diff"] >= 1.95) & (combined_df["t2m_x_mean_diff"] <= 2.05)
]

fig_skew_2deg_obs = get_scatter(
    deg2_obs_df, x_var="t2m_x_skew", x_label="skewness", deg=2, size=15
)
# hvplot.save(fig_skew_2deg_obs, 'fig_skew_2deg_obs.html')

fig_var_2deg_obs = get_scatter(
    deg2_obs_df, x_var="t2m_x_var", x_label="variance", deg=2, size=15
)
# hvplot.save(fig_var_2deg_obs, 'fig_var_2deg_obs.html')

fig_ar1_2deg_obs = get_scatter(
    deg2_obs_df, x_var="t2m_x_ar1", x_label="AR(1)", deg=2, size=15
)
# hvplot.save(fig_ar1_2deg_obs, 'fig_ar1_2deg_obs.html')


######################################################
# make synthetic verisons of these degree scatterplots
#####################################################


def get_fig_hex_deg(
    deg_df,
    x_var,
    x_label,
    deg,
    x_lim=(-1, 0.75),
    ylim_hwf=(-1, 18),
    ylim_hwd=(0, 6),
    ylim_avi=(-0.25, 0.6),
    ylim_sumheat=(0, 20),
):
    fig_hwf_scatter = deg_df.hvplot.hexbin(
        x=x_var,
        y="t2m_x.t2m_x_threshold.HWF",
    ).opts(
        xlabel=f"climatological {x_label}",
        ylabel="change in heatwave frequency (days)",
        title=f"synth, {deg} degree tmax change\n({new_years[0]}:{new_years[1]}) - ({ref_years[0]}:{ref_years[1]})",
        width=600,
        height=400,
        ylim=ylim_hwf,
        xlim=x_lim,
        # gridsize = 10
        min_count=50,
    )
    hwf_fitted = sm.nonparametric.lowess(
        exog=deg_df[x_var], endog=deg_df["t2m_x.t2m_x_threshold.HWF"], frac=2 / 3
    )
    fig_hwf_fitted = hv.Curve(zip(hwf_fitted[:, 0], hwf_fitted[:, 1])).opts(
        color="black"
    )
    fig_hwf = fig_hwf_scatter * fig_hwf_fitted

    # hwd ---------------------------------------------

    fig_hwd_scatter = deg_df.hvplot.hexbin(
        x=x_var,
        y="t2m_x.t2m_x_threshold.HWD",
    ).opts(
        xlabel=f"climatological {x_label}",
        ylabel="change in heatwave duration (days)",
        title=f"synth, {deg} degree tmax change\n({new_years[0]}:{new_years[1]}) - ({ref_years[0]}:{ref_years[1]})",
        width=600,
        height=400,
        ylim=ylim_hwd,
        xlim=x_lim,
        # gridsize = 40,
        min_count=50,
    )
    hwd_fitted = sm.nonparametric.lowess(
        exog=deg_df[x_var], endog=deg_df["t2m_x.t2m_x_threshold.HWD"], frac=2 / 3
    )
    fig_hwd_fitted = hv.Curve(zip(hwd_fitted[:, 0], hwd_fitted[:, 1])).opts(
        color="black"
    )
    fig_hwd = fig_hwd_scatter * fig_hwd_fitted

    # avi ---------------------------------------------
    fig_avi_scatter = deg_df.hvplot.hexbin(
        x=x_var,
        y="t2m_x.t2m_x_threshold.AVI",
    ).opts(
        xlabel=f"climatological {x_label}",
        ylabel="change in average intensity",
        title=f"synth, {deg} degree tmax change\n({new_years[0]}:{new_years[1]}) - ({ref_years[0]}:{ref_years[1]})",
        width=600,
        height=400,
        ylim=ylim_avi,
        xlim=x_lim,
        # gridsize = 40,
        min_count=50,
    )

    avi_fitted = sm.nonparametric.lowess(
        exog=deg_df[x_var], endog=deg_df["t2m_x.t2m_x_threshold.AVI"], frac=2 / 3
    )
    fig_avi_fitted = hv.Curve(zip(avi_fitted[:, 0], avi_fitted[:, 1])).opts(
        color="black"
    )
    fig_avi = fig_avi_scatter * fig_avi_fitted

    # cumulative heat ---------------------------------------------
    fig_sumheat_scatter = deg_df.hvplot.hexbin(
        x=x_var,
        y="t2m_x.t2m_x_threshold.sumHeat",
    ).opts(
        xlabel=f"climatological {x_label}",
        ylabel="change in cumulative heat (degC)",
        title=f"synth, {deg} degree tmax change\n({new_years[0]}:{new_years[1]}) - ({ref_years[0]}:{ref_years[1]})",
        width=600,
        height=400,
        ylim=ylim_sumheat,
        xlim=x_lim,
        # gridsize = 40,
        min_count=50,
    )

    sumheat_fitted = sm.nonparametric.lowess(
        exog=deg_df[x_var], endog=deg_df["t2m_x.t2m_x_threshold.sumHeat"], frac=2 / 3
    )
    fig_sumheat_fitted = hv.Curve(zip(sumheat_fitted[:, 0], sumheat_fitted[:, 1])).opts(
        color="black"
    )
    fig_sumheat = fig_sumheat_scatter * fig_sumheat_fitted

    figlist = [
        fig_hwf,
        fig_hwd,
        fig_avi,
        fig_sumheat,
    ]
    return hv.Layout(figlist).cols(2)


# 1 degree ---------------------------------------------------
hw_old_1deg = hw_synth_1deg.sel(time=slice(str(ref_years[0]), str(ref_years[1])))
hw_new_1deg = hw_synth_1deg.sel(time=slice(str(new_years[0]), str(new_years[1])))
hw_mean_diff_1deg = hw_new_1deg.mean(dim="time") - hw_old_1deg.mean(dim="time")

combined_synth_1deg_ds = xr.merge([climatology_stats, hw_mean_diff_1deg], join="exact")
combined_synth_1deg_df = combined_synth_1deg_ds.to_dataframe().dropna(how="all")

fig_var_1deg_synth = get_fig_hex_deg(
    combined_synth_1deg_df, "t2m_x_var", "variance", deg=1, x_lim=(0, 50)
)
# hvplot.save(fig_var_1deg_synth, 'fig_var_1deg_synth.html')
fig_skew_1deg_synth = get_fig_hex_deg(
    combined_synth_1deg_df, "t2m_x_skew", "skewness", deg=1
)
# hvplot.save(fig_skew_1deg_synth, 'fig_skew_1deg_synth.html')
fig_ar1_1deg_synth = get_fig_hex_deg(
    combined_synth_1deg_df, "t2m_x_ar1", "AR(1)", deg=1, x_lim=(0.5, 1)
)
# hvplot.save(fig_ar1_1deg_synth, 'fig_ar1_1deg_synth.html')


# 2 degree ---------------------------------------------------
hw_old_2deg = hw_synth_2deg.sel(time=slice(str(ref_years[0]), str(ref_years[1])))
hw_new_2deg = hw_synth_2deg.sel(time=slice(str(new_years[0]), str(new_years[1])))
hw_mean_diff_2deg = hw_new_2deg.mean(dim="time") - hw_old_2deg.mean(dim="time")

combined_synth_2deg_ds = xr.merge([climatology_stats, hw_mean_diff_2deg], join="exact")
combined_synth_2deg_df = combined_synth_2deg_ds.to_dataframe().dropna(how="all")

fig_var_2deg_synth = get_fig_hex_deg(
    combined_synth_2deg_df,
    "t2m_x_var",
    "variance",
    deg=2,
    x_lim=(0, 55),
    ylim_hwf=(0, 50),
    ylim_hwd=(0, 20),
    ylim_avi=(0, 0.8),
    ylim_sumheat=(10, 60),
)
# hvplot.save(fig_var_2deg_synth, 'fig_var_2deg_synth.html')
fig_skew_2deg_synth = get_fig_hex_deg(
    combined_synth_2deg_df,
    "t2m_x_skew",
    "skewness",
    deg=2,
    ylim_hwf=(0, 30),
    ylim_hwd=(0, 13),
    ylim_avi=(0, 1),
    ylim_sumheat=(10, 50),
)
# hvplot.save(fig_skew_2deg_synth, 'fig_skew_2deg_synth.html')
fig_ar1_2deg_synth = get_fig_hex_deg(
    combined_synth_2deg_df,
    "t2m_x_ar1",
    "AR(1)",
    deg=2,
    x_lim=(0.5, 1),
    ylim_hwf=(0, 25),
    ylim_hwd=(0, 12),
    ylim_avi=(-0.2, 1),
    ylim_sumheat=(10, 45),
)
# hvplot.save(fig_ar1_2deg_synth, 'fig_ar1_2deg_synth.html')

# plotting just the lines ------------


def combine_curves(fig_1deg, fig_2deg, x_label, col1="blue", col2="red"):
    hwf_var = hv.NdOverlay(
        {
            "1deg": fig_1deg[0].Curve.I.opts(color=col1),
            "2deg": fig_2deg[0].Curve.I.opts(color=col2),
        },
        kdims="tmax shift",
    ).opts(
        title=f"synth loess: heatwave frequency ~ {x_label}",
        xlabel=f"climatological {x_label}",
        ylabel="change in heatwave frequency",
    )

    hwd_var = hv.NdOverlay(
        {
            "1deg": fig_1deg[1].Curve.I.opts(color=col1),
            "2deg": fig_2deg[1].Curve.I.opts(color=col2),
        },
        kdims="tmax shift",
    ).opts(
        title=f"obs loess: heatwave duration ~ {x_label}",
        xlabel=f"climatological {x_label}",
        ylabel="change in heatwave duration",
    )

    avi_var = hv.NdOverlay(
        {
            "1deg": fig_1deg[2].Curve.I.opts(color=col1),
            "2deg": fig_2deg[2].Curve.I.opts(color=col2),
        },
        kdims="tmax shift",
    ).opts(
        title=f"obs loess: heatwave avg intensity ~ {x_label}",
        xlabel=f"climatological {x_label}",
        ylabel="change in heatwave avg intensity (degC)",
    )

    sumheat_var = hv.NdOverlay(
        {
            "1deg": fig_1deg[3].Curve.I.opts(color=col1),
            "2deg": fig_2deg[3].Curve.I.opts(color=col2),
        },
        kdims="tmax shift",
    ).opts(
        title=f"obs loess: heatwave cumulative heat ~ {x_label}",
        xlabel=f"climatological {x_label}",
        ylabel="change in heatwave cumulative heat (degC)",
    )

    fig_var_curves_deg = (
        (hwf_var + hwd_var + avi_var + sumheat_var)
        .cols(2)
        .opts(shared_axes=False, width=600, height=400)
    )
    return fig_var_curves_deg


fig_var_curves_deg = combine_curves(
    fig_var_1deg_obs, fig_var_2deg_obs, x_label="variance"
).opts(width=800)
# hvplot.save(fig_var_curves_deg, "fig_var_curves_deg.html")

fig_skew_curves_deg = combine_curves(
    fig_skew_1deg_obs, fig_skew_2deg_obs, x_label="skewness"
)
# hvplot.save(fig_skew_curves_deg, "fig_skew_curves_deg.html")

fig_ar1_curves_deg = combine_curves(fig_ar1_1deg_obs, fig_ar1_2deg_obs, x_label="AR(1)")
# hvplot.save(fig_ar1_curves_deg, "fig_ar1_curves_deg.html")

# synth

fig_var_curves_deg_synth = combine_curves(
    fig_var_1deg_synth, fig_var_2deg_synth, x_label="variance"
).opts(width=800)
# hvplot.save(fig_var_curves_deg_synth, "fig_var_curves_deg_synth.html")

fig_skew_curves_deg_synth = combine_curves(
    fig_skew_1deg_synth, fig_skew_2deg_synth, x_label="skewness"
)
# hvplot.save(fig_skew_curves_deg_synth, "fig_skew_curves_deg_synth.html")

fig_ar1_curves_deg_synth = combine_curves(
    fig_ar1_1deg_synth, fig_ar1_2deg_synth, x_label="AR(1)"
)
# hvplot.save(fig_ar1_curves_deg_synth, "fig_ar1_curves_deg_synth.html")
