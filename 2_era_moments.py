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


xr.set_options(use_new_combine_kwarg_defaults=True)

# hvplot.extension("matplotlib")
# hvplot.extension("bokeh")

rdbu_discrete = tastymap.cook_tmap("RdYlBu_r", num_colors=12).cmap
reds_discrete = tastymap.cook_tmap("cet_CET_L18", num_colors=11)[
    1:10
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

use_calendar_summer = True  # if true, use JJA as summer. else use dayofyear mask
if use_calendar_summer:
    hw_all = (
        xr.open_dataset("era_hw_metrics_1950_2021_anom.nc")
        .sel(percentile=0.9, definition="3-0-0")
        .drop_vars(["percentile", "definition"])
    )
else:
    hw_all = (
        xr.open_dataset("era_hw_metrics_1950_2021_anom_doy.nc")
        .sel(percentile=0.9, definition="3-0-0")
        .drop_vars(["percentile", "definition"])
    )

# compute deltas
hw_old = hw_all.sel(time=slice("1950", "1985"))
hw_new = hw_all.sel(time=slice("1986", "2021"))
hw_mean_diff = hw_new.mean(dim="time") - hw_old.mean(dim="time")


#######################################################################
# Calculate mean differences (1986-2021) - (1950-1985) for temperature
#######################################################################

# anomalies relative to 1950-1985, calculated in 0_era_medianshift.py
era_anom_path = "era_land_anom.nc"
era_land_anom = xr.open_dataset(era_anom_path)

# compute deltas-------------------------------------------
era_land_old = era_land_anom.sel(time=slice("1950", "1985"))
era_land_new = era_land_anom.sel(time=slice("1986", "2021"))
tmax_mean_diff = (era_land_new.mean(dim="time") - era_land_old.mean(dim="time")).rename(
    {"t2m_x": "t2m_x_mean_diff"}
)


##############################################
# Calculate climatological (1950-1985) moments
# NOTE! these are moments of the *doy anomalies* wrt to (1950-1985), i.e. mean 0 over this period
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
        xlabel="tmax mean(1986:2021) - mean(1950:1985)\nanomalies wrt 1960-1985 (C)",
        ylabel=f"sample {y_name_label} over 1960-1985",
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
        xlabel="tmax mean(1986:2021) - mean(1950:1985)\nanomalies wrt 1960-1985 (C)",
        ylabel=f"sample {y_name_label} over 1960-1985",
        clabel="heatwave frequency (days)\nmean(1986:2021) - mean(1950:1985)",
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
        xlabel="tmax mean(1986:2021) - mean(1950:1985)\nanomalies wrt 1960-1985 (C)",
        ylabel=f"sample {y_name_label} over 1960-1985",
        clabel="heatwave duration (days)\nmean(1986:2021) - mean(1950:1985)",
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
        xlabel="tmax mean(1986:2021) - mean(1950:1985)\nanomalies wrt 1960-1985 (C)",
        ylabel=f"sample {y_name_label} over 1960-1985",
        clabel="heatwave avg intensity (degC anom)\nmean(1986:2021) - mean(1950:1985)",
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
        xlabel="tmax mean(1986:2021) - mean(1950:1985)\nanomalies wrt 1960-1985 (C)",
        ylabel=f"sample {y_name_label} over 1960-1985",
        clabel="heatwave cumulative intensity (degC anom)\nmean(1986:2021) - mean(1950:1985)",
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
    title="climatological skewness (1960:1985)",
    clabel="skewness",
).opts(fontscale=1.5)  # .opts(width=600, height=400)#
figlist_skewness = get_hexplots("t2m_x_skew", "skewness")
fig_skewness_count = figlist_skewness[0]
fig_layout_skewness = hv.Layout(figlist_skewness[1:]).cols(2)
fig_layout_skewness

# size_opts = dict(width=700, height=400)
# fig_layout_skewness.opts(opts.HexTiles(**size_opts))

# hvplot.save(fig_layout_skewness, "fit_skewness_hwf.html")


### variance ----------------------------------------

combined_ds["t2m_x_var"].hvplot(
    projection=ccrs.PlateCarree(),
    coastline=True,
    cmap=reds_discrete,
    # clim=(0, 30),
    title="climatological variance (1950:1985)",
    clabel="variance (degC^2)",
).opts(fontscale=1.5)

figlist_var = get_hexplots("t2m_x_var", "variance")
fig_var_count = figlist_var[0].opts(xlim=(-1, 3), width=600, height=400)

fig_layout_var = hv.Layout(figlist_var[1:]).cols(2)
fig_layout_var
# hvplot.save(fig_layout_var, "fig_hex_var.html")

### ar1 ----------------------------------------
combined_ds["t2m_x_ar1"].hvplot(
    projection=ccrs.PlateCarree(),
    coastline=True,
    cmap=reds_discrete,
    # clim=(0, 30),
    title="climatological AR(1) (1950:1985)",
    clabel="AR(1)",
).opts(fontscale=1.5)


figlist_ar1 = get_hexplots("t2m_x_ar1", "AR(1)")
fig_ar1_count = figlist_ar1[0].opts(width=600, height=400)

fig_layout_ar1 = hv.Layout(figlist_ar1[1:]).cols(2)
# hvplot.save(fig_layout_ar1, "fig_hex_ar1.html")


##########################################
# if the above has too many points
# let's combine points into quantile bins
#########################################

combined_df["tmax_bins"] = pd.qcut(combined_df["t2m_x_mean_diff"], q=10, precision=1)
combined_df["skew_bins"] = pd.qcut(combined_df["t2m_x_skew"], q=10, precision=1)

# see how many locations are in each bin

# combined_df.reset_index()[['lon', 'lat']].value_counts() # there are 14728 land gridcells
combined_df = combined_df.sort_values(by=["tmax_bins", "skew_bins"])
combined_df["tmax_bins"] = combined_df["tmax_bins"].astype(str)
combined_df["skew_bins"] = combined_df["skew_bins"].astype(str)
# combined_df["skew_bins"] = combined_df["skew_bins"].replace(
#     "(-1.2000000000000002, -0.3]", "(-1.2, -0.3]"
# )
combined_df.hvplot.heatmap(
    x="tmax_bins",
    y="skew_bins",
    C="t2m_x.t2m_x_threshold.HWF",  # in this plot, this isn't doing anything
    reduce_function=np.size,
    fields={
        "t2m_x.t2m_x_threshold.HWF": "count"
    },  # the tooltip displays count instead of hwf
    cmap=reds_discrete,
).opts()


# look at mean heatwave
combined_df.hvplot.heatmap(
    x="tmax_bins",
    y="skew_bins",
    C="t2m_x.t2m_x_threshold.HWF",
    reduce_function=np.mean,
    cmap=reds_discrete,
).opts()
