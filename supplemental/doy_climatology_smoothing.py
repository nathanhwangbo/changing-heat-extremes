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
import glob
from xarray_einstats import stats  # wrapper around apply_ufunc for moments
import pandas as pd
import hvplot.pandas
import hdp

xr.set_options(use_new_combine_kwarg_defaults=True)

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


def fourier_climatology_smoother(da, n_time, n_bases=5):
    """
    taken from karen's code

    calculates a fourier-smoothed climatology at each gridcell, using n_bases components
    output is an xarray data array with climatologies, with dimension (n_time, lon, lat)

    da is a data array, with dimensions (time, lon, lat)
    n_time is 365 if removing the doy climatology or 12 if removing the monthly climatology
    nbases is the number of fourier components we want to use
    """
    # create basis functions to remove seasonal cycle
    time = np.arange(1, n_time + 1)
    t_basis = time / n_time

    # list of the first n_bases fourier components
    bases = np.empty((n_bases, n_time), dtype=complex)
    for counter in range(n_bases):
        bases[counter, :] = np.exp(2 * (counter + 1) * np.pi * 1j * t_basis)

    if "time" in list(da.coords):
        if n_time == 365:
            # get empirical average for the doy
            empirical_sc = da.groupby("time.dayofyear").mean()  # dim (doy, lat, lon)
            mu = empirical_sc.mean(
                dim="dayofyear"
            )  # map of average across all days. dim (lat, lon)
        elif n_time == 12:
            # get empirical average for the month
            empirical_sc = da.groupby("time.month").mean()  # dim (month, lat, lon)
            mu = empirical_sc.mean(
                dim="month"
            )  # map of average across all days. dim (lat, lon)
        else:
            raise ValueError("only n_time = 12 or 365 are handled")
    # if da is pre-averaged and has dimension name dim_name (i.e. "doy" or "month")
    # i.e. da is already equiv to empirical_sc
    else:
        dim_names = [dim for dim in list(da.coords) if dim not in ["lat", "lon"]]
        if len(dim_names) != 1:
            raise ValueError(
                "You have the wrong number of coordinates. There should only be three dimensions: (lat, lon, and some time variable)"
            )
        empirical_sc = da.copy()
        mu = empirical_sc.mean(dim=dim_names[0])

    nt, nlat, nlon = empirical_sc.shape
    loc_len = nlat * nlon

    # project zero-mean data onto basis functions
    data = (empirical_sc - mu).data

    coeff = 2 / n_time * (np.dot(bases, data.reshape((nt, loc_len))))

    # reconstruct seasonal cycle
    rec = np.real(np.dot(bases.T, np.conj(coeff)))
    rec = rec.reshape((nt, nlat, nlon))

    # add back the mean
    da_rec = empirical_sc.copy(data=rec) + mu
    return da_rec


#######################################################################
# Calculate mean differences (1986-2021) - (1950-1985) for temperature
#######################################################################

era_filelist = glob.glob("D:data\\ERA5\\t2m_x_1x1\\*.nc")
era = xr.open_mfdataset(era_filelist).drop_vars("expver")

era = era.convert_calendar(calendar="noleap", use_cftime=True)
era = era.sel(lat=slice(-60, 80))  # matching karen's doy mask

era_land = add_landmask(era).compute()

##### calculate anomalies relative to 1950-1985 #######
era_land_ref = era_land.sel(time=slice("1950", "1985"))
# Note! we're taking anomalies with respect to the REFERENCE time period
ref_doy_climatology = era_land_ref.groupby("time.dayofyear").mean()

# pull out a few test cases


def get_doy_climatology_casestudy(da, label="", ylab=""):
    """
    da is an xarray data array, with dimensions (dayofyear, lat, lon)
    lon is assumed to be in (0, 360)
    """
    # test_lon = -60
    # test_lon = np.mod(test_lon, 360)
    lon_sw = 245
    lat_sw = 35
    test_sw = da.sel(lat=lat_sw, lon=lon_sw, method="nearest")
    fig_sw = test_sw.hvplot(
        title=f"north american southwest (lonlat is ({lon_sw}, {lat_sw}))",
        label=label,
        legend="right",
        ylabel=ylab,
    )

    # cambodia
    lon_khm = 104
    lat_khm = 11
    tmax_clim_khm = da.sel(lat=lat_khm, lon=lon_khm, method="nearest")
    fig_khm = tmax_clim_khm.hvplot(
        title=f"cambodia  (lonlat is ({lon_khm}, {lat_khm}))",
        label=label,
        legend="right",
        ylabel=ylab,
    )

    # norway
    lon_nor = 10
    lat_nor = 59
    tmax_clim_nor = da.sel(lat=lat_nor, lon=lon_nor, method="nearest")
    fig_nor = tmax_clim_nor.hvplot(
        title=f"norway (lonlat is ({lon_nor}, {lat_nor}))",
        label=label,
        legend="right",
        ylabel=ylab,
    )

    # bolivia
    lon_bol = 300
    lat_bol = -17
    tmax_clim_bol = da.sel(lat=lat_bol, lon=lon_bol, method="nearest")
    fig_bol = tmax_clim_bol.hvplot(
        title=f"bolivia  (lonlat is ({lon_bol}, {lat_bol}))",
        label=label,
        legend="right",
        ylabel=ylab,
    )

    fig_tmax_climatology_baseline = (fig_sw + fig_khm + fig_nor + fig_bol).cols(2)
    return fig_tmax_climatology_baseline


fig_tmax_climatology_baseline = get_doy_climatology_casestudy(
    ref_doy_climatology["t2m_x"]
)
hvplot.save(fig_tmax_climatology_baseline, "fig_tmax_climatology_casestudy.html")


### compare with smoothed climatology -----------------------

ref_doy_climatology = fourier_climatology_smoother(
    era_land_ref["t2m_x"], n_time=365, n_bases=5
)
fig_tmax_climatology_smoothed = get_doy_climatology_casestudy(ref_doy_climatology)

figlist_tmax_climatology_compare = []
for i in range(len(fig_tmax_climatology_baseline)):
    figlist_tmax_climatology_compare.append(
        fig_tmax_climatology_baseline[i] * fig_tmax_climatology_smoothed[i]
    )

fig_tmax_climatology_compare = hv.Layout(figlist_tmax_climatology_compare).cols(2)
hvplot.save(fig_tmax_climatology_compare, "fig_tmax_climatology_casestudy.html")
############################################


# take anomalies
era_land_ref_anom = (
    era_land_ref.groupby("time.dayofyear") - ref_doy_climatology
).drop_vars("dayofyear")
era_land_ref_anom["t2m_x"].attrs = {"units": "C"}  # hdp package needs units

# conversion to celcius
measures_ref = hdp.measure.format_standard_measures(
    temp_datasets=[era_land_ref_anom["t2m_x"]]
)
percentiles = [0.9]


# try 3 different windows
thresholds_ref7 = (
    hdp.threshold.compute_thresholds(measures_ref, percentiles, rolling_window_size=7)[
        "t2m_x_threshold"
    ]
    .isel(percentile=0)
    .transpose("doy", "lat", "lon")
    .compute()
)

thresholds_ref15 = (
    hdp.threshold.compute_thresholds(measures_ref, percentiles, rolling_window_size=15)[
        "t2m_x_threshold"
    ]
    .isel(percentile=0)
    .transpose("doy", "lat", "lon")
    .compute()
)

thresholds_ref30 = (
    hdp.threshold.compute_thresholds(measures_ref, percentiles, rolling_window_size=30)[
        "t2m_x_threshold"
    ]
    .isel(percentile=0)
    .transpose("doy", "lat", "lon")
    .compute()
)


# take a look
fig_q90_baseline7 = get_doy_climatology_casestudy(
    thresholds_ref7,
    label="7day",
    ylab="q90 threshold (anom C)",
)
fig_q90_baseline15 = get_doy_climatology_casestudy(
    thresholds_ref15,
    label="15day",
    ylab="q90 threshold (anom C)",
)
fig_q90_baseline30 = get_doy_climatology_casestudy(
    thresholds_ref30,
    label="30day",
    ylab="q90 threshold (anom C)",
)

figlist_q90_climatology_compare = []
for i in range(len(fig_q90_baseline7)):
    figlist_q90_climatology_compare.append(
        fig_q90_baseline7[i] * fig_q90_baseline15[i] * fig_q90_baseline30[i]
    )

fig_q90_climatology_compare = hv.Layout(figlist_q90_climatology_compare).cols(2)
hvplot.save(fig_q90_climatology_compare, "fig_q90_climatology_casestudy.html")


#### now try smoothing all the thresholds and see how they look
q90_smoothed_climatology7 = fourier_climatology_smoother(
    thresholds_ref7, n_time=365, n_bases=5, is_time_dim=False
)
q90_smoothed_climatology15 = fourier_climatology_smoother(
    thresholds_ref15, n_time=365, n_bases=5, is_time_dim=False
)
q90_smoothed_climatology30 = fourier_climatology_smoother(
    thresholds_ref30, n_time=365, n_bases=5, is_time_dim=False
)
# take a look
fig_q90_baseline7_smooth = get_doy_climatology_casestudy(
    q90_smoothed_climatology7,
    label="7day",
    ylab="q90 threshold (anom C)",
)
fig_q90_baseline15_smooth = get_doy_climatology_casestudy(
    q90_smoothed_climatology15,
    label="15day",
    ylab="q90 threshold (anom C)",
)
fig_q90_baseline30_smooth = get_doy_climatology_casestudy(
    q90_smoothed_climatology30,
    label="30day",
    ylab="q90 threshold (anom C)",
)

figlist_q90_climatology_compare_smooth = []
for i in range(len(fig_q90_baseline7_smooth)):
    figlist_q90_climatology_compare_smooth.append(
        fig_q90_baseline7_smooth[i]
        * fig_q90_baseline15_smooth[i]
        * fig_q90_baseline30_smooth[i]
    )

fig_q90_climatology_compare_smooth = hv.Layout(
    figlist_q90_climatology_compare_smooth
).cols(2)
hvplot.save(
    fig_q90_climatology_compare_smooth, "fig_q90_climatology_smooth_casestudy.html"
)


# out of curiosity, look at the variance of tmax -----------------
# does this explain the fluctuations in the q90 threshold?
var_climatology = measures_ref["t2m_x"].groupby("time.dayofyear").var(dim="time")
fig_var_climatology = get_doy_climatology_casestudy(
    a,
    ylab="var (tmax anom C)",
)
hvplot.save(fig_var_climatology, "fig_var_climatology.html")
