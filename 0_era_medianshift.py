# script to calculate summer heatwave metrics from ERA

###### notable choices ##############################################

# era daily tmax is preprocessed to remove day-of-year climatology
#   by smoothing the 1950-1985 doy-means using 5 fourier basis functions

# heatwaves are defined as 3+ consecutive days where the tmax anomalies exceed Q90
#   q90 is location and doy-specific, and are defined via a 7 day rolling window around each doy
#   across the years 1950-1985.
#
# because these thresholds are (location, doy)-specific, they can be thought of as their own type of doy climatology
#     So... these thresholds are also smoothed using 5 fourier basis functions.
################################################################################

import numpy as np
import xarray as xr
import glob
import regionmask
import dask
import hdp  # pip install -e "E:\\Projects\\HDP\\"

# from dask.distributed import Client, LocalCluster
dask.config.set(scheduler="single-threaded")


xr.set_options(use_new_combine_kwarg_defaults=True)


#########################################################
#### DECISION! is summer defined via calendar year (JJA?)
### change the flag below

# if false, summer is defined via 90 days surrounding the hottest day (for each latitude)
#########################################################
use_calendar_summer = True  # change this to True if you want to use JJA

if use_calendar_summer:
    suffix = ""
else:
    # day of year summer mask
    era_summer_path = "D:data\\ERA5\\ERA5_hottest_doys_t2m_x.nc"
    era_summer_mask = xr.open_dataarray(era_summer_path)
    suffix = "_doy"


###############################################
## helper functions
###############################################


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
        empirical_sc = da.copy().transpose(dim_names[0], "lat", "lon")
        mu = empirical_sc.mean(dim=dim_names[0])

    # nt, nlat, nlon = empirical_sc.shape
    nlat = da.lat.size
    nlon = da.lon.size
    loc_len = nlat * nlon

    # project zero-mean data onto basis functions
    data = (empirical_sc - mu).data

    # data must be in (time, lat, lon) order!
    coeff = 2 / n_time * (np.dot(bases, data.reshape((n_time, loc_len))))

    # reconstruct seasonal cycle
    rec = np.real(np.dot(bases.T, np.conj(coeff)))
    rec = rec.reshape((n_time, nlat, nlon))

    # add back the mean
    da_rec = empirical_sc.copy(data=rec) + mu
    return da_rec


#######################################
# read in ERA data
#######################################

era_filelist = glob.glob("D:data\\ERA5\\t2m_x_1x1\\*.nc")
era = xr.open_mfdataset(era_filelist).drop_vars("expver")

# fixing formatting for the hdp package
era["t2m_x"].attrs = {"units": "K"}  # hdp package needs units
# era = era.convert_calendar(calendar="standard", use_cftime=True)  # .compute()
era = era.convert_calendar(calendar="noleap", use_cftime=True)
era = era.sel(lat=slice(-60, 80)).chunk(
    {"time": -1, "lat": 10, "lon": 10}
)  # matching karen's doy mask

# convert to (-180, 180) lon. specific to our use case
era = era.assign_coords(lon=(((era.lon + 180) % 360) - 180)).sortby("lon")


# add landmask
era_land = add_landmask(era).compute()


#################################################################
# following russo et al,
# include 2 months before and after JJA,
# # then identify heatwaves,
## for each heatwave, count the proportion of days the heatwave has in JJA.
## if this proportion is more than 50%, then keep the heatwave as a "JJA" heatwave
#################################################################################


## identify heatwaves: ------------------------------------------

# 3 consecutive days of temperatures above T90_d, calculated as follows:
# for each day d in MJJAS
# pull out a 30 day window centered at d, using years 1961-1990 (i.e. shoould have 30*30 days included)
# calcuate q90 of daily max temperature -> T90_d
# check if days (d, d+1, d+2) all have higher temperature than T90_d
# if so, then keep counting consecutive days that have higher temperature. This is a heatwave event.


################################################
# calculate extremal metrics at each gridcell
###############################################

# defining the location-specific heat threshold ---------------

# using reference period 1950-1985
era_land_ref = era_land.sel(time=slice("1950", "1985"))

#### Note! we're taking anomalies with respect to the REFERENCE time period
# ref_doy_climatology = era_land_ref.groupby("time.dayofyear").mean()
ref_doy_climatology = fourier_climatology_smoother(
    era_land_ref["t2m_x"], n_time=365, n_bases=5
)
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


#### calculate doy thresholds, and smooth ----------------------
thresholds_ref_unsmooth = hdp.threshold.compute_thresholds(
    measures_ref, percentiles, rolling_window_size=7
).compute()

## smoothing out the the threshold climatology as well
thresholds_ref_smoothed = fourier_climatology_smoother(
    thresholds_ref_unsmooth["t2m_x_threshold"]
    .sel(percentile=percentiles[0])
    .drop_vars("percentile"),
    n_time=365,
    n_bases=5,
)
# match the formatting of the original hdp function -----
thresholds_ref = (
    thresholds_ref_smoothed.to_dataset()
    .expand_dims(percentile=percentiles)
    .transpose("lat", "lon", "doy", "percentile")
)
thresholds_ref["t2m_x_threshold"].attrs["baseline_variable"] = "t2m_x"
thresholds_ref["t2m_x_threshold"].attrs["hdp_type"] = "threshold"
thresholds_ref["t2m_x_threshold"].attrs["baseline_calendar"] = "noleap"

# era_tmax_ref_summer_masked_nh = measures_ref.sel(time=measures_ref['time.month'].isin([6, 7, 8])).sel(lat = slice(0, None))
# era_tmax_ref_summer_masked_sh = measures_ref.sel(time=measures_ref['time.month'].isin([12, 1, 2])).sel(lat = slice(None, 0))
# era_tmax_ref_summer_masked = xr.concat([era_tmax_ref_summer_masked_nh,era_tmax_ref_summer_masked_sh], dim = ['lon', 'time'])

# era_tmax_ref_summer_masked = measures_ref.groupby("time.dayofyear").where(
#     era_summer_mask == 1
# )
# threshold_ref_summer_mask = era_tmax_ref_summer_masked.quantile(0.9, dim="time")
# #### match the output of hdp.threshold.compute_threshold
# thresholds_ref = (
#     threshold_ref_summer_mask.expand_dims(doy=era_summer_mask.dayofyear.values)
#     .expand_dims(percentile=percentiles)
#     .rename({"t2m_x": "t2m_x_threshold"})
#     .transpose("lat", "lon", "doy", "percentile")
# ).compute()
# thresholds_ref["t2m_x_threshold"].attrs["baseline_variable"] = "t2m_x"
# thresholds_ref["t2m_x_threshold"].attrs["hdp_type"] = "threshold"
# thresholds_ref["t2m_x_threshold"].attrs["baseline_calendar"] = "noleap"
# thresholds_ref = thresholds_ref.drop_vars("quantile")


definitions = [[3, 0, 0]]  # heatwave is 3 consec days


# --------------------------------------------------

# time period 1, 1950-1985, following russo and domeisen 2023 ---------------------------
era_land_old = era_land.sel(time=slice("1950", "1985"))
# time period 2, 1986-2021, following russo and domeisen 2023 ---------------------------
era_land_new = era_land.sel(time=slice("1986", "2021"))


###########
# synthetic, medianshift
#############

# shifting the median for each grid cell
old_medians = era_land_old["t2m_x"].median(dim=["time"])
new_medians = era_land_new["t2m_x"].median(dim=["time"])

era_land_synth_new = era_land_old - old_medians + new_medians
era_land_synth_new = era_land_synth_new.assign_coords(
    time=era_land_new.time
)  # pretend its the new time period


# combine back. this is comparable to era_land_all below
era_land_synth = xr.concat([era_land_old, era_land_synth_new], dim="time")
era_land_synth_anom = (
    era_land_synth.groupby("time.dayofyear") - ref_doy_climatology
).drop_vars("dayofyear")
era_land_synth_anom["t2m_x"].attrs = {
    "units": "C"
}  # hdp package needs units, and anom are same in K or C

# calculate heatwave metrics
measures_synth = hdp.measure.format_standard_measures(
    temp_datasets=[era_land_synth_anom["t2m_x"]]
).chunk({"time": -1, "lat": 10, "lon": 10})
thresholds_ref = thresholds_ref.chunk({"lat": 10, "lon": 10})

if use_calendar_summer:
    metrics_dataset_synth = hdp.metric.compute_group_metrics(
        measures_synth,
        thresholds_ref,
        definitions,
        use_doy=False,
        start=(6, 1),
        end=(9, 1),
    )
else:
    metrics_dataset_synth = hdp.metric.compute_group_metrics(
        measures_synth,
        thresholds_ref,
        definitions,
        use_doy=True,
        doy_mask=era_summer_mask,
    )

# in these intensity metrics (AVI and AVA, zeros mean that there were no heatwaves that year)
# so let's turn those 0s to nans
metrics_dataset_synth["t2m_x.t2m_x_threshold.AVI"] = metrics_dataset_synth[
    "t2m_x.t2m_x_threshold.AVI"
].where(metrics_dataset_synth["t2m_x.t2m_x_threshold.AVI"] != 0)

metrics_dataset_synth["t2m_x.t2m_x_threshold.AVA"] = metrics_dataset_synth[
    "t2m_x.t2m_x_threshold.AVA"
].where(metrics_dataset_synth["t2m_x.t2m_x_threshold.AVA"] != 0)

# compute cumulative heat
metrics_dataset_synth["t2m_x.t2m_x_threshold.sumHeat"] = (
    metrics_dataset_synth["t2m_x.t2m_x_threshold.AVA"]
    * metrics_dataset_synth["t2m_x.t2m_x_threshold.HWF"]
)

metrics_synth_land = add_landmask(metrics_dataset_synth)
# metrics_synth_land.to_netcdf(f"era_hw_metrics_1950_2021_synth_anom{suffix}.nc")

############################
# observations (not synthetic)
# time period ALL 1950-2021, to match russo (but with slighty different threshold period--------------------------------
##############################
era_land_all = era_land.sel(time=slice("1950", "2021"))
era_land_all_anom = (
    era_land_all.groupby("time.dayofyear") - ref_doy_climatology
).drop_vars("dayofyear")

# save this for future
# era_land_all_anom.to_netcdf("era_land_anom.nc")


era_land_all_anom["t2m_x"].attrs = {"units": "C"}  # hdp package needs units
# use c when dealing with anomalies (bc anomalies are the same in either units)

# calculate heatwave metrics on time period.
measures_all = hdp.measure.format_standard_measures(
    temp_datasets=[era_land_all_anom["t2m_x"]]
).chunk({"time": -1, "lat": 10, "lon": 10})

if use_calendar_summer:
    metrics_dataset_all = hdp.metric.compute_group_metrics(
        measures_all, thresholds_ref, definitions, start=(6, 1), end=(9, 1)
    )
else:
    metrics_dataset_all = hdp.metric.compute_group_metrics(
        measures_all,
        thresholds_ref,
        definitions,
        use_doy=True,
        doy_mask=era_summer_mask,
    )

# turn 0s in AVI to nan
metrics_dataset_all["t2m_x.t2m_x_threshold.AVI"] = metrics_dataset_all[
    "t2m_x.t2m_x_threshold.AVI"
].where(metrics_dataset_all["t2m_x.t2m_x_threshold.AVI"] != 0)

metrics_dataset_all["t2m_x.t2m_x_threshold.AVA"] = metrics_dataset_all[
    "t2m_x.t2m_x_threshold.AVA"
].where(metrics_dataset_all["t2m_x.t2m_x_threshold.AVA"] != 0)

# compute cumulative heats
metrics_dataset_all["t2m_x.t2m_x_threshold.sumHeat"] = (
    metrics_dataset_all["t2m_x.t2m_x_threshold.AVA"]
    * metrics_dataset_all["t2m_x.t2m_x_threshold.HWF"]
)

metrics_all_land = add_landmask(metrics_dataset_all)
# metrics_all_land.to_netcdf(f"era_hw_metrics_1950_2021_anom{suffix}.nc")
