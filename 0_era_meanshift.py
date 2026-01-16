# script to calculate summer heatwave metrics from ERA

###### notable choices ########################asdf######################

# era daily tmax is preprocessed to remove day-of-year climatology
#   by smoothing the 1960-2025 doy-means using 5 fourier basis functions

# heatwaves are defined as 3+ consecutive days where the tmax anomalies exceed Q90
#   q90 is location and doy-specific, and are defined via a 7 day rolling window around each doy
#   across the years 1960-2025.

# because these thresholds are (location, doy)-specific, they can be thought of as their own type of doy climatology
#     So... these thresholds are also smoothed using 5 fourier basis functions.

# this script makes a non-standard choice in the way that all "climatological" quantities are calculated
#    (i.e. doy-climatology, q90, skewness and variance)
#    the standard choice is to calculate these quantities over a reference period (e.g. 1960-1990),
#    which minimizes the influence of human-driven climate change on our estimates of the climatology
#
# we wanted to maximize the amount of data available to estimate quantities like skewness, so we instead:
#    use the full time period 1960-2025 to estimate climatological quantities
#    but remove a year-specific mean prior to this estimation.
#  e.g the raw tmax time series in 1960 has the 1960 mean removed. the raw tmax time series in 1961 has the 1961 mean removed
#  This has the effect of removing the climate change signal while still giving us access to modern years.
#  And we use this time series without the yearly means to then define the doy-climatology, q90, variance, and skewness

# To apply these climatological quantities in our analysis of the change in heatwave statistics (which SHOULD include climate change),
#    we shift the entire raw tmax daily time series to be mean zero, and then remove the previously defined doy-climatology, and so on.
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
    # ref_years = [1950, 1987]  # reference period (used for thresholds)
    # new_years = [1988, 2025]  # used for synthetic example
    ref_years = [1960, 1990]
    new_years = [1995, 2025]
else:
    # day of year summer mask
    era_summer_path = "D:data\\ERA5\\ERA5_hottest_doys_t2m_x.nc"
    era_summer_mask = xr.open_dataarray(era_summer_path)
    suffix = "_doy"
    # ref_years = [1950, 1987]  # reference period (used for thresholds)
    ref_years = [1960, 1990]
    new_years = [1996, 2025]


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

    # also remove central africa bc of data quality concerns in the 1960s
    central_africa = regionmask.defined_regions.srex[["W. Africa", "E. Africa"]]
    central_africa_mask = central_africa.mask(ds)
    is_not_central_africa = central_africa_mask.isnull()

    # apply landmask
    ds = ds.where(is_land & is_not_greenland & is_not_antarctic & is_not_central_africa)

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


def process_heatwave_metrics(da_metrics):
    """
    - calculate cumulative heat
    - convert average intensity to a metric which excludes zeros
    - add a landmask
    :param: da_metrics is the output of hdp's compute_group_metrics()
    returns: a version of da_metrics with added variables, but same coordinates
    """

    ## NOTE! ordering matters! It's important that we calculate sumHeat before changing turning AVA 0 -> nan
    ## sumHEAT which combines frequency with intensity, so it makes more sense for sumHeat to be 0

    ## toy example for mental model:
    ##     eg_hwf = [0,0,3,4,3,0] # say the first 3 values are the ref period, and the last 3 is the new period
    ##     eg_ava = [0,0,10,20,5, 0] OR a_ava = [nan, nan, 10, 20, 5, nan]
    ##     a_cumheat = [0,0,30,80,15,0] OR a_cumheat = [nan, nan, 30, 80, 15, nan]
    ##     a_diff = 95/3 - 10 =~ 21.7, OR a_diff = 65/2 - 30 =~ 2.5

    # compute cumulative heat
    da_metrics["t2m_x.t2m_x_threshold.sumHeat"] = (
        da_metrics["t2m_x.t2m_x_threshold.AVA"]
        * da_metrics["t2m_x.t2m_x_threshold.HWF"]
    )

    # in these intensity metrics (AVI and AVA, zeros mean that there were no heatwaves that year)
    # so let's turn those 0s to nans
    # this would be useful, if we wanted AVI to measure "average intensity of a day | day was a heatwave day"
    # This reduces the dependence of AVI on heatwave frequency.
    da_metrics["t2m_x.t2m_x_threshold.AVI"] = da_metrics[
        "t2m_x.t2m_x_threshold.AVI"
    ].where(da_metrics["t2m_x.t2m_x_threshold.AVI"] != 0)

    da_metrics["t2m_x.t2m_x_threshold.AVA"] = da_metrics[
        "t2m_x.t2m_x_threshold.AVA"
    ].where(da_metrics["t2m_x.t2m_x_threshold.AVA"] != 0)

    metrics_synth_land = add_landmask(da_metrics)
    return metrics_synth_land


def get_synthetic(da_ref, da_synth_new, start_year_new, use_calendar_summer=True):
    synth_time = xr.date_range(
        start=str(start_year_new),
        periods=da_ref.time.size,
        freq="D",
        calendar="noleap",
        use_cftime=True,
    )
    era_land_synth_new = da_synth_new.assign_coords(
        time=synth_time
        # time=era_land_new.time
    )  # pretend its the new time period

    # combine back. this is comparable to era_land_all_anom above, except with a small gap in the middle (1990-95)
    era_land_synth_anom = xr.concat([da_ref, era_land_synth_new], dim="time")

    era_land_synth_anom["t2m_x"].attrs = {
        "units": "C"
    }  # hdp package needs units, and anom are same in K or C

    # calculate heatwave metrics
    measures_synth = hdp.measure.format_standard_measures(
        temp_datasets=[era_land_synth_anom["t2m_x"]]
    ).chunk({"time": -1, "lat": 10, "lon": 10})

    # if there isnt a gap between the 2 periods, then you can just pass measures_synth to compute_group_metrics
    # but if there's a gap, then need to split into old and new
    measures_synth_old = measures_synth.sel(
        time=slice(str(ref_years[0] - 1), str(ref_years[1]))
    )
    measures_synth_new = measures_synth.sel(
        time=slice(str(new_years[0] - 1), str(new_years[1]))
    )

    if use_calendar_summer:
        metrics_synth_old = hdp.metric.compute_group_metrics(
            measures_synth_old,
            thresholds_ref,
            definitions,
            use_doy=False,
            start=(6, 1),
            end=(9, 1),
        )
        metrics_synth_new = hdp.metric.compute_group_metrics(
            measures_synth_new,
            thresholds_ref,
            definitions,
            use_doy=False,
            start=(6, 1),
            end=(9, 1),
        )
        metrics_dataset_synth = xr.concat(
            [metrics_synth_old, metrics_synth_new], dim="time"
        )

    else:
        metrics_synth_old = hdp.metric.compute_group_metrics(
            measures_synth_old,
            thresholds_ref,
            definitions,
            use_doy=True,
            doy_mask=era_summer_mask,
        )
        metrics_synth_new = hdp.metric.compute_group_metrics(
            measures_synth_new,
            thresholds_ref,
            definitions,
            use_doy=True,
            doy_mask=era_summer_mask,
        )
        metrics_dataset_synth = xr.concat(
            [metrics_synth_old, metrics_synth_new], dim="time"
        )

    metrics_synth_land = process_heatwave_metrics(metrics_dataset_synth)

    return metrics_synth_land


#######################################
# read in ERA data
#######################################

era_filelist = glob.glob("D:data\\ERA5\\t2m_x_1x1\\*.nc")
# era = xr.open_mfdataset(era_filelist).drop_vars("expver")
era = (
    xr.open_mfdataset(era_filelist)
    .rename({"__xarray_dataarray_variable__": "t2m_x", "valid_time": "time"})
    .reset_coords(names="number", drop=True)
)


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


##############################################
# Calculate climatological characteristics
# this will be used to calculate doy-climatology, q90,
# (and climatological variance and skewness, in 2_era_moments.py)
#############################################

# defining the location-specific heat threshold ---------------
# using time period 1960-2025

# # note! if using jja in nh and djf in southern hemisphere, then we should also include the year before in the ref period
# # bc djf 1950 requires december of 1949.
era_land_climatology_years = era_land.sel(
    time=slice(str(ref_years[0] - 1), str(new_years[1]))
)

# make the whole time series zero mean, across 1960-2025. apply separately to each gridcell
era_land_climatology_years_zeromean = (
    era_land_climatology_years - era_land_climatology_years.mean(dim="time")
)

# capture "global warming" at each grid cell by getting the mean at each year
era_land_yearly_mean = era_land_climatology_years_zeromean.groupby("time.year").mean()
era_land_no_yearly_mean = (
    era_land_climatology_years_zeromean.groupby("time.year") - era_land_yearly_mean
).reset_coords("year", drop=True)

#### Note! we're taking anomalies with respect to the entire time period
# ref_doy_climatology = era_land_ref.groupby("time.dayofyear").mean()
ref_doy_climatology = fourier_climatology_smoother(
    era_land_no_yearly_mean["t2m_x"], n_time=365, n_bases=5
)

# # note! if using jja in nh and djf in southern hemisphere, then we should also include the year before in the ref period
# # bc djf 1950 requires december of 1949.
# era_land_ref = era_land.sel(time=slice(str(ref_years[0] - 1), str(new_years[1])))

# take doy anomalies
era_land_ref_anom = (
    era_land_no_yearly_mean.groupby("time.dayofyear") - ref_doy_climatology
).drop_vars("dayofyear")
era_land_ref_anom["t2m_x"].attrs = {"units": "C"}  # hdp package needs units


# NOTE!
# this dataset (1960-2025 mean removed -> individual yearly means removed -> doy mean removed)
#   is the dataset that should be used to estimate "climatological" characteristics
# era_land_ref_anom.to_netcdf("era_land_anom_for_climatology.nc")


#################################################################
#### calculate doy thresholds, and smooth the threshold ---------
##################################################################

# conversion to celcius
measures_ref = hdp.measure.format_standard_measures(
    temp_datasets=[era_land_ref_anom["t2m_x"]]
)
percentiles = [0.9]

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
thresholds_ref = thresholds_ref.chunk({"lat": 10, "lon": 10})

# thresholds_ref.to_netcdf("thresholds_ref.nc")


################################################
# calculate extremal metrics at each gridcell
###############################################

# compared to era_land_ref_anom, the dataset we're using to calculate heatwave metrics "skips" the step where we subtract a separate mean for each year.
# so instead, we just have:
#   daily time series -> shifted by a scalar to have zero mean across 1960-2025 -> remove doy climatology
# where the doy climatology was calculated from the third step in:
#  daily time series -> shifted by a scalar to have zero mean across 1960-2025 -> shifted by a year-specific value to have zero mean each year -> remove doy climatology -> calculate q90 threshold

definitions = [[3, 0, 0]]  # heatwave is 3 consec days

############################
# observations (not synthetic)
# time period ALL 1950-2025,
##############################

# era_land_all = era_land.sel(time=slice(str(ref_years[0] - 1), str(new_years[1])))
era_land_all_anom = (
    era_land_climatology_years_zeromean.groupby("time.dayofyear") - ref_doy_climatology
).drop_vars("dayofyear")

# NOTE: save this for future -- this is is the data that's used to calculate heatwave metrics
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

metrics_all_land = process_heatwave_metrics(metrics_dataset_all)
# metrics_all_land.to_netcdf(
#     f"era_hw_metrics_{ref_years[0]}_{new_years[1]}_anom{suffix}.nc"
# )


###########
# synthetic, meanshift
#############

# note that we do this on the shifted, doy-anomaly removed ts.

# time period 2,
# era_land_new = era_land.sel(time=slice(str(new_years[0]), str(new_years[1])))
era_land_ref_anom = era_land_all_anom.sel(
    time=slice(str(ref_years[0] - 1), str(ref_years[1]))
)
era_land_new_anom = era_land_all_anom.sel(
    time=slice(str(new_years[0] - 1), str(new_years[1]))
)

## shifting the mean for each grid cell.
## note: this is in anomaly space!! shifted, doy-anomaly removed.
old_means = era_land_ref_anom["t2m_x"].mean(dim=["time"])
new_means = era_land_new_anom["t2m_x"].mean(dim=["time"])
# old_means = era_land_ref["t2m_x"].mean(dim=["time"])
# new_means = era_land_new["t2m_x"].mean(dim=["time"])

# update the "time" coordinate in the future to pretend it's the "future"
era_land_synth_new = (era_land_ref_anom - old_means) + new_means

metrics_synth_land = get_synthetic(
    era_land_ref_anom,
    era_land_synth_new,
    start_year_new=new_years[0] - 1,
    use_calendar_summer=True,
)

# metrics_synth_land.to_netcdf(
#     f"era_hw_metrics_{ref_years[0]}_{new_years[1]}_synth_anom{suffix}.nc"
# )


###########
# synthetic, 1deg shift
# i.e. the "future" time period (1995-2025) is the same as 1960-1990 but shifted by 1deg
#############

# update the "time" coordinate in the future to pretend it's the "future"
# shifted by 1 degree.
era_land_synth_new_1deg = era_land_ref_anom + 1

metrics_synth_land_1deg = get_synthetic(
    era_land_ref_anom,
    era_land_synth_new_1deg,
    start_year_new=new_years[0] - 1,
    use_calendar_summer=True,
)
# metrics_synth_land_1deg.to_netcdf(
#     f"era_hw_metrics_{ref_years[0]}_{new_years[1]}_synth_1deg_anom{suffix}.nc"
# )


###########
# synthetic, 2deg shift
#############

# update the "time" coordinate in the future to pretend it's the "future"
# shift by 2 degrees
era_land_synth_new_2deg = era_land_ref_anom + 2
metrics_synth_land_2deg = get_synthetic(
    era_land_ref_anom,
    era_land_synth_new_2deg,
    start_year_new=new_years[0] - 1,
    use_calendar_summer=True,
)
# metrics_synth_land_2deg.to_netcdf(
#     f"era_hw_metrics_{ref_years[0]}_{new_years[1]}_synth_2deg_anom{suffix}.nc"
# )
