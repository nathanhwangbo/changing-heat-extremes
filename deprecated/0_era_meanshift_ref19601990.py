##################################################
# DEPRECATED!!!
#  in favor of 0_era_meanshift.py, which doesn't use a reference period.
##############################################3


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


#######################################
# read in ERA data
#######################################

# era_filelist = glob.glob("D:data\\ERA5\\t2m_x_1x1\\*.nc")
# era = xr.open_mfdataset(era_filelist).drop_vars("expver")
era_filelist = glob.glob("D:data\\ERA5\\t2m_x_1x1_2025\\*.nc")
era = xr.open_mfdataset(era_filelist).rename({"t2m": "t2m_x"})


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


################################################
# calculate extremal metrics at each gridcell
###############################################

# defining the location-specific heat threshold ---------------
# using reference period 1950-1987
era_land_climatology_years = era_land.sel(
    time=slice(str(ref_years[0]), str(ref_years[1]))
)

#### Note! we're taking anomalies with respect to the REFERENCE time period
# ref_doy_climatology = era_land_ref.groupby("time.dayofyear").mean()
ref_doy_climatology = fourier_climatology_smoother(
    era_land_climatology_years["t2m_x"], n_time=365, n_bases=5
)

# note! if using jja in nh and djf in southern hemisphere, then we should also include the year before in the ref period
# bc djf 1950 requires december of 1949.
era_land_ref = era_land.sel(time=slice(str(ref_years[0] - 1), str(ref_years[1])))
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
thresholds_ref = thresholds_ref.chunk({"lat": 10, "lon": 10})

# thresholds_ref.to_netcdf("thresholds_ref.nc")

definitions = [[3, 0, 0]]  # heatwave is 3 consec days


# --------------------------------------------------

############################
# observations (not synthetic)
# time period ALL 1950-2025, to match russo (but with slighty different threshold period--------------------------------
##############################
era_land_all = era_land.sel(time=slice(str(ref_years[0] - 1), str(new_years[1])))
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

# compute cumulative heat
metrics_dataset_all["t2m_x.t2m_x_threshold.sumHeat"] = (
    metrics_dataset_all["t2m_x.t2m_x_threshold.AVA"]
    * metrics_dataset_all["t2m_x.t2m_x_threshold.HWF"]
)

metrics_all_land = add_landmask(metrics_dataset_all)
# metrics_all_land.to_netcdf(f"era_hw_metrics_{ref_years[0]}_{new_years[1]}_anom{suffix}.nc")


###########
# synthetic, meanshift
#############

# time period 2,
era_land_new = era_land.sel(time=slice(str(new_years[0]), str(new_years[1])))

# shifting the mean for each grid cell
old_means = era_land_ref["t2m_x"].mean(dim=["time"])
new_means = era_land_new["t2m_x"].mean(dim=["time"])

# update the "time" coordinate in the future to pretend it's the "future"
era_land_synth_new = era_land_climatology_years - old_means + new_means
synth_time = xr.date_range(
    start=str(new_years[0]),
    periods=era_land_climatology_years.time.size,
    freq="D",
    calendar="noleap",
    use_cftime=True,
)
era_land_synth_new = era_land_synth_new.assign_coords(
    time=synth_time
    # time=era_land_new.time
)  # pretend its the new time period


# combine back. this is comparable to era_land_all below
era_land_synth = xr.concat([era_land_ref, era_land_synth_new], dim="time")
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

    # metrics_dataset_synth = hdp.metric.compute_group_metrics(
    #     measures_synth,
    #     thresholds_ref,
    #     definitions,
    #     use_doy=False,
    #     start=(6, 1),
    #     end=(9, 1),
    # )
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
# metrics_synth_land.to_netcdf(f"era_hw_metrics_{ref_years[0]}_{new_years[1]}_synth_anom{suffix}.nc")


###########
# synthetic, 1deg shift
#############

# update the "time" coordinate in the future to pretend it's the "future"
# shifted by 1 degree.
era_land_synth_new_1deg = era_land_climatology_years + 1
era_land_synth_new_1deg = era_land_synth_new_1deg.assign_coords(
    time=synth_time
    # time=era_land_new.time
)  # pretend its the new time period

# combine back. this is comparable to era_land_all below
era_land_synth_1deg = xr.concat([era_land_ref, era_land_synth_new_1deg], dim="time")
era_land_synth_anom_1deg = (
    era_land_synth_1deg.groupby("time.dayofyear") - ref_doy_climatology
).drop_vars("dayofyear")
era_land_synth_anom_1deg["t2m_x"].attrs = {
    "units": "C"
}  # hdp package needs units, and anom are same in K or C

# calculate heatwave metrics
measures_synth_1deg = hdp.measure.format_standard_measures(
    temp_datasets=[era_land_synth_anom_1deg["t2m_x"]]
).chunk({"time": -1, "lat": 10, "lon": 10})

# if there isnt a gap between the 2 periods, then you can just pass measures_synth to compute_group_metrics
# but if there's a gap, then need to split into old and new
measures_synth_old_1deg = measures_synth_1deg.sel(
    time=slice(str(ref_years[0] - 1), str(ref_years[1]))
)
measures_synth_new_1deg = measures_synth_1deg.sel(
    time=slice(str(new_years[0] - 1), str(new_years[1]))
)

if use_calendar_summer:
    metrics_synth_old_1deg = hdp.metric.compute_group_metrics(
        measures_synth_old_1deg,
        thresholds_ref,
        definitions,
        use_doy=False,
        start=(6, 1),
        end=(9, 1),
    )
    metrics_synth_new_1deg = hdp.metric.compute_group_metrics(
        measures_synth_new_1deg,
        thresholds_ref,
        definitions,
        use_doy=False,
        start=(6, 1),
        end=(9, 1),
    )
    metrics_dataset_synth_1deg = xr.concat(
        [metrics_synth_old_1deg, metrics_synth_new_1deg], dim="time"
    )


else:
    metrics_synth_old_1deg = hdp.metric.compute_group_metrics(
        measures_synth_old_1deg,
        thresholds_ref,
        definitions,
        use_doy=True,
        doy_mask=era_summer_mask,
    )
    metrics_synth_new_1deg = hdp.metric.compute_group_metrics(
        measures_synth_new_1deg,
        thresholds_ref,
        definitions,
        use_doy=True,
        doy_mask=era_summer_mask,
    )
    metrics_dataset_synth_1deg = xr.concat(
        [metrics_synth_old_1deg, metrics_synth_new_1deg], dim="time"
    )


# in these intensity metrics (AVI and AVA, zeros mean that there were no heatwaves that year)
# so let's turn those 0s to nans
metrics_dataset_synth_1deg["t2m_x.t2m_x_threshold.AVI"] = metrics_dataset_synth_1deg[
    "t2m_x.t2m_x_threshold.AVI"
].where(metrics_dataset_synth_1deg["t2m_x.t2m_x_threshold.AVI"] != 0)

metrics_dataset_synth_1deg["t2m_x.t2m_x_threshold.AVA"] = metrics_dataset_synth_1deg[
    "t2m_x.t2m_x_threshold.AVA"
].where(metrics_dataset_synth_1deg["t2m_x.t2m_x_threshold.AVA"] != 0)

# compute cumulative heat
metrics_dataset_synth_1deg["t2m_x.t2m_x_threshold.sumHeat"] = (
    metrics_dataset_synth_1deg["t2m_x.t2m_x_threshold.AVA"]
    * metrics_dataset_synth_1deg["t2m_x.t2m_x_threshold.HWF"]
)

metrics_synth_land_1deg = add_landmask(metrics_dataset_synth_1deg)
# metrics_synth_land_1deg.to_netcdf(f"era_hw_metrics_{ref_years[0]}_{new_years[1]}_synth_1deg_anom{suffix}.nc")


###########
# synthetic, 2deg shift
#############

# update the "time" coordinate in the future to pretend it's the "future"
# shift by 2 degrees
era_land_synth_new_2deg = era_land_climatology_years + 2
era_land_synth_new_2deg = era_land_synth_new_2deg.assign_coords(
    time=synth_time
    # time=era_land_new.time
)  # pretend its the new time period

# combine back. this is comparable to era_land_all below
era_land_synth_2deg = xr.concat([era_land_ref, era_land_synth_new_2deg], dim="time")
era_land_synth_anom_2deg = (
    era_land_synth_2deg.groupby("time.dayofyear") - ref_doy_climatology
).drop_vars("dayofyear")
era_land_synth_anom_2deg["t2m_x"].attrs = {
    "units": "C"
}  # hdp package needs units, and anom are same in K or C

# calculate heatwave metrics
measures_synth_2deg = hdp.measure.format_standard_measures(
    temp_datasets=[era_land_synth_anom_2deg["t2m_x"]]
).chunk({"time": -1, "lat": 10, "lon": 10})

# if there isnt a gap between the 2 periods, then you can just pass measures_synth to compute_group_metrics
# but if there's a gap, then need to split into old and new
measures_synth_old_2deg = measures_synth_2deg.sel(
    time=slice(str(ref_years[0] - 1), str(ref_years[1]))
)
measures_synth_new_2deg = measures_synth_2deg.sel(
    time=slice(str(new_years[0] - 1), str(new_years[1]))
)

if use_calendar_summer:
    metrics_synth_old_2deg = hdp.metric.compute_group_metrics(
        measures_synth_old_2deg,
        thresholds_ref,
        definitions,
        use_doy=False,
        start=(6, 1),
        end=(9, 1),
    )
    metrics_synth_new_2deg = hdp.metric.compute_group_metrics(
        measures_synth_new_2deg,
        thresholds_ref,
        definitions,
        use_doy=False,
        start=(6, 1),
        end=(9, 1),
    )
    metrics_dataset_synth_2deg = xr.concat(
        [metrics_synth_old_2deg, metrics_synth_new_2deg], dim="time"
    )


else:
    metrics_synth_old_2deg = hdp.metric.compute_group_metrics(
        measures_synth_old_2deg,
        thresholds_ref,
        definitions,
        use_doy=True,
        doy_mask=era_summer_mask,
    )
    metrics_synth_new_2deg = hdp.metric.compute_group_metrics(
        measures_synth_new_2deg,
        thresholds_ref,
        definitions,
        use_doy=True,
        doy_mask=era_summer_mask,
    )
    metrics_dataset_synth_2deg = xr.concat(
        [metrics_synth_old_2deg, metrics_synth_new_2deg], dim="time"
    )


# in these intensity metrics (AVI and AVA, zeros mean that there were no heatwaves that year)
# so let's turn those 0s to nans
metrics_dataset_synth_2deg["t2m_x.t2m_x_threshold.AVI"] = metrics_dataset_synth_2deg[
    "t2m_x.t2m_x_threshold.AVI"
].where(metrics_dataset_synth_2deg["t2m_x.t2m_x_threshold.AVI"] != 0)

metrics_dataset_synth_2deg["t2m_x.t2m_x_threshold.AVA"] = metrics_dataset_synth_2deg[
    "t2m_x.t2m_x_threshold.AVA"
].where(metrics_dataset_synth_2deg["t2m_x.t2m_x_threshold.AVA"] != 0)

# compute cumulative heat
metrics_dataset_synth_2deg["t2m_x.t2m_x_threshold.sumHeat"] = (
    metrics_dataset_synth_2deg["t2m_x.t2m_x_threshold.AVA"]
    * metrics_dataset_synth_2deg["t2m_x.t2m_x_threshold.HWF"]
)

metrics_synth_land_2deg = add_landmask(metrics_dataset_synth_2deg)
# metrics_synth_land_2deg.to_netcdf(f"era_hw_metrics_{ref_years[0]}_{new_years[1]}_synth_2deg_anom{suffix}.nc")


# #############################
# # deep dives
# #############################


# sudan_lon = 30
# sudan_lat = 6

# brazil_lon = -50
# brazil_lat = -9

# # era_land_all_anom = era_land_all_anom.assign_coords(lon=(((era_land_all_anom.lon + 180) % 360) - 180)).sortby("lon")

# # np.argwhere(era_summer_mask.sel(lat = sudan_lat, method = 'nearest').values == 1) # see what the summer doys are
# # era_land_all_anom_doy_masked = era_land_all_anom.groupby("time.dayofyear").where(
# #     era_summer_mask == 1
# # )
# # era_land_all_anom_jja_masked = era_land_all_anom.where(era_land_all_anom['time.month'].isin([6,7,8]), drop = True)

# # tmax_sudan_doy = era_land_all_anom_doy_masked.sel(lon=sudan_lon, lat=sudan_lat, method="nearest").groupby('time.year').mean(dim='time')
# # fig_sudan_doy = tmax_sudan_doy.hvplot(title = 'sudan tmax anom, using doy 14-104')


# # tmax_sudan_jja = era_land_all_anom_jja_masked.sel(lon=sudan_lon, lat=sudan_lat, method="nearest").groupby('time.year').mean(dim='time')
# # fig_sudan_jja = tmax_sudan_jja.hvplot(title = 'sudan tmax anom, using jja')

# # fig_sudan = (fig_sudan_doy + fig_sudan_jja).cols(1)
# # # hvplot.save(fig_sudan, 'tmp.html')

# # # brazil
# # np.argwhere(era_summer_mask.sel(lat = brazil_lat, method = 'nearest').values == 1) # see what the summer doys are
# # tmax_brazil_doy = era_land_all_anom_doy_masked.sel(lon=brazil_lon, lat=brazil_lat, method="nearest").groupby('time.year').mean(dim='time')
# # fig_brazil_doy = tmax_brazil_doy.hvplot(title = 'brazil tmax anom, using doy 207-297 (starts in ~ mid july)')


# # tmax_brazil_jja = era_land_all_anom_jja_masked.sel(lon=brazil_lon, lat=brazil_lat, method="nearest").groupby('time.year').mean(dim='time')
# # fig_brazil_jja = tmax_brazil_jja.hvplot(title = 'brazil tmax anom, using djf')

# # fig_brazil = (fig_brazil_doy + fig_brazil_jja).cols(1)
# # # hvplot.save(fig_brazil, 'tmp.html')


# # # lookinag at raw units (not anomalies)


# era_land_all_doy_masked = era_land_all.groupby("time.dayofyear").where(
#     era_summer_mask == 1
# )
# era_land_all_jja_masked = era_land_all.where(
#     era_land_all["time.month"].isin([6, 7, 8]), drop=True
# )

# tmax_sudan_doy_raw = (
#     era_land_all_doy_masked.sel(lon=sudan_lon, lat=sudan_lat, method="nearest")
#     .groupby("time.year")
#     .mean(dim="time")
# )
# tmax_sudan_jja_raw = (
#     era_land_all_jja_masked.sel(lon=sudan_lon, lat=sudan_lat, method="nearest")
#     .groupby("time.year")
#     .mean(dim="time")
# )

# tmax_brazil_doy_raw = (
#     era_land_all_doy_masked.sel(lon=brazil_lon, lat=brazil_lat, method="nearest")
#     .groupby("time.year")
#     .mean(dim="time")
# )
# tmax_brazil_jja_raw = (
#     era_land_all_jja_masked.sel(lon=brazil_lon, lat=brazil_lat, method="nearest")
#     .groupby("time.year")
#     .mean(dim="time")
# )

# import hvplot.xarray

# hvplot.extension("bokeh")

# fig_brazil_doy_raw = tmax_brazil_doy_raw.hvplot(
#     title="brazil tmax, using doy 207-297 (starts in ~ mid july)"
# )
# fig_brazil_jja_raw = tmax_brazil_jja_raw.hvplot(title="brazil tmax, using djf")

# fig_brazil_raw = (fig_brazil_doy_raw + fig_brazil_jja_raw).cols(1)
# # hvplot.save(fig_brazil_raw, 'tmp.html')


# fig_sudan_doy_raw = tmax_sudan_doy_raw.hvplot(title="sudan tmax, using doy 14-104")
# fig_sudan_jja_raw = tmax_sudan_jja_raw.hvplot(title="sudan tmax, using jja")
# fig_sudan_raw = (fig_sudan_doy_raw + fig_sudan_jja_raw).cols(1)
# # hvplot.save(fig_sudan_raw, 'tmp.html')


# sudan_snippet = era_land_all.sel(lon=sudan_lon, lat=sudan_lat, method="nearest").sel(
#     time=slice("1963", "1964")
# )
# fig_ela = sudan_snippet.hvplot()
# fig_era = (
#     era["t2m_x"]
#     .sel(lon=sudan_lon, lat=sudan_lat, method="nearest")
#     .sel(time=slice("1963", "1964"))
#     .hvplot()
# )

# era_df = (
#     era["t2m_x"]
#     .sel(lon=sudan_lon, lat=sudan_lat, method="nearest")
#     .sel(time=slice("1964", "1964"))
#     .to_dataframe()
# )

# a = xr.open_dataset("D:data\\ERA5\\t2m_x_1x1_2025\\t2m_x_daily_1x1_1964.nc")
# a = a.assign_coords(lon=(((a.lon + 180) % 360) - 180)).sortby("lon")
# aa = (
#     a["t2m"]
#     .sel(lon=sudan_lon, lat=sudan_lat, method="nearest")
#     .sel(time=slice("1963", "1964"))
# )
# fig_aa = aa.hvplot()

# (fig_era + fig_ela + fig_aa).cols(1)
