"""script to calculate summer land heatwave metrics from ERA

inputs:
    - daily maximum temperature from ERA on a 1x1 grid (downloaded in in src/00_get_era_daily_tmax.py and processed in src/01_process_era.py)
    - (optional) a doy-mask specifying the "summer" season, if using the heat-based definition of summer instead of JJA.

outputs (for use in subsequent scripts in /analysis/):
    - a map with the q90(tmax) threshold used to define heatwaves (uses reference period 1960-1990)
    - heatwave metrics per year                                   (uses all years)

- To estimate heatwave metrics, we:
    - 1. estimate the day-of-year climatology by taking the mean across 1960:1990 and smoothing via 5 fourier basis functions
    - 2. remove this day-of-year climatology, then calculate the q90 thresholds
        - q90 is specific to each gridcell and each day-of-year, using 7 day windows around each day-of-year
        - i.e. for each day of year, q90 is estimated from 7 days * 31 years
        - The q90 thresholds are then smoothed via 5 fourier basis functions
    - 3. use this threshold to calculate heatwave metrics across 1960-2025 (one value per summer)
        - heatwaves are defined as 3+ consecutive days where the anomalies exceed q90


We note that by calculating q90 over just the 1960-1990 period, we guarantee that every gridcell has days which exceed this threshold in the 1960-1990 period.
"""

from changing_heat_extremes import flags
from changing_heat_extremes import analysis_helpers as ahelpers
import xarray as xr
import glob
import hdp  # pip install -e "E:\\Projects\\HDP\\"

#########################################################
#### DECISION! is summer defined via calendar year (JJA?)
# if false, summer is defined via 90 days surrounding the hottest day (for each latitude)
# flag is set in flags.py
#########################################################
if flags.use_calendar_summer is False:
    # day of year summer mask
    era_summer_path = "/mnt/media-drive/data/ERA5/ERA5_hottest_doys_t2m_x.nc"
    era_summer_mask = xr.open_dataarray(era_summer_path)


#######################################
# read in ERA data
#######################################


# era_filelist = glob.glob("D:data\\ERA5\\t2m_x_1x1\\*.nc")
# era = xr.open_mfdataset(era_filelist).drop_vars("expver")
era_filelist = glob.glob("/mnt/media-drive/data/ERA5/t2m_x_1x1/*.nc")
era = (
    xr.open_mfdataset(era_filelist)
    .rename({"__xarray_dataarray_variable__": "t2m_x", "valid_time": "time"})
    .reset_coords(names="number", drop=True)
)


# fixing formatting for the hdp package
era["t2m_x"].attrs = {"units": "K"}  # hdp package needs units
era = era.convert_calendar(calendar="noleap", use_cftime=True)
era = era.sel(lat=slice(-60, 80)).chunk(
    {"time": -1, "lat": 10, "lon": 10}
)  # matching karen's doy mask

# convert to (-180, 180) lon.
era = era.assign_coords(lon=(((era.lon + 180) % 360) - 180)).sortby("lon")


# add landmask
era_land = ahelpers.add_landmask(era).compute()


#################################################################
# Calculate base period (1960-1990) characteristics
#### calculate doy thresholds, and smooth the threshold ---------
##################################################################

era_land_ref_years = era_land.sel(
    time=slice(str(flags.ref_years[0] - 1), str(flags.ref_years[1]))
)
# take doy anomalies
ref_doy_climatology = ahelpers.fourier_climatology_smoother(
    era_land_ref_years["t2m_x"], n_time=365, n_bases=5
)
era_land_ref_anom = (
    era_land_ref_years.groupby("time.dayofyear") - ref_doy_climatology
).drop_vars("dayofyear")
era_land_ref_anom["t2m_x"].attrs = {"units": "C"}  # hdp package needs units


# conversion to celcius
measures_ref = hdp.measure.format_standard_measures(
    temp_datasets=[era_land_ref_anom["t2m_x"]]
)
thresholds_ref_unsmooth = hdp.threshold.compute_thresholds(
    measures_ref, percentiles=[flags.percentile_threshold], rolling_window_size=7
).compute()

## smoothing out the the threshold climatology as well --------

thresholds_ref_smoothed = ahelpers.fourier_climatology_smoother(
    thresholds_ref_unsmooth["t2m_x_threshold"]
    .sel(percentile=flags.percentile_threshold)
    .drop_vars("percentile"),
    n_time=365,
    n_bases=5,
)

# match the formatting of the original hdp function ---
thresholds_ref = (
    thresholds_ref_smoothed.to_dataset()
    .expand_dims(percentile=[flags.percentile_threshold])
    .transpose("lat", "lon", "doy", "percentile")
)
thresholds_ref["t2m_x_threshold"].attrs["baseline_variable"] = "t2m_x"
thresholds_ref["t2m_x_threshold"].attrs["hdp_type"] = "threshold"
thresholds_ref["t2m_x_threshold"].attrs["baseline_calendar"] = "noleap"
thresholds_ref = thresholds_ref.chunk({"lat": 10, "lon": 10})

# ! uncomment to save output
# ahelpers.write_nc(
#     thresholds_ref, f"processed_data/thresholds_{flags.ref_years[0]}_{flags.ref_years[1]}_{flags.label}.nc"
# )


################################################
# calculate extremal metrics at each gridcell
###############################################

############################
# observations (not synthetic)
# time period ALL 1960-2025,
##############################

era_land_all = era_land.sel(
    time=slice(str(flags.ref_years[0] - 1), str(flags.new_years[1]))
)
era_land_all_anom = (
    era_land_all.groupby("time.dayofyear") - ref_doy_climatology
).drop_vars("dayofyear")

# ! uncomment to save output. this is is the data that's used to calculate heatwave metrics
# ahelpers.write_nc(era_land_all_anom, f"processed_data/land_anom_{flags.ref_years[0]}_{flags.ref_years[1]}.nc")

era_land_all_anom["t2m_x"].attrs = {"units": "C"}  # hdp package needs units
# use c when dealing with anomalies (bc anomalies are the same in either units)

# calculate heatwave metrics on time period.
measures_all = hdp.measure.format_standard_measures(
    temp_datasets=[era_land_all_anom["t2m_x"]]
).chunk({"time": -1, "lat": 10, "lon": 10})


if flags.use_calendar_summer:
    metrics_dataset_all = hdp.metric.compute_group_metrics(
        measures_all, thresholds_ref, flags.hw_def, start=(6, 1), end=(9, 1)
    )
else:
    metrics_dataset_all = hdp.metric.compute_group_metrics(
        measures_all,
        thresholds_ref,
        flags.hw_def,
        use_doy=True,
        doy_mask=era_summer_mask,
    )

metrics_all_land = ahelpers.process_heatwave_metrics(metrics_dataset_all)

# ! uncomment to save output
# ahelpers.write_nc(
#     metrics_all_land, f"processed_data/hw_metrics_{flags.ref_years[0]}_{flags.new_years[1]}_anom{flags.label}.nc"
# )


###########
# synthetic, meanshift
#############

# note that we do this on the shifted, doy-anomaly removed ts.

# time period 2,
# era_land_new = era_land.sel(time=slice(str(flags.new_years[0]), str(flags.new_years[1])))
era_land_ref_anom = era_land_all_anom.sel(
    time=slice(str(flags.ref_years[0] - 1), str(flags.ref_years[1]))
)
era_land_new_anom = era_land_all_anom.sel(
    time=slice(str(flags.new_years[0] - 1), str(flags.new_years[1]))
)

## shifting the mean for each grid cell.
## note: this is in anomaly space!! shifted, doy-anomaly removed.
old_means = era_land_ref_anom["t2m_x"].mean(dim=["time"])
new_means = era_land_new_anom["t2m_x"].mean(dim=["time"])
# old_means = era_land_ref["t2m_x"].mean(dim=["time"])
# new_means = era_land_new["t2m_x"].mean(dim=["time"])

# update the "time" coordinate in the future to pretend it's the "future"
era_land_synth_new = (era_land_ref_anom - old_means) + new_means

metrics_synth_land = ahelpers.get_synthetic_hw_metrics(
    era_land_ref_anom,
    era_land_synth_new,
    flags.new_years,
    thresholds_ref,
    flags.hw_def,
    use_calendar_summer=flags.use_calendar_summer,
)

# ! uncomment to save output
# ahelpers.write_nc(
#     metrics_synth_land,
#     f"processed_data/hw_metrics_{flags.ref_years[0]}_{flags.new_years[1]}_synth_anom{flags.label}.nc",
# )


###########
# synthetic, 1deg shift
# i.e. the "future" time period (1995-2025) is the same as 1960-1990 but shifted by 1deg
#############

# update the "time" coordinate in the future to pretend it's the "future"
# shifted by 1 degree.
era_land_synth_new_1deg = era_land_ref_anom + 1

metrics_synth_land_1deg = ahelpers.get_synthetic_hw_metrics(
    era_land_ref_anom,
    era_land_synth_new_1deg,
    flags.new_years,
    thresholds_ref,
    flags.hw_def,
    use_calendar_summer=flags.use_calendar_summer,
)

# ! uncomment to save output
# ahelpers.write_nc(
#     metrics_synth_land_1deg,
#     f"processed_data/hw_metrics_{flags.ref_years[0]}_{flags.new_years[1]}_synth_1deg_anom{flags.label}.nc",
# )


###########
# synthetic, 2deg shift
#############

# update the "time" coordinate in the future to pretend it's the "future"
# shift by 2 degrees
era_land_synth_new_2deg = era_land_ref_anom + 2
metrics_synth_land_2deg = ahelpers.get_synthetic_hw_metrics(
    era_land_ref_anom,
    era_land_synth_new_2deg,
    flags.new_years,
    thresholds_ref,
    flags.hw_def,
    use_calendar_summer=flags.use_calendar_summer,
)

# ! uncomment to save output
# ahelpers.write_nc(
#     metrics_synth_land_2deg,
#     f"processed_data/hw_metrics_{flags.ref_years[0]}_{flags.new_years[1]}_synth_2deg_anom{flags.label}.nc",
# )
