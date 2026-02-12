"""
analyzing the output of 0_era_meanshift.py
- looking at heatwave metrics as a function of "climatological" variance and skew
"""

from changing_heat_extremes import flags
from changing_heat_extremes import analysis_helpers as ahelpers
import xarray as xr
from xarray_einstats import stats  # wrapper around apply_ufunc for moments
from pathlib import Path

data_dir = Path("processed_data")


##############################################################################
# Calculate mean differences (1986-2021) - (1950-1985) for heatwave metrics
##############################################################################


hw_all = (
    xr.open_dataset(
        data_dir
        / f"hw_metrics_{flags.ref_years[0]}_{flags.new_years[1]}_anom{flags.label}.nc"
    )
    .sel(
        percentile=flags.percentile_threshold,
        definition="-".join(map(str, flags.hw_def[0])),
    )
    .drop_vars(["percentile", "definition"])
)
hw_synth_1deg = (
    xr.open_dataset(
        data_dir
        / f"hw_metrics_{flags.ref_years[0]}_{flags.new_years[1]}_synth_1deg_anom{flags.label}.nc"
    )
    .sel(
        percentile=flags.percentile_threshold,
        definition="-".join(map(str, flags.hw_def[0])),
    )
    .drop_vars(["percentile", "definition"])
)
hw_synth_2deg = (
    xr.open_dataset(
        data_dir
        / f"hw_metrics_{flags.ref_years[0]}_{flags.new_years[1]}_synth_2deg_anom{flags.label}.nc"
    )
    .sel(
        percentile=flags.percentile_threshold,
        definition="-".join(map(str, flags.hw_def[0])),
    )
    .drop_vars(["percentile", "definition"])
)


# compute deltas
hw_old = hw_all.sel(time=slice(str(flags.ref_years[0]), str(flags.ref_years[1])))
hw_new = hw_all.sel(time=slice(str(flags.new_years[0]), str(flags.new_years[1])))
hw_mean_diff = hw_new.mean(dim="time") - hw_old.mean(dim="time")


#######################################################################
# Calculate mean differences (new period) - (old period) for temperature

# NOTE! This comes from ERA daily TMAX which has been processed by:
#   - centered to have mean zero over the entire 1960-2025 period
#   - removing day-of-year means, so that the Jan 12 time series is mean 0.
#       - with the caveat that the day-of-year means are calculated using the "climatological dataset" (see next chunk)
#  (this is the same data that was used to calculate the heatwave metrics.)
#######################################################################

# anomalies calculated in 0_era_meanshift.py
era_land_anom = xr.open_dataset(
    data_dir / f"land_anom_{flags.ref_years[0]}_{flags.ref_years[1]}.nc"
)

# compute deltas-------------------------------------------
era_land_old = era_land_anom.sel(
    time=slice(str(flags.ref_years[0]), str(flags.ref_years[1]))
)
era_land_new = era_land_anom.sel(
    time=slice(str(flags.new_years[0]), str(flags.new_years[1]))
)
tmax_mean_diff = (era_land_new.mean(dim="time") - era_land_old.mean(dim="time")).rename(
    {"t2m_x": "t2m_x_mean_diff"}
)

##############################################
# Calculate climatological moments
# NOTE! This comes from ERA daily TMAX which has been processed by:
#   - centered to have mean zero over the entire 1960-2025 period
#   - removing individual yearly means, so that each year is mean 0 (removing warming signal)
#   - removing day-of-year means, so that the Jan 12 time series is mean 0.
##############################################

era_land_anom_for_climatology = xr.open_dataset(
    data_dir / f"land_anom_for_climatology_{flags.ref_years[0]}_{flags.new_years[1]}.nc"
)

clim_skew = stats.skew(era_land_anom_for_climatology["t2m_x"], dims=["time"]).rename(
    "t2m_x_skew"
)
clim_kurt = stats.kurtosis(
    era_land_anom_for_climatology["t2m_x"], dims=["time"]
).rename("t2m_x_kurt")
clim_var = era_land_anom_for_climatology["t2m_x"].var(dim="time").rename("t2m_x_var")
clim_ar1 = xr.corr(
    era_land_anom_for_climatology["t2m_x"],
    era_land_anom_for_climatology["t2m_x"].shift(time=1),
    dim="time",
).rename("t2m_x_ar1")

climatology_stats = xr.merge([clim_skew, clim_kurt, clim_var, clim_ar1])


##############################################
# combine maps into 1 xr.dataset
##############################################

combined_ds = xr.merge([tmax_mean_diff, climatology_stats, hw_mean_diff], join="exact")

# ! uncomment to write to file
# ahelpers.write_nc(combined_ds, data_dir / f"moments_ds_{flags.label}.nc")
