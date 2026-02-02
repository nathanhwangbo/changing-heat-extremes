import numpy as np
import xarray as xr
import regionmask
import hdp
import dask

xr.set_options(use_new_combine_kwarg_defaults=True)
dask.config.set(scheduler="single-threaded")

###############################################
## helper functions used in 0_era_meanshift.py
###############################################


def add_landmask(ds):
    """
    add landmask, remove greenland, antarctica, and central africa

    Args:
        ds (xarray dataset): must contain lon and lat (should probably be a gridded product)

    Returns:
        xarray dataset: the same shape as ds, but with the mask applied.
    """
    # create a landmask
    land = regionmask.defined_regions.natural_earth_v5_0_0.land_110
    landmask = land.mask(ds)  # ocean is nan, land is 0
    is_land = landmask == 0

    # also get rid of greenland
    greenland = regionmask.defined_regions.natural_earth_v5_0_0.countries_110[["Greenland"]]
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
            mu = empirical_sc.mean(dim="dayofyear")  # map of average across all days. dim (lat, lon)
        elif n_time == 12:
            # get empirical average for the month
            empirical_sc = da.groupby("time.month").mean()  # dim (month, lat, lon)
            mu = empirical_sc.mean(dim="month")  # map of average across all days. dim (lat, lon)
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
        da_metrics["t2m_x.t2m_x_threshold.AVA"] * da_metrics["t2m_x.t2m_x_threshold.HWF"]
    )

    # in these intensity metrics (AVI and AVA, zeros mean that there were no heatwaves that year)
    # so let's turn those 0s to nans
    # this would be useful, if we wanted AVI to measure "average intensity of a day | day was a heatwave day"
    # This reduces the dependence of AVI on heatwave frequency.
    da_metrics["t2m_x.t2m_x_threshold.AVI"] = da_metrics["t2m_x.t2m_x_threshold.AVI"].where(
        da_metrics["t2m_x.t2m_x_threshold.AVI"] != 0
    )

    da_metrics["t2m_x.t2m_x_threshold.AVA"] = da_metrics["t2m_x.t2m_x_threshold.AVA"].where(
        da_metrics["t2m_x.t2m_x_threshold.AVA"] != 0
    )

    metrics_synth_land = add_landmask(da_metrics)
    return metrics_synth_land


def get_synthetic_hw_metrics(da_ref, da_synth_new, new_years, thresholds_ref, definitions, use_calendar_summer=True):
    """
    generate heatwave metrics given a early period xr object (da_ref) and a synthetic latter period (da_synth_new)

    Args:
        da_ref (xr dataarray or dataset): observed tmax in the "reference" period
        da_synth_new (xr dataarray or datast): synthetic tmax in the "new" period
        new_years (list of size 2): the intended first and last years of the "new" period
        thresholds_ref (xr object from hdp.threshold.compute_threshold): q90 tmax from the HDP package.
        definitions (list of size 3): heatwave definition, following HDP package formatting
        use_calendar_summer (bool, optional): whether to use calendar summer or heat-defined summer as the 90 days surrounding
            the climatological hottest day. Defaults to True.

    Returns:
        xr dataset: heatwave metrics for each year in the two period (format follows result of hdp.metric.compute_group_metrics).
        Note that there might be a gap in time, if there's a gap in time between da_ref and da_synth_new.
    """
    ref_years = [da_ref.time.values.min().year, da_ref.time.values.max().year]
    synth_time = xr.date_range(
        start=str(new_years[0] - 1),
        periods=da_ref.time.size,
        freq="D",
        calendar="noleap",
        use_cftime=True,
    )
    era_land_synth_new = da_synth_new.assign_coords(time=synth_time)  # pretend its the new time period

    # combine back. this is comparable to era_land_all_anom above, except with a small gap in the middle (1990-95)
    era_land_synth_anom = xr.concat([da_ref, era_land_synth_new], dim="time")

    era_land_synth_anom["t2m_x"].attrs = {"units": "C"}  # hdp package needs units, and anom are same in K or C

    # calculate heatwave metrics
    measures_synth = hdp.measure.format_standard_measures(temp_datasets=[era_land_synth_anom["t2m_x"]]).chunk(
        {"time": -1, "lat": 10, "lon": 10}
    )

    # if there isnt a gap between the 2 periods, then you can just pass measures_synth to compute_group_metrics
    # but if there's a gap, then need to split into old and new
    measures_synth_old = measures_synth.sel(time=slice(str(ref_years[0] - 1), str(ref_years[1])))
    measures_synth_new = measures_synth.sel(time=slice(str(new_years[0] - 1), str(new_years[1])))

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
        metrics_dataset_synth = xr.concat([metrics_synth_old, metrics_synth_new], dim="time")

    else:
        # day of year summer mask
        era_summer_path = "/mnt/media-drive/data/ERA5/ERA5_hottest_doys_t2m_x.nc"
        era_summer_mask = xr.open_dataarray(era_summer_path)
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
        metrics_dataset_synth = xr.concat([metrics_synth_old, metrics_synth_new], dim="time")

    metrics_synth_land = process_heatwave_metrics(metrics_dataset_synth)

    return metrics_synth_land


def write_nc(ds, filepath):
    # a default choice of compression
    comp = dict(zlib=True, complevel=5)

    if type(ds) is xr.Dataset:
        for var in ds:
            ds[var].encoding.update(comp)
    if type(ds) is xr.DataArray:
        ds.encoding.update(comp)

    ds.to_netcdf(filepath, format="NETCDF4")
