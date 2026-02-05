"""
Demonstrating data quality issues in central Africa in ERA daily maximum, in the 1960s
"""

from changing_heat_extremes import plot_helpers as phelpers
import regionmask
import glob
import xarray as xr
import hvplot.xarray  # noqa: F401

####################
# read in ERA
####################
era_filelist = glob.glob("/mnt/media-drive/data/ERA5/t2m_x_1x1/*.nc")
era = (
    xr.open_mfdataset(era_filelist)
    .rename({"__xarray_dataarray_variable__": "t2m_x", "valid_time": "time"})
    .reset_coords(names="number", drop=True)
    .convert_calendar(calendar="noleap", use_cftime=True)
)
# convert to (-180, 180) lon.
era = era.assign_coords(lon=(((era.lon + 180) % 360) - 180)).sortby("lon")

####################
# subset to region
####################

# create a landmask
land = regionmask.defined_regions.natural_earth_v5_0_0.land_110
landmask = land.mask(era)  # ocean is nan, land is 0
is_land = landmask == 0

# also remove central africa bc of data quality concerns in the 1960s
central_africa = regionmask.defined_regions.srex[["W. Africa", "E. Africa"]]
central_africa_mask = central_africa.mask(era)
is_central_africa = ~central_africa_mask.isnull()

# apply landmask
era_masked = era.where(is_land & is_central_africa)

##################################################################
# look at an example gridcell with suspected data quality issues
##################################################################

# pull out an example gridcell in ~south sudan
eg_lon = 30.5
eg_lat = 6.5

era_eg = era_masked["t2m_x"].sel(lon=eg_lon, lat=eg_lat, method="nearest")

# era_sudan.sel(time=slice("1964", "1964")).hvplot(title="see the big drop in late july 1964")
fig_qc = (
    era_eg.sel(time=slice("1964", "1965"))
    .hvplot(title="6.5N, 30.5E", color="black", xlabel="", ylabel="Tx (K)")
    .opts(frame_height=phelpers.fheight_wide, xrotation=45, **phelpers.global_kwargs)
)

# hvplot.save(fig_qc, phelpers.fig_dir / "supplemental" / "fig_qc.png")
