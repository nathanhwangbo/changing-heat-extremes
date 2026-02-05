"""
maps of
- climatological variance
- climatoloical skew
- how much the mean of Tx has shifted in ERA5 between ref_years and new_years
"""

from changing_heat_extremes import flags
from changing_heat_extremes import plot_helpers as phelpers
import xarray as xr
import holoviews as hv
import hvplot.xarray  # noqa: F401
import hvplot.pandas  # noqa: F401
from pathlib import Path
import cartopy.crs as ccrs

fig_dir = Path("figures")
data_dir = Path("processed_data")


combined_ds = xr.open_dataset(data_dir / f"moments_ds_{flags.label}.nc")


# scale = 1  # in this case,
# title_size = 16 * scale
# label_size = 14 * scale
# tick_size = 10 * scale
# fwidth = 400
# fheight = 150

# shared plotting arguments
qm_kwargs = dict(coastline=True, projection=ccrs.PlateCarree())
fig_kwargs = dict(
    xlabel="",
    ylabel="",
    xticks=0,
    yticks=0,
    xlim=(-180, 180),
    frame_height=phelpers.fheight_wide,
    **phelpers.global_kwargs,
)

################
# mean shift map
################

cbar_kwargs_meanshift = phelpers.cbar_discrete(-0.5, 2, cmap="RdBu_r", zero_centered=True)
fig_meanshift = (
    combined_ds["t2m_x_mean_diff"]
    .hvplot.quadmesh(**qm_kwargs)
    .opts(hv.opts.QuadMesh(title="(a) Mean Shift", clabel="°C", **fig_kwargs, **cbar_kwargs_meanshift))
)


################
# variance shift map
################

cbar_kwargs_var = phelpers.cbar_discrete(0, 50, cmap=phelpers.reds_cmap, zero_centered=False)
fig_var = (
    combined_ds["t2m_x_var"]
    .hvplot.quadmesh(**qm_kwargs)
    .opts(hv.opts.QuadMesh(title="(b) Climatological Variance", clabel="°C²", **fig_kwargs, **cbar_kwargs_var))
)


################
# skew shift map
################

cbar_kwargs_skew = phelpers.cbar_discrete(-1.1, 0.5, cmap="RdBu_r")
fig_skew = (
    combined_ds["t2m_x_skew"]
    .hvplot.quadmesh(**qm_kwargs)
    .opts(hv.opts.QuadMesh(title="(c) Climatological Skew", **fig_kwargs, **cbar_kwargs_skew))
)


###################
# Combining figures
###################

fig_moments = (fig_meanshift + fig_var + fig_skew).cols(3)
# hvplot.save(fig_moments, fig_dir / f"fig_moments_{flags.label}.png")


###################################
# supplemental analyses used in the paper
####################################

# adding contour at 20C^2, where there seems to be a elbow in the variance (in 2d_scatter_figs.py)
fig_var_contour = combined_ds["t2m_x_var"].hvplot.contour(
    levels=[20], line_width=2, cmap=["blue"], colorbar=False, legend=False, **qm_kwargs, **phelpers.global_kwargs
)

fig_var_contour_final = (fig_var * fig_var_contour).opts(title="Climatological Variance\ncontour at 20°C²")
# hvplot.save(fig_var_contour_final, fig_dir / "supplemental" /  f"fig_var_contour_{flags.label}.png")
