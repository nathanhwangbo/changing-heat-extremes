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
from functools import partial

hvplot.extension(phelpers.backend_hv)


fig_dir = Path("figures")
data_dir = Path("processed_data")


combined_ds = xr.open_dataset(data_dir / f"moments_ds_{flags.label}.nc")


# shared plotting arguments
qm_kwargs = dict(coastline=True, projection=ccrs.PlateCarree())
fig_kwargs = dict(
    xlabel="",
    ylabel="",
    fig_inches=(phelpers.width_default, phelpers.height_wide),
    xaxis=None,
    yaxis=None,
    **phelpers.global_kwargs,
)

################
# mean shift map
################

## old method of generating colorbars
# cbar_kwargs_meanshift0 = phelpers.cbar_discrete(
#     -0.5, 2, cmap="RdBu_r", zero_centered=True, extension=phelpers.backend_hv
# )
# fig_meanshift0 = (
#     combined_ds["t2m_x_mean_diff"]
#     .hvplot.quadmesh(**qm_kwargs)
#     .opts(
#         hv.opts.QuadMesh(
#             title="(a) Mean Shift",
#             clabel="°C",
#             **fig_kwargs,
#             **cbar_kwargs_meanshift0
#         )
#     )
# )

cbar_kwargs_meanshift = phelpers.cbar_helper(-0.5, 2, cmap="RdBu_r", cmap_center=0)


fig_meanshift = (
    combined_ds["t2m_x_mean_diff"]
    .hvplot.quadmesh(**qm_kwargs)
    .opts(
        hv.opts.QuadMesh(
            title="(a) Mean Shift",
            clabel="°C",
            hooks=[
                partial(
                    phelpers.zero_center_hook_mpl, cbar_kwargs=cbar_kwargs_meanshift
                )
            ],
            **fig_kwargs,
        )
    )
)


################
# variance shift map
################


cbar_kwargs_var = phelpers.cbar_helper_hv(
    0, 50, cmap=phelpers.reds_cmap, extension=phelpers.backend_hv
)
fig_var = (
    combined_ds["t2m_x_var"]
    .hvplot.quadmesh(**qm_kwargs)
    .opts(
        hv.opts.QuadMesh(
            title="(b) Climatological Variance",
            clabel="°C²",
            hooks=[cbar_kwargs_var],
            **fig_kwargs,
        )
    )
)


################
# skew shift map
################

cbar_kwargs_skew = phelpers.cbar_helper_hv(
    -1.1, 0.5, cmap="RdBu_r", cmap_center=0, extension=phelpers.backend_hv
)
fig_skew = (
    combined_ds["t2m_x_skew"]
    .hvplot.quadmesh(**qm_kwargs)
    .opts(
        hv.opts.QuadMesh(
            title="(c) Climatological Skew",
            clabel="",
            hooks=[cbar_kwargs_skew],
            **fig_kwargs,
        )
    )
)


###################
# Combining figures
###################


fig_moments = (fig_meanshift + fig_var + fig_skew).cols(3)
fig_moments.opts(sublabel_format="", tight=True, tight_padding=7)
# hvplot.save(fig_moments, fig_dir / f"fig_moments_{flags.label}.png")


###################################
# supplemental analyses used in the paper
####################################

# # adding contour at 20C^2, where there seems to be a elbow in the variance (in 2d_scatter_figs.py)
# fig_var_contour = combined_ds["t2m_x_var"].hvplot.contour(
#     levels=[20],
#     linewidth=2,
#     cmap=["blue"],
#     colorbar=False,
#     legend=False,
#     **qm_kwargs,
#     **phelpers.global_kwargs,
# )

# fig_var_contour_final = (fig_var * fig_var_contour).opts(
#     title="Climatological Variance\ncontour at 20°C²"
# )
# # hvplot.save(fig_var_contour_final, fig_dir / "supplemental" /  f"fig_var_contour_{flags.label}.png")
