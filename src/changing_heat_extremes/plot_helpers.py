import holoviews as hv
import colorcet  # noqa: F401, bc importing colorcet adds cmaps to mpl.pyplot.colormaps()
import tastymap
import numpy as np
import statsmodels.api as sm
from functools import partial
import string
from bokeh.models import FixedTicker
from bokeh.themes import Theme
from pathlib import Path


# hv.extension("bokeh", enable_mathjax=True)

########################
# plot global variables
# eg themes, colormaps
########################

fig_dir = Path("figures")

rdbu_discrete = tastymap.cook_tmap("RdYlBu_r", num_colors=12).cmap

# rdbu_hex = [mcolors.rgb2hex(rdbu_discrete(i)) for i in range(rdbu_discrete.N)]
# equivalent:
# rdbu_discrete = tastymap.cook_tmap("RdYlBu_r", num_colors=12)
# rdbu_hex = rdbu_discrete.to_model("hex")


# get rid of white
reds_cmap = tastymap.cook_tmap("cet_CET_L18")[20:].cmap  # there are 256 values

# alt:
reds_discrete = tastymap.cook_tmap("cet_CET_L18", num_colors=12)[
    1:11
].cmap  # get rid of white
# equiv:
# reds_discrete = tastymap.cook_tmap("cet_CET_L18", num_colors = 12)
# reds_discrete_no_white = tastymap.utils.subset_cmap(slice(1, 11)))
blues_discrete = tastymap.cook_tmap("blues", num_colors=10).cmap


# Forces a specific font ('sans-serif') -----------
new_font = "sans-serif"
#  [n for n in bokeh.models.Text.properties() if "font" in n] # see what's out there
custom_theme = Theme(
    json={
        "attrs": {
            "Title": {"text_font": new_font},
            "Axis": {
                "axis_label_text_font": new_font,
                "major_label_text_font": new_font,
            },
            "Legend": {"title_text_font": new_font, "label_text_font": new_font},
            "ColorBar": {
                "title_text_font": new_font,
                "major_label_text_font": new_font,
            },
            "Label": {"text_font": new_font},
            "Text": {"text_font": {"value": new_font}},
        }
    }
)

# ppply this theme to all bokeh plots
hv.renderer("bokeh").theme = custom_theme

## shared figure sizes and font sizes ---------
# scale = 1  # in this case,
# title_size = 16 * scale
# label_size = 14 * scale
# tick_size = 10 * scale

# using the maps as reference
title_size = 24
label_size = 21
tick_size = 15
fwidth = 400  # frame height for an individual panel

# change aspect ratio using height only
fheight_wide = 150
fheight_default = 300

# this should be applied to individual panels.
global_kwargs = dict(
    toolbar=None,
    fontscale=1,
    fontsize={
        "title": title_size,
        "labels": label_size,
        "ticks": tick_size,
        "legend": tick_size,
    },
    frame_width=fwidth,
)


##############################
# helper functions
##############################
def subplot_label_hook(plot, element, sub_label=""):
    """
    add subplot label to a single bokeh figure

    Parameters
    ----------
    plot : a single holoviews (bokeh) figure (i.e. not a layout)
    element : ??
        needed for the bokeh backend I think? not really sure
    sub_label : str, optional
        the subplot label (commonly something like (a) or a., ect..), by default ""
    """
    # Access the underlying Bokeh figure
    fig = plot.state

    original_title = fig.title.text
    fig.title.text = f"{sub_label} {original_title}"


def add_subplot_labels(plot, labels=string.ascii_lowercase):
    """
    add subplot labels to a composed layout or a list of plots

    Parameters
    ----------
    plot : hv.Layout with bokeh backend
    custom_labels : list
        list of subplot labels. length should match the number of subplots,
        optional, by default ["a", "b", "c", ...]
    """
    # iterate over the subplots and add the label to the title.
    updated_figlist = []
    for i, subplot in enumerate(plot):
        new_label = f"({labels[i]})"  # this sets the format to (a), (b), ..
        updated_subplot = subplot.opts(
            hooks=[partial(subplot_label_hook, sub_label=new_label)]
        )
        updated_figlist.append(updated_subplot)

    return updated_figlist


def cbar_discrete(
    start, end, cmap="RdBu", width=None, zero_centered=False, extension="bokeh"
):
    """
    a hvplot helper to make a zero-centered segmented colorbar
    :param: start and end are the two endpoints of the colorbar. 0 should be contained inside
    :param: cmap is a string with the desired colormap. should be accepted by tastymap.cook_tmap(), and should probably be a diverging colorbar.
    :param: width is a number, the distance between colors. This will be used to determine the number of colors.
    :param: extension is the hvplot extension -- for now, bokeh and matplotlib are supported.

    :returns: a dictionary to be passed into holoviews .opts
    :example:
    da = xr.tutorial.load_dataset('air_temperature.nc').isel(time=10) - 273
    eg_cbar = cbar_zero_centered(-30, 40)
    da.hvplot().opts(**eg_cbar)
    """
    crange = end - start  # range of the cbar

    # set default width (aka distance between colors)
    if width is None:
        # strategy for setting default width:
        # we're gonna aim for ~10 colors
        # but we're going to prioritize "round" spacings.
        # this strategy is assuming that the crange is an integer

        def divisors(n):
            """
            get divisors of n, used to calculate default width.
            https://stackoverflow.com/a/36700288
            """
            divs = [1]
            for i in range(2, int(np.sqrt(n)) + 1):
                if n % i == 0:
                    divs.extend([i, n / i])
            # divs.extend([n])  # n is a divisor, but not needed for our function
            return np.array(divs)

        def orderOfMag(n):
            """
            example: if n = 0.03, then should return 1/100
            """
            return 10 ** np.floor(np.log10(n))

        # aiming for ~10 colors. this is our initial guess.
        width_ballpark = crange / 10
        if width_ballpark < 1:
            # if 10 evenly spaced colors produces widths < 1, then try widths in [.1, .125,.2, .25,.5]
            # which guarantees that all integers get their own ticks
            # this scales by order of magnitude, in case the range is small.
            # (e.g. if width is 0.03, then should try 0.01, 0.0125, ect..)
            width_candidates = np.array([1, 1.25, 2, 2.5, 5]) * orderOfMag(
                width_ballpark
            )

        else:
            # if the width between colors spans more than 1, then try the divisors
            # TODO: might be worth filtering this so that 0 MUST be one of the tickmarks.
            width_candidates = divisors(crange)

        check_n_colors = crange / width_candidates  # number of colors for each width
        width = width_candidates[
            np.argmin(np.abs(check_n_colors - 10))
        ]  # pick the width that produces closest to 10 colors

    # end getting default width ---

    n_colors = int(np.round(crange / width))  # round to handle weird precision.
    # use n_colors + 1 bc including both endpoints
    ticks = np.linspace(start, end, n_colors + 1).round(10).tolist()

    # label the number at every colorchage
    # todo: if there are a lot of colors, we could do every other number
    if extension == "bokeh":
        tick_loc = {"ticker": FixedTicker(ticks=ticks)}
    elif extension == "matplotlib":
        tick_loc = {"ticks": ticks}
    else:
        exit("only bokeh and matplotlib extensions are supported.")

    ############################################
    # adds logic to center a diverging cmap at 0
    ############################################
    is_not_symmetric = abs(start) != abs(end)
    # check that (1) we want zero centered and (2) the cbar isn't already symmetric
    if (zero_centered) and (is_not_symmetric):
        # find location of 0
        zero_ind = np.searchsorted(ticks, 0)

        # use twice the colors, so that cmap_base[n_colors] is white.
        # TODO: this is a little hacky and leads to muted colors.
        # one alternative would be to split the cbar at zero and rescale (as in matplotlib twoslopenorm)
        # could be done using two calls to tastymap.resize, which are combined using &
        # e.g. cmap_below0 = cmap_base[0:n_colors].resize(zero_ind),
        # cmap_above0 = cmap_base[n_colors:].resize(n_colors - zero_ind)
        # cmap_final = (cmap_below0 & cmap_above0).cmap

        cmap_base = tastymap.cook_tmap(cmap, num_colors=n_colors * 2)

        # n_colors is white, so we want to start zero_ind below, and then pick out the next n_colors.
        # mult by 2 version if you wanna up the saturation
        # cmap_final = cmap_base[(n_colors - zero_ind) : (n_colors - zero_ind + n_colors)] * 2
        cmap_final = cmap_base[(n_colors - zero_ind) : (n_colors - zero_ind + n_colors)]
    else:
        cmap_final = tastymap.cook_tmap(cmap, num_colors=n_colors)

    cbar_dict = dict(
        cmap=cmap_final.cmap,
        clim=(start, end),
        colorbar_opts=tick_loc,
        color_levels=ticks,
    )

    return cbar_dict


def cbar_zero_centered(start, end, num_bins="auto", cmap="RdBu", rescale=False):
    """
    generate a discrete matplotlib colorbar with white at the zero mark, even if colorbar isn't symmetric around zero.
    This function relies on matplotlib's method of automatically choosing tickmark locations, which means it doesn't always respect "(start, end)".
        If you want a version of this that implements some gross custom logic I made for automatically choosing levels, let me know and I'll send it over

    Args:
        start (numeric): the bottom endpoint of the colobar, must be less than zero
        end (numeric): the top endpoint of the colorbar, must be greater than zero
        num_bins (int, optional): number of colors in the colorbar. See matplotlib.ticker.MaxNLocator. Defaults to "auto".
        cmap (str, optional): what colormap to use. see list(plt.colormaps) for options. Diverging colorbars make the most sense here. Defaults to "RdBu".
                              bonus! if you import colorcet, you can pass in any of their colormap names to this argument!
        rescale (bool, optional): whether to rescale the colormap so that the endpoints are equally intense, regardess of their values. Defaults to False.

    Returns:
        dict: dictionary of kwargs that can be passed into xarary's plot accessor. (will also work for most matplotlib functions)

    Example:
        my_kwargs = cbar_zero_centered(-30, 20)
        da = xr.tutorial.open_dataset('air_temperature').isel(time=1).air - 273.15
        da.plot(**my_kwargs)
    """
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    import matplotlib.ticker
    import numpy as np

    # 256 colors, interpolating between 0 and 1
    og_cmap = plt.colormaps[cmap]  # endpoints are 0 and 1, by default

    zero_fraction = (0 - start) / (
        end - start
    )  # eg. zero is "0.4" of the way from the top
    shift_amount = 0.5 - zero_fraction

    if rescale:
        # the bottom and top of the colorbar will be the "darkest" shades, even if the numbers aren't symmetric
        # e.g. for {cmap= 'RdBu', start=-5, end = 30}, -5 will be just as dark red as 30 is dark blue.
        new_colors = og_cmap(
            np.interp(np.linspace(0, 1, og_cmap.N), [0, zero_fraction, 1], [0, 0.5, 1])
        )
    else:
        # e.g. for {cmap= 'RdBu', start=-5, end = 30}, -5 will be just light red and 30 dark blue.
        if shift_amount >= 0:  # if |start| < |end|
            new_colors = og_cmap(
                np.interp(
                    np.linspace(0, 1, og_cmap.N),
                    [0, zero_fraction, 1],
                    [shift_amount, 0.5, 1],
                )
            )
        else:  # if |start| > |end|
            # shift amount is negative, so 1 + shift_amount <  1
            new_colors = og_cmap(
                np.interp(
                    np.linspace(0, 1, og_cmap.N),
                    [0, zero_fraction, 1],
                    [0, 0.5, 1 + shift_amount],
                )
            )

    shifted_cmap = mcolors.ListedColormap(new_colors)

    # rely on mpl's auto-generation of tickmarks
    # this can be annoying sometimes (i.e. it doesn't respect (start, end) exactly)
    levels = matplotlib.ticker.MaxNLocator(nbins=num_bins).tick_values(start, end)
    norm = mcolors.BoundaryNorm(levels, shifted_cmap.N)  # .N should still be 256

    # is putting norms and levels redundant?
    cbar_kwargs = dict(
        cmap=shifted_cmap,
        norm=norm,
        levels=levels,
    )
    return cbar_kwargs


def get_heatmap(
    combined_df,
    ref_years,
    new_years,
    y_name_var,
    y_name_label,
    x_name_base="tmax_diff",
    use_qbins=True,
):
    """
    everything except for combined_df, y_name_var, x_name is just used for labelling purposes.
    """
    if use_qbins:
        x_name = f"{x_name_base}_qbins"
    else:
        x_name = f"{x_name_base}_bins"

    df = combined_df.copy()

    bin_ordering_dict = {
        x_name: df[x_name].cat.categories.astype(str),
        y_name_var: df[y_name_var].cat.categories.astype(str),
    }

    df[x_name] = df[x_name].astype(str)
    df[y_name_var] = df[y_name_var].astype(str)

    # count (2d histogram)
    cbar_count = cbar_discrete(0, 150, cmap="cet_CET_L18")
    fig_count = (
        df.hvplot.heatmap(
            x=x_name,
            y=y_name_var,
            C="t2m_x.t2m_x_threshold.HWF",  # in this plot, this isn't doing anything
            reduce_function=np.size,
            fields={
                "t2m_x.t2m_x_threshold.HWF": "count"
            },  # the tooltip displays count instead of hwf
            # cmap=reds_discrete,
            title=f"Gridcell Counts ({y_name_label})",
            xlabel="Change in Daily Max Anomalies (C)",
            ylabel=f"Climatological {y_name_label}",
            clabel="Number of Gridcells",
        )
        .opts(xrotation=45, **cbar_count)
        .redim.values(  # hack to order the bins
            **bin_ordering_dict
        )
    )

    #############################################################################################
    # There are 14728 land gridcells, so each of the 100 cells would have ~147 points if uniform
    # let's add a marker to the cells with more than 100 points.
    ###########################################################################################

    counts_df = df.value_counts([x_name, y_name_var]).reset_index(name="count")
    select_cells = counts_df[counts_df["count"] >= 100]

    markers = hv.Points(select_cells, kdims=[x_name, y_name_var]).opts(
        color="black",
        marker="x",  # Shape: 'circle', 'square', 'triangle', 'cross', 'x', 'diamond'
        size=10,
    )

    #############################################################################################
    # plot the heatwave metrics
    ###########################################################################################

    # hwf
    fig_hwf = (
        df.hvplot.heatmap(
            x=x_name,
            y=y_name_var,
            C="t2m_x.t2m_x_threshold.HWF",
            reduce_function=np.mean,
            cmap=reds_discrete,
            title="Change in HWF",
            xlabel="Change in Daily Max Anomalies (C)",
            ylabel=f"Climatological {y_name_label}",
            clabel="Days",
            # title=f"mean shift in heatwave frequency\nby climatological {y_name_label} and mean tmax shift",
            # xlabel=f"tmax mean({new_years[0]}:{new_years[1]}) - mean({ref_years[0]}:{ref_years[1]})\nanomalies wrt {ref_years[0]}-{ref_years[1]} (C)",
            # ylabel=f"sample {y_name_label} over {ref_years[0]}-{ref_years[1]}",
            # clabel=f"heatwave frequency (days)\nmean({new_years[0]}:{new_years[1]}) - mean({ref_years[0]}:{ref_years[1]})",
            clim=(1, 11),
            # min_count=10,
        )
        .redim.values(  # hack to order the bins
            **bin_ordering_dict
        )
        .opts(cmap=reds_discrete, cticks=[1, 3, 5, 7, 9, 11], xrotation=45)
    ) * markers

    # hwd
    fig_hwd = (
        df.hvplot.heatmap(
            x=x_name,
            y=y_name_var,
            C="t2m_x.t2m_x_threshold.HWD",
            reduce_function=np.mean,
            title="Change in HWD",
            xlabel="Change in Daily Max Anomalies (C)",
            ylabel=f"Climatological {y_name_label}",
            clabel="Days",
            # title=f"mean shift in heatwave duration\nby climatological {y_name_label} and mean tmax shift",
            # xlabel=f"tmax mean({new_years[0]}:{new_years[1]}) - mean({ref_years[0]}:{ref_years[1]})\nanomalies wrt {ref_years[0]}-{ref_years[1]} (C)",
            # ylabel=f"sample {y_name_label} over {ref_years[0]}-{ref_years[1]}",
            # min_count=10,
        )
        .redim.values(  # hack to order the bins
            **bin_ordering_dict
        )
        .opts(
            cmap=reds_discrete,
            cticks=[0, 1, 2, 3, 4, 5],
            clim=(0, 5),
            # clabel=f"heatwave duration (days)\nmean({new_years[0]}:{new_years[1]}) - mean({ref_years[0]}:{ref_years[1]})",
            xrotation=45,
        )
    ) * markers
    # cumulative intensity
    fig_sumheat = (
        df.hvplot.heatmap(
            x=x_name,
            y=y_name_var,
            C="t2m_x.t2m_x_threshold.sumHeat",
            reduce_function=np.mean,
            # data_aspect=1,
            cmap=reds_discrete,
            title="Change in sumHeat",
            xlabel="Change in Daily Max Anomalies (C)",
            ylabel=f"Climatological {y_name_label}",
            clabel="°C",
            # title=f"mean shift in heatwave cumulative intensity\nby climatological {y_name_label} and mean tmax shift",
            # xlabel="tmax mean(1986:2021) - mean(1950:1985)\nanomalies wrt 1960-1985 (C)",
            # ylabel=f"sample {y_name_label} over 1960-1985",
            # clabel=f"heatwave cumulative intensity (°C anom)\nmean({new_years[0]}:{new_years[1]}) - mean({ref_years[0]}:{ref_years[1]})",
            clim=(1, 21),
            # min_count=10,
        )
        .redim.values(  # hack to order the bins
            **bin_ordering_dict
        )
        .opts(cticks=np.linspace(1, 21, 6), xrotation=45)
    ) * markers

    if use_qbins:
        figlist = [
            fig_hwf,
            fig_hwd,
            fig_sumheat,
            fig_count,
        ]
    else:
        figlist = [
            fig_count,
            fig_hwf,
            fig_hwd,
            fig_sumheat,
        ]
    return figlist


def get_scatter(
    deg_df,
    x_var,
    x_label,
    deg,
    ref_years,
    new_years,
    size=5,
    alpha_pt=0.02,
    ylim_hwf=(-5, 25),
    ylim_hwd=(-5, 10),
    # ylim_avi=(-2, 3),
    ylim_sumheat=(-5, 70),
    color_pt="red",
    color_line="red",
    label_curve="",
):
    fig_hwf_scatter = deg_df.hvplot.scatter(
        x=x_var,
        y="t2m_x.t2m_x_threshold.HWF",
        c="t2m_x_mean_diff",
        s=size,
        alpha=alpha_pt,
        # cmap=reds_discrete,
        color=color_pt,
    ).opts(
        xlabel=f"climatological {x_label}",
        ylabel="change in HWF (days)",
        title=f"obs, filtered to {deg - 1}.75 < mean tmax change < {deg}.25\nmean({new_years[0]}:{new_years[1]}) - mean({ref_years[0]}:{ref_years[1]})",
        # width=600,
        # height=400,
        ylim=ylim_hwf,
    )

    hwf_fitted = sm.nonparametric.lowess(
        exog=deg_df[x_var], endog=deg_df["t2m_x.t2m_x_threshold.HWF"], frac=2 / 3
    )
    # eval_x = np.linspace(deg_df[x_var].min(), deg_df[x_var].max(), num=500)
    # hwf_fitted, hwf_bottom, hwf_top = lowess_with_confidence_bounds(
    #     deg_df[x_var],
    #     deg_df["t2m_x.t2m_x_threshold.HWF"],
    #     eval_x,
    #     lowess_kw={"frac": 2 / 3},
    # )
    # fig_hwf_ci = hv.Area(
    #     x=eval_x, y=hwf_bottom, y2=hwf_top, alpha=0.3, color=color_ci#, label="Uncertainty"
    # )
    fig_hwf_fitted = hv.Curve(
        zip(hwf_fitted[:, 0], hwf_fitted[:, 1]), label=label_curve
    ).opts(color=color_line)
    # make figure
    fig_hwf = fig_hwf_scatter * fig_hwf_fitted

    # hwd ---------------------------------------------
    fig_hwd_scatter = deg_df.hvplot.scatter(
        x=x_var,
        y="t2m_x.t2m_x_threshold.HWD",
        c="t2m_x_mean_diff",
        s=size,
        alpha=alpha_pt,
        # cmap=reds_discrete,
        color=color_pt,
    ).opts(
        xlabel=f"climatological {x_label}",
        ylabel="change in HWD (days)",
        title=f"obs, filtered to {deg - 1}.75 < mean tmax change < {deg}.25\nmean({new_years[0]}:{new_years[1]}) - mean({ref_years[0]}:{ref_years[1]})",
        # width=600,
        # height=400,
        ylim=ylim_hwd,
    )
    hwd_fitted = sm.nonparametric.lowess(
        exog=deg_df[x_var], endog=deg_df["t2m_x.t2m_x_threshold.HWD"], frac=2 / 3
    )
    fig_hwd_fitted = hv.Curve(
        zip(hwd_fitted[:, 0], hwd_fitted[:, 1]), label=label_curve
    ).opts(color=color_line)
    fig_hwd = fig_hwd_scatter * fig_hwd_fitted

    # sumheat ---------------------------------------------
    fig_sumheat_scatter = deg_df.hvplot.scatter(
        x=x_var,
        y="t2m_x.t2m_x_threshold.sumHeat",
        c="t2m_x_mean_diff",
        s=size,
        alpha=alpha_pt,
        # cmap=reds_discrete,
        color=color_pt,
    ).opts(
        xlabel=f"climatological {x_label}",
        ylabel="change in sumHeat (C)",
        title=f"obs, filtered to {deg - 1}.75 < mean tmax change < {deg}.25\nmean({new_years[0]}:{new_years[1]}) - mean({ref_years[0]}:{ref_years[1]})",
        # ylim=(-5, 25),
        ylim=ylim_sumheat,
        # width=600,
        # height=400,
    )
    sumheat_fitted = sm.nonparametric.lowess(
        exog=deg_df[x_var], endog=deg_df["t2m_x.t2m_x_threshold.sumHeat"], frac=2 / 3
    )
    fig_sumheat_fitted = hv.Curve(
        zip(sumheat_fitted[:, 0], sumheat_fitted[:, 1]), label=label_curve
    ).opts(color=color_line)
    fig_sumheat = fig_sumheat_scatter * fig_sumheat_fitted

    figlist = [
        fig_hwf,
        fig_hwd,
        # fig_avi,
        fig_sumheat,
    ]
    return hv.Layout(figlist).cols(1)
