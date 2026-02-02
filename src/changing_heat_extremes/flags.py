"""
Define the following parameters, which will be used downstream by all of the analysis scripts

- use_calendar_summer: boolean
    - true uses JJA as summer in the northern hemisphere and DJF in southern
    - false uses 90 days surrounding hottest climatological day, ala mckinnon et al 2024

- ref_years
    - list of 2 years.
    - period used to define heatwave metrics in the "base period"
    - tested option: [1960, 1990]

- new_years
    - list of 2 years.
    - gap between the two years should probably match ref_years
    - tested option: [1995, 2025]

- percentile_threshold
    - threshold used to define extreme heat at each gridcell
    - tested options: 0.9, 0.95

- hw_def:
    - heatwave definitions, as in the HDP package.
    - tested options: "3-0-0", "2-0-0", "6-0-0"
    - eg. 3-0-0 means a heatwave is defined as 3+ consecutive days of extreme heat

- label:
    - string
    - used to label your choice of flags! Will show up in saved file names.


note: if you try an option and it doesn't work, look at modifying 0_era_meanshift.py

note: if you only care about `percentile_threshold` and the `hw_def` args, it might be faster for you to use the built-in vectorization in the HDP package
    - i.e. you don't need to pass in one config at a time to the hdp functions.
    - (this is why the hw_def has an extra set of brackets)

"""

label = "q90_300"
use_calendar_summer = True  # if true, use JJA as summer. else use dayofyear mask
ref_years = [1960, 1990]  # the time period the thresholds are calculated over
new_years = [1995, 2025]  # the time period we're gonna compare to
percentile_threshold = 0.9
hw_def = [[3, 0, 0]]


##################################################
# examples of other configs
# these are in the supplement as sensitivity tests
##################################################

## sensitivity to summer definition --------------------
# label = "doy_q90_300"
# use_calendar_summer = False  # CHANGED
# ref_years = [1960, 1990]
# new_years = [1995 + 1, 2025]  # CHANGED: + 1 accounts for calendar year wrapping in my modified hdp functions
# percentile_threshold = 0.9
# hw_def = [[3,0,0]]

## sensitivity to "extreme heat" definition (i.e. quantile) --------------------------------
# label = "q95_300"
# use_calendar_summer = True
# ref_years = [1960, 1990]
# new_years = [1995, 2025]
# percentile_threshold = 0.95  # CHANGED
# hw_def = [[3,0,0]]

## sensitivity to heatwave definition --------------------------------
# label = "q90_200"
# use_calendar_summer = True
# ref_years = [1960, 1990]
# new_years = [1995, 2025]
# percentile_threshold = 0.90
# hw_def = [[2,0,0]]  # CHANGED

## sensitivity to heatwave definition --------------------------------
# label = "q90_500"
# use_calendar_summer = True
# ref_years = [1960, 1990]
# new_years = [1995, 2025]
# percentile_threshold = 0.90
# hw_def = [[5, 0, 0]]  # CHANGED
