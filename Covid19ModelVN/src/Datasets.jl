module Datasets

export TimeseriesDataset,
    load_covid_cases_datasets,
    load_fb_movement_range,
    DEFAULT_VIETNAM_GADM1_POPULATION_DATASET,
    DEFAULT_VIETNAM_COVID_DATA_TIMESERIES,
    DEFAULT_VIETNAM_PROVINCES_CONFIRMED_TIMESERIES,
    DEFAULT_VIETNAM_PROVINCES_TOTAL_CONFIRMED_TIMESERIES,
    DEFAULT_VIETNAM_AVERAGE_MOVEMENT_RANGE,
    DEFAULT_VIETNAM_INTRA_CONNECTEDNESS_INDEX,
    DEFAULT_VIETNAM_PROVINCE_CONFIRMED_AND_DEATHS_TIMESERIES,
    DEFAULT_VIETNAM_PROVINCE_AVERAGE_MOVEMENT_RANGE

using Dates, DataFrames

import Covid19ModelVN.FacebookData,
    Covid19ModelVN.VnExpressData,
    Covid19ModelVN.PopulationData,
    Covid19ModelVN.VnCdcData

DEFAULT_VIETNAM_PROVINCE_CONFIRMED_AND_DEATHS_TIMESERIES(datasets_dir, name) =
    VnCdcData.parse_json_cases_and_deaths(
        joinpath(datasets_dir, "vncdc", "$name.json"),
    )

DEFAULT_VIETNAM_PROVINCE_AVERAGE_MOVEMENT_RANGE(datasets_dir, province_id; recreate = false) =
    FacebookData.save_country_average_movement_range(
        joinpath(
            datasets_dir,
            "facebook",
            "movement-range-data-2021-10-09",
            "movement-range-2021-10-09.txt",
        ),
        datasets_dir ,
        "facebook-average-movement-range",
        "VNM",
        province_id,
        recreate = recreate,
    )


DEFAULT_VIETNAM_GADM1_POPULATION_DATASET(datasets_dir; recreate = false) =
    PopulationData.save_vietnam_gadm1_population(
        joinpath(datasets_dir, "gadm", "VNM_adm.gpkg"),
        joinpath(datasets_dir, "gso", "VNM-2020-population-all-regions.csv"),
        datasets_dir,
        "VNM-gadm1-population",
        recreate = recreate,
    )

DEFAULT_VIETNAM_COVID_DATA_TIMESERIES(datasets_dir; recreate = false) =
    VnExpressData.save_cases_timeseries(
        datasets_dir,
        "vietnam-covid-data-timeseries",
        Date(2021, 4, 27),
        Date(2021, 10, 13),
        recreate = recreate,
    )

DEFAULT_VIETNAM_PROVINCES_CONFIRMED_TIMESERIES(datasets_dir; recreate = false) =
    VnExpressData.save_provinces_confirmed_cases_timeseries(
        datasets_dir,
        "vietnam-provinces-confirmed-timeseries",
        Date(2021, 4, 27),
        Date(2021, 10, 13),
        recreate = recreate,
    )

DEFAULT_VIETNAM_PROVINCES_TOTAL_CONFIRMED_TIMESERIES(datasets_dir; recreate = false) =
    VnExpressData.save_provinces_total_confirmed_cases_timeseries(
        datasets_dir,
        "vietnam-provinces-total-confirmed-timeseries",
        Date(2021, 4, 27),
        Date(2021, 10, 13),
        recreate = recreate,
    )

DEFAULT_VIETNAM_AVERAGE_MOVEMENT_RANGE(datasets_dir; recreate = false) =
    FacebookData.save_country_average_movement_range(
        joinpath(
            datasets_dir,
            "facebook",
            "movement-range-data-2021-10-09",
            "movement-range-2021-10-09.txt",
        ),
        datasets_dir,
        "facebook-average-movement-range",
        "VNM",
        recreate = recreate,
    )

DEFAULT_VIETNAM_INTRA_CONNECTEDNESS_INDEX(datasets_dir; recreate = false) =
    FacebookData.save_intra_country_gadm1_nuts2_connectedness_index(
        joinpath(
            datasets_dir,
            "facebook",
            "social-connectedness-index",
            "gadm1_nuts2_gadm1_nuts2_aug2020.tsv",
        ),
        datasets_dir,
        "facebook-intra-connectedness-index",
        "VNM",
        recreate = recreate,
    )

"""
This contains the minimum required information for a timeseriese dataset that is used by UDEs

# Fields

* `data::AbstractArray{<:Real}`: an array that holds the timeseries data
* `tspan::Tuple{<:Real,<:Real}`: the first and last time coordinates of the timeseries data
* `tsteps::Union{Real,AbstractVector{<:Real},StepRange,StepRangeLen}`: collocations points
"""
struct TimeseriesDataset
    data::AbstractArray{<:Real}
    tspan::Tuple{<:Real,<:Real}
    tsteps::Union{Real,AbstractVector{<:Real},StepRange,StepRangeLen}
end

"""
Construct a `TimeseriesDataset` from the `DataFrame`

# Arguments

* `df::DataFrame`: the `DataFrame` that contains a timeseries
* `first_date::Date`: date of earliest data point
* `last_date::Date`: date of latest data point
* `cols`: `DataFrame` columns to consider
* `timeoffset`: offset for `tspan` and `tsteps` when creating the dataset
"""
function TimeseriesDataset(
    df::DataFrame,
    cols,
    first_date::Date,
    last_date::Date;
    timeoffset = 0,
)
    df = filter(x -> x.date >= first_date && x.date <= last_date, df)
    data = Float64.(Array(df[!, cols])')
    tspan = Float64.((0, Dates.value(last_date - first_date) + timeoffset))
    tsteps = (tspan[1]+timeoffset):1:tspan[2]
    return TimeseriesDataset(data, tspan, tsteps)
end

"""
Load the train and test datasets for the given dates

# Arguments

* `df::DataFrame`: the `DataFrame` that contains a timeseries
* `cols`: columns to consider
* `train_first_date::Date`: date of earliest data point
* `train_range::Day`: number of days in the training set
* `forecast_range::Day`: number of days in the testing set
"""
function load_covid_cases_datasets(
    df::DataFrame,
    cols,
    train_first_date::Date,
    train_range::Day,
    forecast_range::Day,
)
    train_last_date = train_first_date + train_range
    test_first_date = train_last_date + Day(1)
    test_last_date = test_first_date + forecast_range
    train_dataset = TimeseriesDataset(df, cols, train_first_date, train_last_date)
    test_dataset = TimeseriesDataset(
        df,
        cols,
        test_first_date,
        test_last_date,
        timeoffset = train_dataset.tspan[2] + 1,
    )
    return train_dataset, test_dataset
end

"""
Load the train and test datasets for the given dates

# Arguments

* `df::DataFrame`: the `DataFrame` that contains a timeseries
* `cols`: columns to consider
* `train_first_date::Date': date of earliest data point
* `train_range::Day': number of days in the training set
* `forecast_range::Day': number of days in the testing set
* `delay`: delay the data input versus the training dates and forecast dates
"""
function load_fb_movement_range(
    df::DataFrame,
    cols,
    train_first_date::Date,
    train_range::Day,
    forecast_range::Day,
    delay::Day,
    moving_average_days::Int,
)
    moving_average(xs) =
        [mean(@view xs[i-moving_average_days+1:i]) for i = moving_average_days:length(xs)]
    df = combine(
        df,
        :ds => x -> x[moving_average_days:end],
        :all_day_bing_tiles_visited_relative_change => moving_average,
        :all_day_ratio_single_tile_users => moving_average,
        renamecols = false,
    )
    first_date = train_first_date - delay
    last_date = train_first_date + train_range + forecast_range - delay
    filter!(x -> x.ds >= first_date && x.ds <= last_date, df)
    return Array(df[!, Cols(cols)])
end

end
