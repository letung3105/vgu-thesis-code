module Datasets

export TimeseriesDataset, load_covid_cases_datasets, load_fb_movement_range

using Dates, DataFrames

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
function TimeseriesDataset(df::DataFrame, cols, first_date::Date, last_date::Date; timeoffset=0)
    df = filter(x -> x.date >= first_date && x.date <= last_date, df)
    data = Float64.(Array(df[!, cols])')
    tspan = Float64.((0, Dates.value(last_date - first_date) + timeoffset))
    tsteps = (tspan[1] + timeoffset):1:tspan[2]
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
function load_covid_cases_datasets(df::DataFrame, cols, train_first_date::Date, train_range::Day, forecast_range::Day)
    train_last_date = train_first_date + train_range
    test_first_date = train_last_date + Day(1)
    test_last_date = test_first_date + forecast_range
    train_dataset = TimeseriesDataset(df, cols, train_first_date, train_last_date)
    test_dataset = TimeseriesDataset(df, cols, test_first_date, test_last_date, timeoffset=train_dataset.tspan[2] + 1)
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
function load_fb_movement_range(df::DataFrame, cols, train_first_date::Date, train_range::Day, forecast_range::Day, delay::Day)
    first_date = train_first_date - delay
    last_date = train_first_date + train_range + forecast_range - delay
    df = filter!(x -> x.ds >= first_date && x.ds <= last_date, df)
    return Array(df[!, Cols(cols)])
end

end