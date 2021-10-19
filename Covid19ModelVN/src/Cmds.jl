module Cmds

export setup_seird_baseline, setup_seird_fb_movement_range, setup_experiment_preset_vietnam

using Dates, DataFrames, Covid19ModelVN.Datasets, Covid19ModelVN.Models

import Covid19ModelVN.FacebookData, Covid19ModelVN.VnExpressData

setup_seird_initial_states(dataset::TimeseriesDataset, population::Real) = [
    population - dataset.data[4, 1] - dataset.data[1, 1] * 2,
    dataset.data[1, 1] * 2,
    dataset.data[1, 1],
    dataset.data[2, 1],
    dataset.data[3, 1],
    dataset.data[4, 1],
    population - dataset.data[3, 1],
]

"""
Prepare the datasets for training and testing, and create the baseline SEIRD model from the data

# Arguments

* `df_cases_timeseries::DataFrame`: a dataframe containing the cases timeseries data with 4 required columns:
"infective" contains the number of current infective indiduals, "recovered_total" contains the number of total
recoveries, "dead_total" contains the number of total deaths caused by the disease, "confirmed_total" contains
the total number of confirmed cases.
* `population::Real`: the total population of the area that is being modeled
* `train_first_date`: the first date where data is used for training the model
* `train_range`: number of days in the training dataset
* `forecast_range`: number of days in the testing dataset
"""
function setup_seird_baseline(
    df_cases_timeseries::DataFrame,
    population::Real,
    train_first_date::Date,
    train_range::Day,
    forecast_range::Day,
)
    train_dataset, test_dataset = load_covid_cases_datasets(
        df_cases_timeseries,
        [:infective, :recovered_total, :dead_total, :confirmed_total],
        train_first_date,
        train_range,
        forecast_range,
    )

    u0 = setup_seird_initial_states(train_dataset, population)
    model = CovidModelSEIRDBaseline(u0, train_dataset.tspan)
    return model, train_dataset, test_dataset
end

"""
Prepare the datasets for training and testing, and create the SEIRD model with Facebook movement range
from the data

# Arguments

* `df_cases_timeseries::DataFrame`: a dataframe containing the cases timeseries data with 4 required columns:
"infective" contains the number of current infective indiduals, "recovered_total" contains the number of total
recoveries, "dead_total" contains the number of total deaths caused by the disease, "confirmed_total" contains
the total number of confirmed cases
* `df_movement_range::DataFrame` a dataframe containing the movement range data for the area with 2 required
columns "all_day_bing_tiles_visited_relative_change" and "all_day_ratio_single_tile_users" as specified in the
original dataset from Facebook
* `population::Real`: the total population of the area that is being modeled
* `train_first_date`: the first date where data is used for training the model
* `train_range`: number of days in the training dataset
* `forecast_range`: number of days in the testing dataset
* `movement_range_delay`: the number of days where movement range day is delayed when given as input to
the neural network of the model
"""
function setup_seird_fb_movement_range(
    df_cases_timeseries::DataFrame,
    df_movement_range::DataFrame,
    population::Real,
    train_first_date::Date,
    train_range::Day,
    forecast_range::Day,
    movement_range_delay::Day,
)
    train_dataset, test_dataset = load_covid_cases_datasets(
        df_cases_timeseries,
        [:infective, :recovered_total, :dead_total, :confirmed_total],
        train_first_date,
        train_range,
        forecast_range,
    )
    movement_range_dataset = load_fb_movement_range(
        df_movement_range,
        [:all_day_bing_tiles_visited_relative_change, :all_day_ratio_single_tile_users],
        train_first_date,
        train_range,
        forecast_range,
        movement_range_delay,
    )

    u0 = setup_seird_initial_states(train_dataset, population)
    model = CovidModelSEIRDFacebookMovementRange(
        u0,
        train_dataset.tspan,
        movement_range_dataset,
    )
    return model, train_dataset, test_dataset
end

"""
Setup different experiement scenarios for Vietnam country-wide data

# Arguments

* `exp_name::AbstractString`: name of the preset experiment
* `datasets_dir`: paths to the folder where newly created datasets are contained
* `fb_movement_range_fpath`: paths to the Facebook movement range data file
* `recreate=false`: true if we want to create a new file when one already exists
"""
function setup_experiment_preset_vietnam(
    exp_name::AbstractString,
    datasets_dir::AbstractString,
    fb_movement_range_fpath::AbstractString;
    recreate = false,
)
    const DEFAULT_POPULATION = 97_582_700
    const DEFAULT_TRAIN_FIRST_DATE = Date(2021, 5, 14) # first date where total cases >= 500
    const DEFAULT_TRAIN_RANGE = Day(31) # roughly 1 month
    const DEFAULT_FORECAST_RANGE = Day(28) # for 7-day, 14-day, and 28-day forecasts
    const DEFAULT_MOVEMENT_RANGE_DELAY = Day(2) # incubation day is roughly 2 days

    df_cases_timeseries = VnExpressData.save_cases_timeseries(
        datasets_dir,
        "vnexpress-timeseries",
        Date(2021, 4, 27),
        Date(2021, 10, 13),
        recreate = recreate,
    )
    df_fb_movement_range = FacebookData.save_country_average_movement_range(
        fb_movement_range_fpath,
        datasets_dir,
        "facebook-average-movement-range",
        "VNM",
        recreate = recreate,
    )
    df_fb_movement_range_ma7 =
        FacebookData.get_movement_range_moving_average(df_fb_movement_range)

    return if exp_name == "baseline.default.vietnam"
        setup_seird_baseline(
            df_cases_timeseries,
            DEFAULT_POPULATION,
            DEFAULT_TRAIN_FIRST_DATE,
            DEFAULT_TRAIN_RANGE,
            DEFAULT_FORECAST_RANGE,
        )
    elseif exp_name == "fbmobility.default.vietnam"
        setup_seird_fb_movement_range(
            df_cases_timeseries,
            df_fb_movement_range,
            DEFAULT_POPULATION,
            DEFAULT_TRAIN_FIRST_DATE,
            DEFAULT_TRAIN_RANGE,
            DEFAULT_FORECAST_RANGE,
            DEFAULT_MOVEMENT_RANGE_DELAY,
        )
    elseif exp_name == "fbmobility.4daydelay.vietnam"
        setup_seird_fb_movement_range(
            df_cases_timeseries,
            df_fb_movement_range,
            DEFAULT_POPULATION,
            DEFAULT_TRAIN_FIRST_DATE,
            DEFAULT_TRAIN_RANGE,
            DEFAULT_FORECAST_RANGE,
            Day(4),
        )
    elseif exp_name == "fbmobility.ma7movementrange.default.vietnam"
        setup_seird_fb_movement_range(
            df_cases_timeseries,
            df_fb_movement_range_ma7,
            DEFAULT_POPULATION,
            DEFAULT_TRAIN_FIRST_DATE,
            DEFAULT_TRAIN_RANGE,
            DEFAULT_FORECAST_RANGE,
            DEFAULT_MOVEMENT_RANGE_DELAY,
        )
    elseif exp_name == "fbmobility.ma7movementrange.4daydelay.vietnam"
        setup_seird_fb_movement_range(
            df_cases_timeseries,
            df_fb_movement_range_ma7,
            DEFAULT_POPULATION,
            DEFAULT_TRAIN_FIRST_DATE,
            DEFAULT_TRAIN_RANGE,
            DEFAULT_FORECAST_RANGE,
            Day(4),
        )
    end
end

end
