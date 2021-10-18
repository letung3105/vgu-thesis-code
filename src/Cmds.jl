module Cmds

export setup_seird_baseline, setup_seird_fb_movement_range
export setup_experiment_data_vietnam, setup_experiment_preset_vietnam
export train_model_default_2steps, evaluate_model_default

using Dates, DiffEqFlux, Plots, Serialization, DataFrames
using Covid19ModelVN.Helpers, Covid19ModelVN.Datasets, Covid19ModelVN.Models

import Covid19ModelVN.FacebookData, Covid19ModelVN.VnExpressData

const DEFAULT_TRAIN_FIRST_DATE = Date(2021, 5, 14) # first date where total cases >= 500
const DEFAULT_TRAIN_RANGE = Day(31) # roughly 1 month
const DEFAULT_FORECAST_RANGE = Day(28) # for 7-day, 14-day, and 28-day forecasts
const DEFAULT_MOVEMENT_RANGE_DELAY = Day(2) # incubation day is roughly 2 days

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
    df_cases_timeseries::DataFrame;
    population::Real,
    train_first_date::Date=DEFAULT_TRAIN_FIRST_DATE, # first date where total cases >= 500
    train_range::Day=DEFAULT_TRAIN_RANGE, # roughly 1 month
    forecast_range::Day=DEFAULT_FORECAST_RANGE # for 7-day, 14-day, and 28-day forecasts
)
    train_dataset, test_dataset = load_covid_cases_datasets(
        df_cases_timeseries,
        [:infective, :recovered_total, :dead_total, :confirmed_total],
        train_first_date,
        train_range,
        forecast_range,
    )
    u0 = [
        population - train_dataset.data[4, 1] - train_dataset.data[1, 1] * 2,
        train_dataset.data[1, 1] * 2,
        train_dataset.data[1, 1],
        train_dataset.data[2, 1],
        train_dataset.data[3, 1],
        train_dataset.data[4, 1],
        population - train_dataset.data[3, 1],
    ]
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
* `movement_range_dealy`: the number of days where movement range day is delayed when given as input to
the neural network of the model
"""
function setup_seird_fb_movement_range(
    df_cases_timeseries::DataFrame,
    df_movement_range::DataFrame;
    population::Real,
    train_first_date::Date=DEFAULT_TRAIN_FIRST_DATE, # first date where total cases >= 500
    train_range::Day=DEFAULT_TRAIN_RANGE, # roughly 1 month
    forecast_range::Day=DEFAULT_FORECAST_RANGE, # for 7-day, 14-day, and 28-day forecasts
    movement_range_delay::Day=DEFAULT_MOVEMENT_RANGE_DELAY, # incubation day is roughly 2 days
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
    u0 = [
        population - train_dataset.data[4, 1] - train_dataset.data[1, 1] * 2,
        train_dataset.data[1, 1] * 2,
        train_dataset.data[1, 1],
        train_dataset.data[2, 1],
        train_dataset.data[3, 1],
        train_dataset.data[4, 1],
        population - train_dataset.data[3, 1],
    ]
    model = CovidModelSEIRDFacebookMovementRange(u0, train_dataset.tspan, movement_range_dataset)
    return model, train_dataset, test_dataset
end

"""
Get default losses figure file path

# Arguments

* `fdir::AbstractString`: the root directory of the file
* `uuid::AbstractString`: the file unique identifier
"""
get_losses_figure_path(fdir::AbstractString, uuid::AbstractString) = joinpath(fdir, "$uuid.losses.png")

"""
Get default file path for saved parameters

# Arguments

* `fdir::AbstractString`: the root directory of the file
* `uuid::AbstractString`: the file unique identifier
"""
get_params_snapshot_path(fdir::AbstractString, uuid::AbstractString) = joinpath(fdir, "$uuid.params.jls")

"""
Train the model once with `ADAM(learning_rate=1e-2)` for `adam_maxiters` then
with `BFGS(initial_stepnorm=1e-2)` for  `bfgs_maxiters`.

# Arguments

* `exp_name::AbstractString`: experiment name
* `train_loss_fn::Loss`: loss function for training
* `test_loss_fn::Loss`: loss function for testing
* `p0::AbstractVector{<:Real}`: initial parameters set
* `snapshots_dir::AbstractString`: directory to save the parameters snapshots
* `adam_maxiters`: number of iterations to run ADAM optimizer
* `bfgs_maxiters`: maximmum number of iterations to run BFGS optimizer
"""
function train_model_default_2steps(
    exp_name::AbstractString,
    train_loss_fn::Loss,
    test_loss_fn::Loss,
    p0::AbstractVector{<:Real};
    snapshots_dir::AbstractString,
    adam_maxiters=6000,
    adam_learning_rate=1e-2,
    bfgs_maxiters=1000,
    bfgs_initial_stepnorm=1e-2,
)
    timestamp = Dates.format(now(), "yyyymmddHHMMSS")
    # create containing folder if not exists
    fdir = joinpath(snapshots_dir, exp_name)
    if !isdir(fdir)
        mkpath(fdir)
    end

    @info "Running ADAM optimizer"
    res1 = let
        uuid = "$timestamp.adam"
        cb_config = TrainCallbackConfig(
            test_loss_fn,
            get_losses_figure_path(fdir, uuid),
            div(adam_maxiters, 100),
            get_params_snapshot_path(fdir, uuid),
            div(adam_maxiters, 100)
        )
        cb = TrainCallback(adam_maxiters, cb_config)
        res = DiffEqFlux.sciml_train(
            train_loss_fn, p0,
            ADAM(adam_learning_rate),
            maxiters=adam_maxiters,
            cb=cb
        )
        Serialization.serialize(get_params_snapshot_path(fdir, uuid), res.minimizer)
        res
    end

    @info "Running BFGS optimizer"
    res2 = let
        uuid = "$timestamp.bfgs"
        cb_config = TrainCallbackConfig(
            test_loss_fn,
            get_losses_figure_path(fdir, uuid),
            div(bfgs_maxiters, 100),
            get_params_snapshot_path(fdir, uuid),
            1,
        )
        cb = TrainCallback(bfgs_maxiters, cb_config)
        res = DiffEqFlux.sciml_train(
            train_loss_fn, res1.minimizer,
            BFGS(initial_stepnorm=bfgs_initial_stepnorm),
            maxiters=bfgs_maxiters,
            cb=cb
        )
        Serialization.serialize(get_params_snapshot_path(fdir, uuid), res.minimizer)
        res
    end

    return res1, res2
end

"""
Get the file paths and uuids of all the saved parameters of an experiment

# Arguments

* `snapshots_dir::AbstractString`: the directory that contains the saved parameters
* `exp_name::AbstractString`: the experiment name
"""
function lookup_saved_params(
    snapshots_dir::AbstractString,
    exp_name::AbstractString
)
    exp_dir = joinpath(snapshots_dir, exp_name)
    params_files = filter(x -> endswith(x, ".jls"), readdir(exp_dir))
    fpaths = map(f -> joinpath(snapshots_dir, exp_name, f), params_files)
    uuids = map(f -> first(rsplit(f, ".", limit=3)), params_files)
    return fpaths, uuids
end


"""
Draw and save the default set of plots for evalutating the models' performance

# Arguments

* `exp_name::AbstractString`: experiment name
* `predict_fn::Predictor`: the function for getting the model output and forecast
* `train_dataset::TimeseriesDataset`: ground truth data used for training
* `test_dataset::TimeseriesDataset`: ground truth data used for testing
* `snapshots_dir::AbstractString`: directory of the saved parameters
"""
function evaluate_model_default(
    exp_name::AbstractString,
    predict_fn::Predictor,
    train_dataset::TimeseriesDataset,
    test_dataset::TimeseriesDataset;
    snapshots_dir::AbstractString,
)
    fpaths_params, uuids = lookup_saved_params(snapshots_dir, exp_name)
    for (fpath_params, uuid) in zip(fpaths_params, uuids)
        fig_fpath = joinpath(snapshots_dir, exp_name, "$uuid.evaluate.forecasts.mape.png")
        if !isfile(fig_fpath)
            plt = plot_forecasts(
                predict_fn,
                mape,
                train_dataset,
                test_dataset,
                Serialization.deserialize(fpath_params),
                [7, 14, 28],
                3:6,
                1:4,
                ["infective" "recoveries" "deaths" "total cases"]
            )
            savefig(plt, joinpath(snapshots_dir, exp_name, "$uuid.evaluate.forecasts.mape.png"))
        end
    end
    return nothing
end

"""
Setup the required datasets for conducting experiment with Vietnam country-wide data

# Arguments

* `datasets_dir`: paths to the folder where newly created datasets are contained
* `fb_movement_range_fpath`: paths to the Facebook movement range data file
* `recreate=false`: true if we want to create a new file when one already exists
"""
function setup_experiment_data_vietnam(
    datasets_dir::AbstractString,
    fb_movement_range_fpath::AbstractString;
    recreate=false,
)
    df_cases_timeseries = VnExpressData.save_cases_timeseries(
        datasets_dir,
        "vnexpress-timeseries",
        Date(2021, 4, 27),
        Date(2021, 10, 13),
        recreate=recreate,
    )
    df_fb_movement_range = FacebookData.save_country_average_movement_range(
        fb_movement_range_fpath,
        datasets_dir,
        "facebook-average-movement-range",
        "VNM",
        recreate=recreate,
    )
    return df_cases_timeseries, df_fb_movement_range
end

"""
Setup different experiement scenarios for Vietnam country-wide data

# Arguments

* `exp_name::AbstractString`: name of the preset experiment
* `df_cases_timeseries::DataFrame`: dataframe of the cases timeseries
* `df_fb_movement_range::DataFrame`: dataframe of the Facebook movement range for models that use it
"""
function setup_experiment_preset_vietnam(
    exp_name::AbstractString,
    df_cases_timeseries::DataFrame,
    df_fb_movement_range::DataFrame
)
    population = 97_582_700
    return if exp_name == "baseline.default.vietnam"
        setup_seird_baseline(df_cases_timeseries; population=population)

    elseif exp_name == "baseline.default.60trainingdays.vietnam"
        setup_seird_baseline(df_cases_timeseries; population=population, train_range=Day(60))

    elseif exp_name == "fbmobility.default.vietnam"
        setup_seird_fb_movement_range(df_cases_timeseries, df_fb_movement_range, population=population)

    elseif exp_name == "fbmobility.4daydelay.vietnam"
        setup_seird_fb_movement_range(df_cases_timeseries, df_fb_movement_range, population=population, movement_range_delay=Day(4))

    elseif exp_name == "fbmobility.ma7movementrange.default.vietnam"
        df_fb_movement_range_ma7 = FacebookData.get_movement_range_moving_average(df_fb_movement_range)
        setup_seird_fb_movement_range(df_cases_timeseries, df_fb_movement_range_ma7, population=population)

    elseif exp_name == "fbmobility.ma7movementrange.4daydelay.vietnam"
        df_fb_movement_range_ma7 = FacebookData.get_movement_range_moving_average(df_fb_movement_range)
        setup_seird_fb_movement_range(df_cases_timeseries, df_fb_movement_range_ma7, population=population, movement_range_delay=Day(4))
    end
end

end