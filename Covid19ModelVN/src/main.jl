# Activate the environment for running the script
if isfile("Project.toml") && isfile("Manifest.toml")
    import Pkg
    Pkg.activate(".")
end

using Dates,
    DiffEqFlux,
    Serialization,
    Covid19ModelVN.Models,
    Covid19ModelVN.Helpers,
    Covid19ModelVN.Datasets

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
    fb_movement_range_fpath::AbstractString,
    datasets_dir::AbstractString;
    recreate = false,
)
    DEFAULT_POPULATION = 97_582_700
    DEFAULT_TRAIN_FIRST_DATE = Date(2021, 5, 14) # first date where total cases >= 500
    DEFAULT_TRAIN_RANGE = Day(31) # roughly 1 month
    DEFAULT_FORECAST_RANGE = Day(28) # for 7-day, 14-day, and 28-day forecasts
    DEFAULT_MOVEMENT_RANGE_DELAY = Day(2) # incubation day is roughly 2 days
    DEFAULT_MOVEMENT_RANGE_MA = 1 # no moving average

    df_cases_timeseries =
        DEFAULT_VIETNAM_COVID_DATA_TIMESERIES(datasets_dir, recreate = recreate)
    df_fb_movement_range =
        DEFAULT_VIETNAM_AVERAGE_MOVEMENT_RANGE(datasets_dir, recreate = recreate)

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
            DEFAULT_MOVEMENT_RANGE_MA,
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
            DEFAULT_MOVEMENT_RANGE_MA,
        )
    elseif exp_name == "fbmobility.ma7movementrange.default.vietnam"
        setup_seird_fb_movement_range(
            df_cases_timeseries,
            df_fb_movement_range,
            DEFAULT_POPULATION,
            DEFAULT_TRAIN_FIRST_DATE,
            DEFAULT_TRAIN_RANGE,
            DEFAULT_FORECAST_RANGE,
            DEFAULT_MOVEMENT_RANGE_DELAY,
            7,
        )
    elseif exp_name == "fbmobility.ma7movementrange.4daydelay.vietnam"
        setup_seird_fb_movement_range(
            df_cases_timeseries,
            df_fb_movement_range,
            DEFAULT_POPULATION,
            DEFAULT_TRAIN_FIRST_DATE,
            DEFAULT_TRAIN_RANGE,
            DEFAULT_FORECAST_RANGE,
            Day(4),
            7,
        )
    end
end


function train_and_evaluate_experiment_preset_vietnam(
    exp_name::AbstractString,
    fb_movement_range_fpath::AbstractString,
    datasets_dir::AbstractString,
    snapshots_dir::AbstractString,
)
    model, train_dataset, test_dataset = setup_experiment_preset_vietnam(
        exp_name,
        fb_movement_range_fpath,
        datasets_dir,
        recreate = false,
    )

    predict_fn = Predictor(model.problem)
    train_loss_fn = Loss(mse, predict_fn, train_dataset, 3:6)
    test_loss_fn = Loss(mse, predict_fn, test_dataset, 3:6)
    p0 = get_model_initial_params(model)

    @info train_loss_fn(p0)
    @info test_loss_fn(p0)

    # create containing folder if not exists
    exp_dir = joinpath(snapshots_dir, exp_name)
    if !isdir(exp_dir)
        mkpath(exp_dir)
    end

    timestamp = Dates.format(now(), "yyyymmddHHMMSS")
    sessions = [
        TrainSession("$timestamp.adam", ADAM(1e-2), 1000, exp_dir, exp_dir),
        TrainSession(
            "$timestamp.bfgs",
            BFGS(initial_stepnorm = 1e-2),
            1000,
            exp_dir,
            exp_dir,
        ),
    ]

    @info "Start training"
    train_model(train_loss_fn, test_loss_fn, p0, sessions)

    @info "Ploting evaluations"
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
                ["infective" "recoveries" "deaths" "total cases"],
            )
            savefig(
                plt,
                joinpath(snapshots_dir, exp_name, "$uuid.evaluate.forecasts.mape.png"),
            )
        end
    end

    return nothing
end

experiment_names = [
    "baseline.default.vietnam",
    "fbmobility.default.vietnam",
    "fbmobility.4daydelay.vietnam",
    "fbmobility.ma7movementrange.default.vietnam",
    "fbmobility.ma7movementrange.default.vietnam",
]

for exp_name in experiment_names
    train_and_evaluate_experiment_preset_vietnam(
        exp_name,
        joinpath(
            "datasets",
            "facebook",
            "movement-range-data-2021-10-09",
            "movement-range-2021-10-09.txt",
        ),
        "datasets",
        "snapshots",
    )
end
