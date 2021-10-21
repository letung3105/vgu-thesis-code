# Activate the environment for running the script
if isfile("Project.toml") && isfile("Manifest.toml")
    import Pkg
    Pkg.activate(".")
end

using Dates,
    Serialization,
    Plots,
    DiffEqFlux,
    Covid19ModelVN.Models,
    Covid19ModelVN.Helpers,
    Covid19ModelVN.Datasets

const DEFAULT_DATASETS_DIR = "datasets"
const DEFAULT_SNAPSHOTS_DIR = "snapshots"

function train_and_evaluate_experiment(
    exp_name::AbstractString,
    model,
    train_dataset::TimeseriesDataset,
    test_dataset::TimeseriesDataset,
    snapshots_dir::AbstractString,
    eval_forecast_ranges::AbstractVector{Int},
    eval_vars::Union{Int,AbstractVector{Int},OrdinalRange},
    eval_labels::AbstractArray{<:AbstractString},
)
    predict_fn = Predictor(model.problem)
    train_loss_fn = Loss(rmse, predict_fn, train_dataset, eval_vars)
    test_loss_fn = Loss(rmse, predict_fn, test_dataset, eval_vars)
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
        TrainSession("$timestamp.adam", ADAM(1e-2), 10000, exp_dir, exp_dir),
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
                eval_forecast_ranges,
                eval_vars,
                eval_labels,
            )
            savefig(
                plt,
                joinpath(snapshots_dir, exp_name, "$uuid.evaluate.forecasts.mape.png"),
            )
        end
    end

    return nothing
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
)
    df_covid_timeseries = DEFAULT_VIETNAM_COVID_DATA_TIMESERIES(datasets_dir)

    train_first_date =
        first(filter(x -> x.confirmed_total >= 500, df_covid_timeseries)).date
    train_range = Day(31) # roughly 1 month
    forecast_range = Day(28) # for 7-day, 14-day, and 28-day forecasts

    train_dataset, test_dataset = load_covid_cases_datasets(
        df_covid_timeseries,
        [:infective, :recovered_total, :dead_total, :confirmed_total],
        train_first_date,
        train_range,
        forecast_range,
    )

    population = 97_582_700
    u0 = [
        population - train_dataset.data[4, 1] - train_dataset.data[1, 1] * 2,
        train_dataset.data[1, 1] * 2,
        train_dataset.data[1, 1],
        train_dataset.data[2, 1],
        train_dataset.data[3, 1],
        train_dataset.data[4, 1],
        population - train_dataset.data[3, 1],
    ]

    load_experiment_movement_range(delay::Day, moving_average_days::Int) =
        load_fb_movement_range(
            DEFAULT_VIETNAM_AVERAGE_MOVEMENT_RANGE(datasets_dir),
            train_first_date,
            train_range,
            forecast_range,
            delay,
            moving_average_days,
        )

    model = if exp_name == "baseline.default.vietnam"
        CovidModelSEIRDBaseline(u0, train_dataset.tspan)
    elseif exp_name == "fbmobility1.default.vietnam"
        movement_range_dataset = load_experiment_movement_range(Day(2), 1)
        CovidModelSEIRDFbMobility1(u0, train_dataset.tspan, movement_range_dataset)
    elseif exp_name == "fbmobility1.4daydelay.vietnam"
        movement_range_dataset = load_experiment_movement_range(Day(4), 1)
        CovidModelSEIRDFbMobility1(u0, train_dataset.tspan, movement_range_dataset)
    elseif exp_name == "fbmobility1.ma7movementrange.default.vietnam"
        movement_range_dataset = load_experiment_movement_range(Day(2), 7)
        CovidModelSEIRDFbMobility1(u0, train_dataset.tspan, movement_range_dataset)
    elseif exp_name == "fbmobility1.ma7movementrange.4daydelay.vietnam"
        movement_range_dataset = load_experiment_movement_range(Day(4), 7)
        CovidModelSEIRDFbMobility1(u0, train_dataset.tspan, movement_range_dataset)
    end

    return model, train_dataset, test_dataset
end

function setup_experiment_preset_vietnam_province(
    exp_name::AbstractString,
    datasets_dir::AbstractString,
)
    train_range = Day(31) # roughly 1 month
    forecast_range = Day(28) # for 7-day, 14-day, and 28-day forecasts

    function get_province_id_and_population(province_name)
        df_vn_gadm1_population = DEFAULT_VIETNAM_GADM1_POPULATION_DATASET(datasets_dir)
        province = first(filter(x -> x.gadm1_name == province_name, df_vn_gadm1_population))
        return province.gadm1_id, province.avg_population
    end

    function load_experiment_covid_data(dataset_name, population)
        df_covid_timeseries = DEFAULT_VIETNAM_PROVINCE_CONFIRMED_AND_DEATHS_TIMESERIES(
            datasets_dir,
            dataset_name,
        )

        train_first_date =
            first(filter(x -> x.confirmed_total >= 500, df_covid_timeseries)).date

        train_dataset, test_dataset = load_covid_cases_datasets(
            df_covid_timeseries,
            [:dead_total, :confirmed_total],
            train_first_date,
            train_range,
            forecast_range,
        )

        # dead
        D0 = train_dataset.data[1, 1]
        # confirmed total
        C0 = train_dataset.data[2, 1]
        # effective population
        N0 = population - D0
        # infective assumed to be 1/2 of total confirmed
        I0 = div(C0 - D0, 2)
        # recovered derived from I, D, and C
        R0 = C0 - I0 - D0
        # exposed assumed to be 2x infectives
        E0 = I0 * 2
        # susceptible
        S0 = population - C0 - E0
        # initial state
        u0 = [S0, E0, I0, R0, D0, C0, N0]

        return u0, train_dataset, test_dataset, train_first_date
    end

    load_experiment_social_proximity_to_cases_index(
        province_name,
        train_first_date,
        delay,
    ) = load_social_proximity_to_cases_index(
        DEFAULT_VIETNAM_SOCIAL_PROXIMITY_TO_CASES_INDEX(datasets_dir),
        province_name,
        train_first_date,
        train_range,
        forecast_range,
        delay,
    )

    load_experiment_movement_range(
        province_id,
        train_first_date,
        delay,
        moving_average_days,
    ) = load_fb_movement_range(
        DEFAULT_VIETNAM_PROVINCE_AVERAGE_MOVEMENT_RANGE(datasets_dir, province_id),
        train_first_date,
        train_range,
        forecast_range,
        delay,
        moving_average_days,
    )

    if exp_name == "baseline.default.hcm"
        _, population = get_province_id_and_population("Hồ Chí Minh city")
        u0, train_dataset, test_dataset, _ =
            load_experiment_covid_data("HoChiMinh", population)
        model = CovidModelSEIRDBaseline(u0, train_dataset.tspan)
        return model, train_dataset, test_dataset
    elseif exp_name == "fbmobility1.default.hcm"
        province_id, population = get_province_id_and_population("Hồ Chí Minh city")
        u0, train_dataset, test_dataset, train_first_date =
            load_experiment_covid_data("HoChiMinh", population)
        movement_range_dataset =
            load_experiment_movement_range(province_id, train_first_date, Day(2), 1)
        model = CovidModelSEIRDFbMobility1(u0, train_dataset.tspan, movement_range_dataset)
        return model, train_dataset, test_dataset
    elseif exp_name == "fbmobility2.default.hcm"
        province_id, population = get_province_id_and_population("Hồ Chí Minh city")
        u0, train_dataset, test_dataset, train_first_date =
            load_experiment_covid_data("HoChiMinh", population)
        movement_range_dataset =
            load_experiment_movement_range(province_id, train_first_date, Day(2), 1)
        social_proximity_to_cases_index = load_experiment_social_proximity_to_cases_index(
            "Hồ Chí Minh city",
            train_first_date,
            Day(2),
        )
        model = CovidModelSEIRDFbMobility2(
            u0,
            train_dataset.tspan,
            movement_range_dataset,
            social_proximity_to_cases_index,
        )
        return model, train_dataset, test_dataset
    end

    return nothing
end

function main(
    # "baseline.default.vietnam",
    # "fbmobility1.default.vietnam",
    # "fbmobility1.4daydelay.vietnam",
    # "fbmobility1.ma7movementrange.default.vietnam",
    # "fbmobility1.ma7movementrange.default.vietnam",
    vn_experiments = [],
    # "baseline.default.hcm",
    # "fbmobility1.default.hcm",
    # "fbmobility2.default.hcm",
    vn_gadm1_experiments = [],
)
    for exp_name in vn_experiments
        model, train_dataset, test_dataset =
            setup_experiment_preset_vietnam(exp_name, DEFAULT_DATASETS_DIR)
        train_and_evaluate_experiment(
            exp_name,
            model,
            train_dataset,
            test_dataset,
            DEFAULT_SNAPSHOTS_DIR,
            [7, 14, 28],
            3:6,
            ["infective" "recovered" "dead" "total confirmed"],
        )
    end

    for exp_name in vn_gadm1_experiments
        model, train_dataset, test_dataset =
            setup_experiment_preset_vietnam_province(exp_name, DEFAULT_DATASETS_DIR)
        train_and_evaluate_experiment(
            exp_name,
            model,
            train_dataset,
            test_dataset,
            DEFAULT_SNAPSHOTS_DIR,
            [7, 14, 28],
            5:6,
            ["dead" "total confirmed"],
        )
    end
end

main()
