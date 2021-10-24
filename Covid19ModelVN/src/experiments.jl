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
        TrainSession("$timestamp.lbfgs", LBFGS(), 1000, exp_dir, exp_dir),
    ]

    @info "Start training"
    train_model(train_loss_fn, test_loss_fn, p0, sessions)

    @info "Ploting evaluations"
    evaluate_model(
        exp_name,
        predict_fn,
        train_dataset,
        test_dataset,
        snapshots_dir,
        eval_forecast_ranges,
        eval_vars,
        eval_labels,
    )

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
    # train for 1 month
    train_range = Day(31)
    # forecast upto 4-week
    forecast_range = Day(28)

    # load covid cases data
    df_covid_timeseries = DEFAULT_VIETNAM_COVID_DATA_TIMESERIES(datasets_dir)
    # first date that total number of confirmed cases passed 500
    first_date = first(filter(x -> x.confirmed_total >= 500, df_covid_timeseries)).date
    split_date = first_date + train_range
    last_date = first_date + train_range + forecast_range

    @info "First date: $first_date"

    # ma7
    covid_timeseries_cols = [:infective, :recovered_total, :dead_total, :confirmed_total]
    Datasets.moving_average!(df_covid_timeseries, covid_timeseries_cols, 7)
    # separate dataframe into data arrays for train and test
    train_dataset, test_dataset = Datasets.train_test_split(
        df_covid_timeseries,
        covid_timeseries_cols,
        :date,
        first_date,
        split_date,
        last_date,
    )

    # load facebook movement range
    df_movement_range = DEFAULT_VIETNAM_AVERAGE_MOVEMENT_RANGE(datasets_dir)
    movement_range_cols =
        [:all_day_bing_tiles_visited_relative_change, :all_day_ratio_single_tile_users]
    Datasets.moving_average!(df_movement_range, movement_range_cols, 7)
    # load timeseries data with the chosen temporal lag
    load_movement_range(lag) = load_timeseries(
        df_movement_range,
        movement_range_cols,
        :ds,
        first_date - lag,
        last_date - lag,
    )

    # Vietnam population from GSO (https://gso.gov.vn)
    population = 97_582_700
    I0 = train_dataset.data[1, 1] # infective individuals
    R0 = train_dataset.data[2, 1] # recovered individuals
    D0 = train_dataset.data[3, 1] # deaths
    C0 = train_dataset.data[4, 1] # total confirmed cases
    N0 = population - D0 # effective population
    E0 = I0 * 2 # exposed individuals
    S0 = population - C0 - E0 # susceptible individuals
    # initial states
    u0 = [S0, E0, I0, R0, D0, C0, N0]

    model = if exp_name == "baseline.default.vietnam"
        CovidModelSEIRDBaseline(u0, train_dataset.tspan)
    elseif exp_name == "fbmobility1.default.vietnam"
        movement_range_dataset = load_movement_range(Day(2))
        CovidModelSEIRDFbMobility1(u0, train_dataset.tspan, movement_range_dataset)
    elseif exp_name == "fbmobility1.4daylag.vietnam"
        movement_range_dataset = load_movement_range(Day(4))
        CovidModelSEIRDFbMobility1(u0, train_dataset.tspan, movement_range_dataset)
    end

    return model, train_dataset, test_dataset
end

function setup_experiment_preset_vietnam_province(
    exp_name::AbstractString,
    datasets_dir::AbstractString,
)
    # train for 1 month
    train_range = Day(31)
    # forecast upto 4-week
    forecast_range = Day(28)

    function get_province_id_and_population(province_name)
        df_vn_gadm1_population = DEFAULT_VIETNAM_GADM1_POPULATION_DATASET(datasets_dir)
        province = first(filter(x -> x.gadm1_name == province_name, df_vn_gadm1_population))
        return province.gadm1_id, province.avg_population
    end

    function load_covid_data(dataset_name, population)
        # load covid cases data
        df_covid_timeseries = DEFAULT_VIETNAM_PROVINCE_CONFIRMED_AND_DEATHS_TIMESERIES(
            datasets_dir,
            dataset_name,
        )
        # first date that total number of confirmed cases passed 500
        first_date = first(filter(x -> x.confirmed_total >= 500, df_covid_timeseries)).date
        split_date = first_date + train_range
        last_date = first_date + train_range + forecast_range

        @info "First date: $first_date"

        # ma7
        covid_timeseries_cols = [:dead_total, :confirmed_total]
        Datasets.moving_average!(df_covid_timeseries, covid_timeseries_cols, 7)
        # separate dataframe into data arrays for train and test
        train_dataset, test_dataset = Datasets.train_test_split(
            df_covid_timeseries,
            covid_timeseries_cols,
            :date,
            first_date,
            split_date,
            last_date,
        )

        D0 = train_dataset.data[1, 1] # total deaths
        C0 = train_dataset.data[2, 1] # total confirmed
        N0 = population - D0 # effective population
        I0 = div(C0 - D0, 2) # infective individuals
        R0 = C0 - I0 - D0 # recovered individuals
        E0 = I0 * 2 # exposed individuals
        S0 = population - C0 - E0 # susceptible individuals
        # initial state
        u0 = [S0, E0, I0, R0, D0, C0, N0]

        return u0, train_dataset, test_dataset, first_date
    end

    function load_social_proximity_to_cases_index(province_name, first_date, lag)
        df_vn_spc = DEFAULT_VIETNAM_SOCIAL_PROXIMITY_TO_CASES_INDEX(datasets_dir)
        Datasets.moving_average!(df_vn_spc, province_name, 7)
        return load_timeseries(
            df_vn_spc,
            province_name,
            :date,
            first_date - lag,
            first_date - lag + train_range + forecast_range,
        )
    end

    function load_movement_range(province_id, first_date, lag)
        df_movement_range =
            DEFAULT_VIETNAM_PROVINCE_AVERAGE_MOVEMENT_RANGE(datasets_dir, province_id)
        movement_range_cols =
            [:all_day_bing_tiles_visited_relative_change, :all_day_ratio_single_tile_users]
        Datasets.moving_average!(df_movement_range, movement_range_cols, 7)
        return load_timeseries(
            df_movement_range,
            movement_range_cols,
            :ds,
            first_date - lag,
            first_date - lag + train_range + forecast_range,
        )
    end

    exp_model_type, exp_location = rsplit(exp_name, ".", limit = 2)
    province_name, dataset_name = if exp_location == "hcm"
        "Hồ Chí Minh city", "HoChiMinh"
    elseif exp_location == "binhduong"
        "Bình Dương", "BinhDuong"
    elseif exp_location == "dongnai"
        "Đồng Nai", "DongNai"
    elseif exp_location == "longan"
        "Long An", "LongAn"
    else
        @error "No matching experiment"
        return
    end

    if exp_model_type == "baseline.default"
        _, population = get_province_id_and_population(province_name)
        u0, train_dataset, test_dataset, _ =
            load_covid_data(dataset_name, population)
        model = CovidModelSEIRDBaseline(u0, train_dataset.tspan)
        return model, train_dataset, test_dataset

    elseif exp_model_type == "fbmobility1.default"
        province_id, population = get_province_id_and_population(province_name)
        u0, train_dataset, test_dataset, train_first_date =
            load_covid_data(dataset_name, population)
        movement_range_dataset =
            load_movement_range(province_id, train_first_date, Day(2))
        model = CovidModelSEIRDFbMobility1(u0, train_dataset.tspan, movement_range_dataset)
        return model, train_dataset, test_dataset

    elseif exp_model_type == "fbmobility2.default"
        province_id, population = get_province_id_and_population(province_name)
        u0, train_dataset, test_dataset, train_first_date =
            load_covid_data(dataset_name, population)
        movement_range_dataset =
            load_movement_range(province_id, train_first_date, Day(2))
        scp_index = load_social_proximity_to_cases_index(
            province_name,
            train_first_date,
            Day(2),
        )
        model = CovidModelSEIRDFbMobility2(
            u0,
            train_dataset.tspan,
            movement_range_dataset,
            scp_index,
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
            [7, 14, 21, 28],
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
            [7, 14, 21, 28],
            5:6,
            ["dead" "total confirmed"],
        )
    end
end

# main([
# "baseline.default.vietnam",
# "fbmobility1.default.vietnam",
# "fbmobility1.4daydelay.vietnam",
# ], [
# "baseline.default.hcm",
# "fbmobility1.default.hcm",
# "fbmobility2.default.hcm",
# "baseline.default.binhduong",
# "fbmobility1.default.binhduong",
# "fbmobility2.default.binhduong",
# "baseline.default.dongnai",
# "fbmobility1.default.dongnai",
# "fbmobility2.default.dongnai",
# ])

# main([
# "baseline.default.vietnam",
# ])