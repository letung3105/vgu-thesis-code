# Activate the environment for running the script
if isfile("Project.toml") && isfile("Manifest.toml")
    import Pkg
    Pkg.activate(".")
end

using CSV,
    Dates,
    DataDeps,
    DataFrames,
    DiffEqFlux,
    Covid19ModelVN.Helpers,
    Covid19ModelVN.Models,
    Covid19ModelVN.TrainEval

import Covid19ModelVN.JHUCSSEData,
    Covid19ModelVN.FacebookData,
    Covid19ModelVN.PopulationData,
    Covid19ModelVN.VnExpressData,
    Covid19ModelVN.VnCdcData

const SNAPSHOTS_DIR = "snapshots"

const CACHE_DIR = ".cache"

const FPATH_VIETNAM_PROVINCES_GADM_AND_GSO_POPULATION =
    joinpath(CACHE_DIR, "average-population-by-province-vietnam.csv")

const FPATH_VIETNAM_COVID_TIMESERIES = joinpath(CACHE_DIR, "timeseries-covid19-vietnam.csv")

const FPATH_VIETNAM_INTER_PROVINCE_SOCIAL_CONNECTEDNESS =
    joinpath(CACHE_DIR, "social-connectedness-vietnam.csv")

const FPATH_VIETNAM_AVERAGE_MOVEMENT_RANGE =
    joinpath(CACHE_DIR, "movement-range-vietnam.csv")

const FPATH_HCM_CITY_AVERAGE_MOVEMENT_RANGE =
    joinpath(CACHE_DIR, "movement-range-hcm-city.csv")

const FPATH_BINH_DUONG_AVERAGE_MOVEMENT_RANGE =
    joinpath(CACHE_DIR, "movement-range-binh-duong.csv")

const FPATH_DONG_NAI_AVERAGE_MOVEMENT_RANGE =
    joinpath(CACHE_DIR, "movement-range-dong-nai.csv")

const FPATH_LONG_AN_AVERAGE_MOVEMENT_RANGE =
    joinpath(CACHE_DIR, "movement-range-long-an.csv")

function cachedata(; recreate = false)
    PopulationData.save_vietnam_province_level_gadm_and_gso_population(
        FPATH_VIETNAM_PROVINCES_GADM_AND_GSO_POPULATION,
        recreate = recreate,
    )
    JHUCSSEData.save_country_level_timeseries(
        [FPATH_VIETNAM_COVID_TIMESERIES],
        ["Vietnam"],
        recreate = recreate,
    )
    FacebookData.save_inter_province_social_connectedness(
        [FPATH_VIETNAM_INTER_PROVINCE_SOCIAL_CONNECTEDNESS],
        ["VNM"],
        recreate = recreate,
    )
    FacebookData.save_region_average_movement_range(
        [
            FPATH_VIETNAM_AVERAGE_MOVEMENT_RANGE,
            FPATH_HCM_CITY_AVERAGE_MOVEMENT_RANGE,
            FPATH_BINH_DUONG_AVERAGE_MOVEMENT_RANGE,
            FPATH_DONG_NAI_AVERAGE_MOVEMENT_RANGE,
            FPATH_LONG_AN_AVERAGE_MOVEMENT_RANGE,
        ],
        ["VNM", "VNM", "VNM", "VNM", "VNM"],
        [nothing, 26, 10, 2, 39],
        recreate = recreate,
    )
end

function get_model_callables(model, train_dataset, test_dataset, vars)
    predict_fn = Predictor(model.problem)
    train_loss_fn = Loss(rmsle, predict_fn, train_dataset, vars)
    test_loss_fn = Loss(rmsle, predict_fn, test_dataset, vars)
    return predict_fn, train_loss_fn, test_loss_fn
end

function train_and_evaluate_experiment(
    exp_name,
    model,
    train_dataset,
    test_dataset,
    sessions,
    snapshots_dir,
    eval_forecast_ranges,
    eval_vars,
    eval_labels,
)
    predict_fn, train_loss_fn, test_loss_fn =
        get_model_callables(model, train_dataset, test_dataset, eval_vars)
    p0 = get_model_initial_params(model)

    @info "Initial training loss: $(train_loss_fn(p0))"
    @info "Initial testing loss: $(test_loss_fn(p0))"

    # create containing folder if not exists
    exp_dir = joinpath(snapshots_dir, exp_name)
    if !isdir(exp_dir)
        mkpath(exp_dir)
    end

    @info "Start training '$exp_name'"
    train_model(train_loss_fn, test_loss_fn, p0, sessions, exp_dir)

    @info "Ploting evaluations for '$exp_name'"
    evaluate_model(
        predict_fn,
        train_dataset,
        test_dataset,
        exp_dir,
        eval_forecast_ranges,
        eval_vars,
        eval_labels,
    )

    return nothing
end

"""
Setup different experiement scenarios for Vietnam country-wide data

# Arguments

* `exp_name`: name of the preset experiment
* `datasets_dir`: paths to the folder where newly created datasets are contained
* `fb_movement_range_fpath`: paths to the Facebook movement range data file
* `recreate=false`: true if we want to create a new file when one already exists
"""
function setup_experiment_preset_vietnam(exp_name)
    # train for 1 month
    train_range = Day(31)
    # forecast upto 4-week
    forecast_range = Day(28)

    # load covid cases data
    df_covid_timeseries = CSV.read(FPATH_VIETNAM_COVID_TIMESERIES, DataFrame)
    subset!(df_covid_timeseries, :date => x -> x .>= Date(2021, 4, 27))

    # first date that total number of confirmed cases passed 500
    first_date = first(filter(x -> x.confirmed_total >= 500, df_covid_timeseries)).date
    split_date = first_date + train_range
    last_date = first_date + train_range + forecast_range

    @info "First date: $first_date"

    # ma7
    covid_timeseries_cols = [:infective, :recovered_total, :deaths_total, :confirmed_total]
    moving_average!(df_covid_timeseries, covid_timeseries_cols, 7)
    # separate dataframe into data arrays for train and test
    train_dataset, test_dataset = train_test_split(
        df_covid_timeseries,
        covid_timeseries_cols,
        :date,
        first_date,
        split_date,
        last_date,
    )

    # load facebook movement range
    df_movement_range = CSV.read(FPATH_VIETNAM_AVERAGE_MOVEMENT_RANGE, DataFrame)
    movement_range_cols =
        [:all_day_bing_tiles_visited_relative_change, :all_day_ratio_single_tile_users]
    moving_average!(df_movement_range, movement_range_cols, 7)
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
    end

    return model, train_dataset, test_dataset
end

function setup_experiment_preset_vietnam_province(exp_name)
    # train for 1 month
    train_range = Day(31)
    # forecast upto 4-week
    forecast_range = Day(28)

    df_population = CSV.read(FPATH_VIETNAM_PROVINCES_GADM_AND_GSO_POPULATION, DataFrame)
    function get_province_population(province_name)
        province = first(filter(x -> x.NAME_1 == province_name, df_population))
        return province.AVGPOPULATION
    end

    function load_covid_data(fpath, population)
        # load covid cases data
        df_covid_timeseries = VnCdcData.read_timeseries_confirmed_and_deaths(fpath)
        # first date that total number of confirmed cases passed 500
        first_date = first(filter(x -> x.confirmed_total >= 500, df_covid_timeseries)).date
        split_date = first_date + train_range
        last_date = first_date + train_range + forecast_range

        @info "First date: $first_date"

        # ma7
        covid_timeseries_cols = [:deaths_total, :confirmed_total]
        moving_average!(df_covid_timeseries, covid_timeseries_cols, 7)
        # separate dataframe into data arrays for train and test
        train_dataset, test_dataset = train_test_split(
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
        df_covid_timeseries_confirmed = CSV.read(
            datadep"vnexpress/timeseries-vietnam-provinces-confirmed.csv",
            DataFrame,
        )
        df_social_connectedness =
            CSV.read(FPATH_VIETNAM_INTER_PROVINCE_SOCIAL_CONNECTEDNESS, DataFrame)
        df_spc = FacebookData.calculate_social_proximity_to_cases(
            df_population,
            df_covid_timeseries_confirmed,
            df_social_connectedness,
        )
        moving_average!(df_spc, province_name, 7)
        return load_timeseries(
            df_spc,
            province_name,
            :date,
            first_date - lag,
            first_date - lag + train_range + forecast_range,
        )
    end

    function load_movement_range(fpath, first_date, lag)
        df_movement_range = CSV.read(fpath, DataFrame)
        movement_range_cols =
            [:all_day_bing_tiles_visited_relative_change, :all_day_ratio_single_tile_users]
        moving_average!(df_movement_range, movement_range_cols, 7)
        return load_timeseries(
            df_movement_range,
            movement_range_cols,
            :ds,
            first_date - lag,
            first_date - lag + train_range + forecast_range,
        )
    end

    exp_model_type, exp_location = rsplit(exp_name, ".", limit = 2)
    province_name, fpath_covid_timeseries, fpath_movement_range = if exp_location == "hcm"
        "Hồ Chí Minh city", datadep"vncdc/HoChiMinh.json", FPATH_HCM_CITY_AVERAGE_MOVEMENT_RANGE
    elseif exp_location == "binhduong"
        "Bình Dương", datadep"vncdc/BinhDuong.json", FPATH_BINH_DUONG_AVERAGE_MOVEMENT_RANGE
    elseif exp_location == "dongnai"
        "Đồng Nai", datadep"vncdc/DongNai.json", FPATH_DONG_NAI_AVERAGE_MOVEMENT_RANGE
    elseif exp_location == "longan"
        "Long An", datadep"vncdc/LongAn.json", FPATH_LONG_AN_AVERAGE_MOVEMENT_RANGE
    else
        @error "No matching experiment"
        return
    end

    if exp_model_type == "baseline.default"
        population = get_province_population(province_name)
        u0, train_dataset, test_dataset, _ =
            load_covid_data(fpath_covid_timeseries, population)
        model = CovidModelSEIRDBaseline(u0, train_dataset.tspan)
        return model, train_dataset, test_dataset

    elseif exp_model_type == "fbmobility1.default"
        population = get_province_population(province_name)
        u0, train_dataset, test_dataset, train_first_date =
            load_covid_data(fpath_covid_timeseries, population)
        movement_range_dataset =
            load_movement_range(fpath_movement_range, train_first_date, Day(2))
        model = CovidModelSEIRDFbMobility1(u0, train_dataset.tspan, movement_range_dataset)
        return model, train_dataset, test_dataset

    elseif exp_model_type == "fbmobility2.default"
        population = get_province_population(province_name)
        u0, train_dataset, test_dataset, train_first_date =
            load_covid_data(fpath_covid_timeseries, population)
        movement_range_dataset =
            load_movement_range(fpath_movement_range, train_first_date, Day(2))
        scp_index =
            load_social_proximity_to_cases_index(province_name, train_first_date, Day(2))
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
    vn_experiments = [],
    # "baseline.default.hcm",
    # "fbmobility1.default.hcm",
    # "fbmobility2.default.hcm",
    vn_province_experiments = [],
)
    for exp_name ∈ vn_experiments
        timestamp = Dates.format(now(), "yyyymmddHHMMSS")
        sessions = [
            TrainSession("$timestamp.adam", ADAM(1e-2), 10),
            TrainSession("$timestamp.lbfgs", LBFGS(), 10),
        ]

        model, train_dataset, test_dataset = setup_experiment_preset_vietnam(exp_name)

        train_and_evaluate_experiment(
            exp_name,
            model,
            train_dataset,
            test_dataset,
            sessions,
            SNAPSHOTS_DIR,
            [7, 14, 21, 28],
            3:6,
            ["infective" "recovered" "deaths" "total confirmed"],
        )
    end

    for exp_name ∈ vn_province_experiments
        timestamp = Dates.format(now(), "yyyymmddHHMMSS")
        sessions = [
            TrainSession("$timestamp.adam", ADAM(1e-3), 10),
            TrainSession("$timestamp.lbfgs", LBFGS(), 10),
        ]

        model, train_dataset, test_dataset =
            setup_experiment_preset_vietnam_province(exp_name)
        train_and_evaluate_experiment(
            exp_name,
            model,
            train_dataset,
            test_dataset,
            sessions,
            SNAPSHOTS_DIR,
            [7, 14, 21, 28],
            5:6,
            ["deaths" "total confirmed"],
        )
    end
end

cachedata(recreate = true)

main(
    ["baseline.default.vietnam", "fbmobility1.default.vietnam"],
    [
        "baseline.default.hcm",
        "fbmobility1.default.hcm",
        "fbmobility2.default.hcm",
        "baseline.default.binhduong",
        "fbmobility1.default.binhduong",
        "fbmobility2.default.binhduong",
        "baseline.default.dongnai",
        "fbmobility1.default.dongnai",
        "fbmobility2.default.dongnai",
        "baseline.default.longan",
        "fbmobility1.default.longan",
        "fbmobility2.default.longan",
    ],
)

# main([
# "baseline.default.vietnam",
# ])