using CSV, Dates, DataDeps, DataFrames, DiffEqFlux, Covid19ModelVN

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

function cachedata(; recreate::Bool = false)
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

get_experiment_covid_timeseries(location_code::AbstractString) =
    if location_code == "vietnam"
        df = CSV.read(FPATH_VIETNAM_COVID_TIMESERIES, DataFrame)
        # after 4th August, the recovered count is not updated
        bound!(df, :date, Date(2021, 4, 26), Date(2021, 8, 4))
        transform!(
            df,
            :infective => x -> x .- df[1, :infective],
            :recovered_total => x -> x .- df[1, :recovered_total],
            :deaths_total => x -> x .- df[1, :deaths_total],
            :confirmed_total => x -> x .- df[1, :confirmed_total],
            renamecols = false,
        )
        bound!(df, :date, Date(2021, 4, 27), Date(2021, 8, 4))
    elseif location_code == "hcm"
        VnCdcData.read_timeseries_confirmed_and_deaths(datadep"vncdc/HoChiMinh.json")
    elseif location_code == "binhduong"
        VnCdcData.read_timeseries_confirmed_and_deaths(datadep"vncdc/BinhDuong.json")
    elseif location_code == "dongnai"
        VnCdcData.read_timeseries_confirmed_and_deaths(datadep"vncdc/DongNai.json")
    elseif location_code == "longan"
        VnCdcData.read_timeseries_confirmed_and_deaths(datadep"vncdc/LongAn.json")
    else
        throw("Unsupported location code '$location_code'!")
    end

get_experiment_population(location_code::AbstractString) =
    if location_code == "vietnam"
        97_582_700
    elseif location_code == "hcm"
        df_population = CSV.read(FPATH_VIETNAM_PROVINCES_GADM_AND_GSO_POPULATION, DataFrame)
        first(filter(x -> x.NAME_1 == "Hồ Chí Minh city", df_population).AVGPOPULATION)
    elseif location_code == "binhduong"
        df_population = CSV.read(FPATH_VIETNAM_PROVINCES_GADM_AND_GSO_POPULATION, DataFrame)
        first(filter(x -> x.NAME_1 == "Bình Dương", df_population).AVGPOPULATION)
    elseif location_code == "dongnai"
        df_population = CSV.read(FPATH_VIETNAM_PROVINCES_GADM_AND_GSO_POPULATION, DataFrame)
        first(filter(x -> x.NAME_1 == "Đồng Nai", df_population).AVGPOPULATION)
    elseif location_code == "longan"
        df_population = CSV.read(FPATH_VIETNAM_PROVINCES_GADM_AND_GSO_POPULATION, DataFrame)
        first(filter(x -> x.NAME_1 == "Long An", df_population).AVGPOPULATION)
    else
        throw("Unsupported location code '$location_code'!")
    end

function get_experiment_data_config(
    location_code::AbstractString,

    train_range::Day,
    forecast_range::Day,
)
    data_cols1 = ["infective", "recovered_total", "deaths_total", "confirmed_total"]
    data_cols2 = ["deaths_total", "confirmed_total"]
    df = get_experiment_covid_timeseries(location_code)

    initial_state_fn, data_cols = if issubset(data_cols1, names(df))
        initial_state_fn = function (data::AbstractVector{<:Real}, population::Real)
            I0 = data[1] # infective individuals
            R0 = data[2] # recovered individuals
            D0 = data[3] # total deaths
            C0 = data[4] # total confirmed
            N0 = population - D0 # effective population
            E0 = I0 * 2 # exposed individuals
            S0 = population - C0 - E0 # susceptible individuals
            # initial state
            u0 = [S0, E0, I0, R0, D0, C0, N0]
            return u0
        end
        initial_state_fn, data_cols1
    elseif issubset(data_cols2, names(df))
        initial_state_fn = function (data::AbstractVector{<:Real}, population::Real)
            D0 = data[1] # total deaths
            C0 = data[2] # total confirmed
            I0 = div(C0 - D0, 2) # infective individuals
            R0 = C0 - I0 - D0 # recovered individuals
            N0 = population - D0 # effective population
            E0 = I0 * 2 # exposed individuals
            S0 = population - C0 - E0 # susceptible individuals
            # initial state
            u0 = [S0, E0, I0, R0, D0, C0, N0]
            return u0
        end
        initial_state_fn, data_cols2
    else
        throw("Unsupported dataframe structure!")
    end

    first_date = first(subset(df, :confirmed_total => x -> x .>= 500, view = true).date)
    split_date = first_date + train_range
    last_date = split_date + forecast_range
    @info first_date

    population = get_experiment_population(location_code)
    moving_average!(df, data_cols, 7)

    return DataConfig(
        df,
        data_cols,
        :date,
        first_date,
        split_date,
        last_date,
        population,
        initial_state_fn,
    )
end

function get_experiment_movement_range_config(
    location_code::AbstractString,
    temporal_lag::Day,
)
    df = if location_code == "vietnam"
        CSV.read(FPATH_VIETNAM_AVERAGE_MOVEMENT_RANGE, DataFrame)
    elseif location_code == "hcm"
        CSV.read(FPATH_HCM_CITY_AVERAGE_MOVEMENT_RANGE, DataFrame)
    elseif location_code == "binhduong"
        CSV.read(FPATH_BINH_DUONG_AVERAGE_MOVEMENT_RANGE, DataFrame)
    elseif location_code == "dongnai"
        CSV.read(FPATH_DONG_NAI_AVERAGE_MOVEMENT_RANGE, DataFrame)
    elseif location_code == "longan"
        CSV.read(FPATH_LONG_AN_AVERAGE_MOVEMENT_RANGE, DataFrame)
    else
        throw("Unsupported location code '$location_code'!")
    end

    data_cols =
        [:all_day_bing_tiles_visited_relative_change, :all_day_ratio_single_tile_users]
    moving_average!(df, data_cols, 7)

    return MobilityConfig(df, data_cols, :ds, temporal_lag)
end

function get_experiment_social_proximity_config(
    location_code::AbstractString,
    temporal_lag::Day,
)
    get_social_proximity_vn = function ()
        df_population = CSV.read(FPATH_VIETNAM_PROVINCES_GADM_AND_GSO_POPULATION, DataFrame)
        df_covid_timeseries_confirmed = CSV.read(
            datadep"vnexpress/timeseries-vietnam-provinces-confirmed.csv",
            DataFrame,
        )
        df_social_connectedness =
            CSV.read(FPATH_VIETNAM_INTER_PROVINCE_SOCIAL_CONNECTEDNESS, DataFrame)

        return FacebookData.calculate_social_proximity_to_cases(
            df_population,
            df_covid_timeseries_confirmed,
            df_social_connectedness,
        )
    end

    df, data_col = if location_code == "hcm"
        get_social_proximity_vn(), "Hồ Chí Minh city"
    elseif location_code == "binhduong"
        get_social_proximity_vn(), "Bình Dương"
    elseif location_code == "dongnai"
        get_social_proximity_vn(), "Đồng Nai"
    elseif location_code == "longan"
        get_social_proximity_vn(), "Long An"
    else
        throw("Unsupported location code '$location_code'!")
    end

    df = df[!, ["date", data_col]]
    moving_average!(df, data_col, 7)

    return MobilityConfig(df, data_col, :date, temporal_lag)
end

function get_experiment_eval_config(name::AbstractString)
    _, location_code = rsplit(name, ".", limit = 2)
    vars, labels = if location_code == "vietnam"
        3:6, ["infective", "recovered", "deaths", "total confirmed"]
    elseif location_code == "hcm" ||
           location_code == "binhduong" ||
           location_code == "dongnai" ||
           location_code == "longan"
        5:6, ["deaths", "total confirmed"]
    else
        throw("Unsupported location code '$location_code'!")
    end

    return EvalConfig([mae, mape, rmse], [7, 14, 21, 28], vars, labels)
end

function train_and_evaluate_model(
    model::AbstractCovidModel,
    loss_metric_fn::Function,
    train_dataset::TimeseriesDataset,
    test_dataset::TimeseriesDataset,
    train_sessions::AbstractVector{TrainSession},
    eval_config::EvalConfig,
    snapshots_dir::AbstractString,
)
    predict_fn = Predictor(model)
    train_loss_fn = Loss(loss_metric_fn, predict_fn, train_dataset, eval_config.vars)
    test_loss_fn = Loss(loss_metric_fn, predict_fn, test_dataset, eval_config.vars)
    p0 = Covid19ModelVN.initial_params(model)

    @info "Initial training loss: $(train_loss_fn(p0))"
    @info "Initial testing loss: $(test_loss_fn(p0))"

    @infor "Snapshot directory '$snapshots_dir'"
    if !isdir(snapshots_dir)
        mkpath(snapshots_dir)
    end

    @info "Start training"
    train_model(train_loss_fn, test_loss_fn, p0, train_sessions, snapshots_dir)

    @info "Ploting evaluations"
    evaluate_model(predict_fn, train_dataset, test_dataset, eval_config, snapshots_dir)

    return nothing
end

function setup_experiment(
    name::AbstractString;
    train_range = Day(31),
    forecast_range = Day(28),
)
    model_type, location_code = rsplit(name, ".", limit = 2)
    base_config = get_experiment_data_config(location_code, train_range, forecast_range)

    model, train_dataset, test_dataset = if model_type == "baseline.default"
        setup_model(base_config)
    elseif model_type == "fbmobility1.default"
        setup_model(base_config, get_experiment_movement_range_config(location_code, Day(2)))
    elseif model_type == "fbmobility2.default"
        setup_model(
            base_config,
            get_experiment_movement_range_config(location_code, Day(2)),
            get_experiment_social_proximity_config(location_code, Day(2)),
        )
    else
        throw("Unsupported model type '$model_type'")
    end

    return model, train_dataset, test_dataset
end

function run_experiments(names::AbstractVector{<:AbstractString} = String[])
    for name ∈ names
        timestamp = Dates.format(now(), "yyyymmddHHMMSS")
        sessions = [
            TrainSession("$timestamp.adam", ADAM(1e-3), 10000, 100),
            TrainSession("$timestamp.lbfgs", LBFGS(), 1000, 100),
        ]
        eval_config = get_experiment_eval_config(name)

        model, train_dataset, test_dataset = setup_experiment(name)
        snapshots_dir = joinpath(SNAPSHOTS_DIR, name)
        train_and_evaluate_model(
            model,
            rmsle,
            train_dataset,
            test_dataset,
            sessions,
            eval_config,
            snapshots_dir,
        )
    end
end

cachedata()

# run_experiments(
#     vec([
#         "$model.$loc" for
#         model ∈ ["baseline.default", "fbmobility1.default"],
#         loc ∈ ["vietnam"]
#     ]),
# )

run_experiments(
    vec([
        "$model.$loc" for
        model ∈ ["baseline.default", "fbmobility1.default", "fbmobility2.default"],
        loc ∈ ["hcm", "binhduong", "dongnai", "longan"]
    ]),
)