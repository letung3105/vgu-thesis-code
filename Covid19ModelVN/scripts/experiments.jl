using Dates, Statistics, Plots, CSV, DataDeps, DataFrames, DiffEqFlux, Covid19ModelVN

import Covid19ModelVN.JHUCSSEData,
    Covid19ModelVN.FacebookData,
    Covid19ModelVN.PopulationData,
    Covid19ModelVN.VnExpressData,
    Covid19ModelVN.VnCdcData

const SNAPSHOTS_DIR = "snapshots"

get_experiment_covid_timeseries(location_code::AbstractString) =
    if location_code == "vietnam"
        df = CSV.read(
            datadep"covid19model/timeseries-covid19-combined-vietnam.csv",
            DataFrame,
        )
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
    elseif location_code == "unitedstates"
        CSV.read(
            datadep"covid19model/timeseries-covid19-combined-united-states.csv",
            DataFrame,
        )
    elseif location_code == "losangeles_ca"
        CSV.read(
            datadep"covid19model/timeseries-covid19-combined-los-angeles-CA.csv",
            DataFrame,
        )
    elseif location_code == "cook_il"
        CSV.read(
            datadep"covid19model/timeseries-covid19-combined-cook-county-IL.csv",
            DataFrame,
        )
    elseif location_code == "harris_tx"
        CSV.read(
            datadep"covid19model/timeseries-covid19-combined-harris-county-TX.csv",
            DataFrame,
        )
    elseif location_code == "maricopa_az"
        CSV.read(
            datadep"covid19model/timeseries-covid19-combined-maricopa-county-AZ.csv",
            DataFrame,
        )
    else
        throw("Unsupported location code '$location_code'!")
    end

get_experiment_population(location_code::AbstractString) =
    if location_code == "vietnam"
        97_582_700
    elseif location_code == "hcm"
        df_population =
            CSV.read(datadep"covid19model/average-population-vn-provinces.csv", DataFrame)
        first(filter(x -> x.NAME_1 == "Hồ Chí Minh city", df_population).AVGPOPULATION)
    elseif location_code == "binhduong"
        df_population =
            CSV.read(datadep"covid19model/average-population-vn-provinces.csv", DataFrame)
        first(filter(x -> x.NAME_1 == "Bình Dương", df_population).AVGPOPULATION)
    elseif location_code == "dongnai"
        df_population =
            CSV.read(datadep"covid19model/average-population-vn-provinces.csv", DataFrame)
        first(filter(x -> x.NAME_1 == "Đồng Nai", df_population).AVGPOPULATION)
    elseif location_code == "longan"
        df_population =
            CSV.read(datadep"covid19model/average-population-vn-provinces.csv", DataFrame)
        first(filter(x -> x.NAME_1 == "Long An", df_population).AVGPOPULATION)
    elseif location_code == "unitedstates"
        332_889_844
    elseif location_code == "losangeles_ca"
        df_population =
            CSV.read(datadep"covid19model/average-population-us-counties.csv", DataFrame)
        first(
            filter(
                x -> x.NAME_1 == "Los Angeles, California, US",
                df_population,
            ).AVGPOPULATION,
        )
    elseif location_code == "cook_il"
        df_population =
            CSV.read(datadep"covid19model/average-population-us-counties.csv", DataFrame)
        first(filter(x -> x.NAME_1 == "Cook, Illinois, US", df_population).AVGPOPULATION)
    elseif location_code == "harris_tx"
        df_population =
            CSV.read(datadep"covid19model/average-population-us-counties.csv", DataFrame)
        first(filter(x -> x.NAME_1 == "Harris, Texas, US", df_population).AVGPOPULATION)
    elseif location_code == "maricopa_az"
        df_population =
            CSV.read(datadep"covid19model/average-population-us-counties.csv", DataFrame)
        first(filter(x -> x.NAME_1 == "Maricopa, Arizona, US", df_population).AVGPOPULATION)
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
    # moving_average!(df, data_cols, 7)

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

function get_experiment_movement_range_config(location_code::AbstractString)
    df = if location_code == "vietnam"
        CSV.read(datadep"covid19model/movement-range-vietnam.csv", DataFrame)
    elseif location_code == "hcm"
        CSV.read(datadep"covid19model/movement-range-hcm-city.csv", DataFrame)
    elseif location_code == "binhduong"
        CSV.read(datadep"covid19model/movement-range-binh-duong.csv", DataFrame)
    elseif location_code == "dongnai"
        CSV.read(datadep"covid19model/movement-range-dong-nai.csv", DataFrame)
    elseif location_code == "longan"
        CSV.read(datadep"covid19model/movement-range-long-an.csv", DataFrame)
    elseif location_code == "unitedstates"
        CSV.read(datadep"covid19model/movement-range-united-states-2020.csv", DataFrame)
    elseif location_code == "losangeles_ca"
        CSV.read(datadep"covid19model/movement-range-los-angeles-CA-2020.csv", DataFrame)
    elseif location_code == "cook_il"
        CSV.read(datadep"covid19model/movement-range-cook-county-IL-2020.csv", DataFrame)
    elseif location_code == "harris_tx"
        CSV.read(datadep"covid19model/movement-range-harris-county-TX-2020.csv", DataFrame)
    elseif location_code == "maricopa_az"
        CSV.read(datadep"covid19model/movement-range-maricopa-county-AZ-2020.csv", DataFrame)
    else
        throw("Unsupported location code '$location_code'!")
    end

    data_cols =
        [:all_day_bing_tiles_visited_relative_change, :all_day_ratio_single_tile_users]
    moving_average!(df, data_cols, 7)

    return MobilityConfig(df, data_cols, :ds, Day(0))
end

function get_experiment_social_proximity_config(
    location_code::AbstractString,
    temporal_lag::Day,
)
    df, data_col = if location_code == "hcm"
        df = CSV.read(
            datadep"covid19model/social-proximity-to-cases-vn-provinces.csv",
            DataFrame,
        )
        df, "Hồ Chí Minh city"
    elseif location_code == "binhduong"
        df = CSV.read(
            datadep"covid19model/social-proximity-to-cases-vn-provinces.csv",
            DataFrame,
        )
        df, "Bình Dương"
    elseif location_code == "dongnai"
        df = CSV.read(
            datadep"covid19model/social-proximity-to-cases-vn-provinces.csv",
            DataFrame,
        )
        df, "Đồng Nai"
    elseif location_code == "longan"
        df = CSV.read(
            datadep"covid19model/social-proximity-to-cases-vn-provinces.csv",
            DataFrame,
        )
        df, "Long An"
    elseif location_code == "losangeles_ca"
        df = CSV.read(
            datadep"covid19model/social-proximity-to-cases-us-counties.csv",
            DataFrame,
        )
        df, "Los Angeles, California, US"
    elseif location_code == "cook_il"
        df = CSV.read(
            datadep"covid19model/social-proximity-to-cases-us-counties.csv",
            DataFrame,
        )
        df, "Cook, Illinois, US"
    elseif location_code == "harris_tx"
        df = CSV.read(
            datadep"covid19model/social-proximity-to-cases-us-counties.csv",
            DataFrame,
        )
        df, "Harris, Texas, US"
    elseif location_code == "maricopa_az"
        df = CSV.read(
            datadep"covid19model/social-proximity-to-cases-us-counties.csv",
            DataFrame,
        )
        df, "Maricopa, Arizona, US"
    else
        throw("Unsupported location code '$location_code'!")
    end

    df = df[!, ["date", data_col]]
    moving_average!(df, data_col, 7)

    return MobilityConfig(df, data_col, :date, temporal_lag)
end

function get_experiment_eval_config(name::AbstractString)
    _, location_code = rsplit(name, ".", limit = 2)
    vars, labels = if location_code == "vietnam" || location_code == "unitedstates"
        3:6, ["infective", "recovered", "deaths", "total confirmed"]
    elseif location_code == "hcm" ||
           location_code == "binhduong" ||
           location_code == "dongnai" ||
           location_code == "longan" ||
           location_code == "losangeles_ca" ||
           location_code == "cook_il" ||
           location_code == "harris_tx" ||
           location_code == "maricopa_az"
        5:6, ["deaths", "total confirmed"]
    else
        throw("Unsupported location code '$location_code'!")
    end

    return EvalConfig([mae, mape, rmse], [7, 14, 21, 28], vars, labels)
end

function setup_experiment(
    name::AbstractString;
    train_range = Day(31),
    forecast_range = Day(28),
)
    model_type, location_code = rsplit(name, ".", limit = 2)
    base_config = get_experiment_data_config(location_code, train_range, forecast_range)

    model, train_dataset, test_dataset = if model_type == "baseline.default"
        setup_model(CovidModelSEIRDBaseline, base_config)
    elseif model_type == "fbmobility1.default"
        setup_model(
            CovidModelSEIRDFbMobility1,
            base_config,
            get_experiment_movement_range_config(location_code),
        )
    elseif model_type == "fbmobility2.default"
        setup_model(
            CovidModelSEIRDFbMobility2,
            base_config,
            get_experiment_movement_range_config(location_code),
            get_experiment_social_proximity_config(location_code, Day(14)),
        )
    else
        throw("Unsupported model type '$model_type'")
    end

    return model, train_dataset, test_dataset
end

function run_experiments(
    names::AbstractVector{<:AbstractString} = String[],
    snapshots_dir::AbstractString = SNAPSHOTS_DIR,
)
    for name ∈ names
        timestamp = Dates.format(now(), "yyyymmddHHMMSS")
        train_sessions = [
            TrainSession("$timestamp.adam", ADAM(1e-3), 8000, 100),
            TrainSession("$timestamp.bfgs", BFGS(initial_stepnorm = 1e-2), 1000, 100),
        ]
        eval_config = get_experiment_eval_config(name)

        model, train_dataset, test_dataset = setup_experiment(name)

        weights = exp.(cumsum(ones(size(train_dataset.data, 2))) .* 0.1)
        loss(ŷ, y) = sum((log.(ŷ .+ 1) .- log.(y .+ 1)) .^ 2 .* weights')

        predict_fn = Predictor(model)
        train_loss_fn = Loss(loss, predict_fn, train_dataset, eval_config.vars)
        test_loss_fn = Loss(rmse, predict_fn, test_dataset, eval_config.vars)
        p0 = Covid19ModelVN.initial_params(model)

        @info "Initial training loss: $(train_loss_fn(p0))"
        @info "Initial testing loss: $(test_loss_fn(p0))"

        experiment_dir = joinpath(snapshots_dir, name)
        if !isdir(experiment_dir)
            mkpath(experiment_dir)
        end
        @info "Snapshot directory '$experiment_dir'"

        @info "Start training"
        sessions_params = train_model(
            train_sessions,
            train_loss_fn,
            p0,
            snapshots_dir = experiment_dir,
            test_loss_fn = test_loss_fn,
        )

        @info "Ploting evaluations"
        for (sess, params) in zip(train_sessions, sessions_params)
            eval_res =
                evaluate_model(eval_config, params, predict_fn, train_dataset, test_dataset)

            csv_fpath = joinpath(experiment_dir, "$(sess.name).evaluate.csv")
            if !isfile(csv_fpath)
                CSV.write(csv_fpath, eval_res.df_errors)
            end

            fig_fpath = joinpath(experiment_dir, "$(sess.name).evaluate.png")
            if !isfile(fig_fpath)
                savefig(eval_res.fig_forecasts, fig_fpath)
            end
        end
    end
end

# run_experiments(
#     vec([
#         "$model.$loc" for model ∈ ["baseline.default", "fbmobility1.default"], loc ∈ [
#             "vietnam",
#             "hcm",
#             "binhduong",
#             "dongnai",
#             "longan",
#             "unitedstates",
#             "losangeles_ca",
#             "cook_il",
#             "harris_tx",
#             "maricopa_az",
#         ]
#     ]),
#     "snapshots/run02",
# )

# run_experiments(
#     vec([
#         "fbmobility2.default.$loc" for loc ∈ [
#             "hcm",
#             "binhduong",
#             "dongnai",
#             "longan",
#             "losangeles_ca",
#             "cook_il",
#             "harris_tx",
#             "maricopa_az",
#         ]
#     ]),
#     "snapshots/run02",
# )
