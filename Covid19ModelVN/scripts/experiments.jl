using Dates, Statistics, Plots, CSV, DataDeps, DataFrames, DiffEqFlux, Covid19ModelVN

import Covid19ModelVN.JHUCSSEData,
    Covid19ModelVN.FacebookData,
    Covid19ModelVN.PopulationData,
    Covid19ModelVN.VnExpressData,
    Covid19ModelVN.VnCdcData

function get_experiment_covid_timeseries(location_code::AbstractString)
    df = get_prebuilt_covid_timeseries(location_code)

    first_date, last_date = if location_code == Covid19ModelVN.LOC_CODE_VIETNAM
        # after 4th August, the recovered count is not updated
        Date(2021, 4, 27), Date(2021, 8, 4)
    elseif location_code == Covid19ModelVN.LOC_CODE_UNITED_STATES ||
           location_code ∈ keys(Covid19ModelVN.LOC_NAMES_US)
        # we considered 1st July 2021 to be the start of the 4th outbreak in the US
        Date(2021, 7, 1), Date(2021, 9, 30)
    else
        return df
    end

    bound!(df, :date, first_date - Day(1), last_date)
    datacols = names(df, Not(:date))
    starting_states = Array(df[1, datacols])
    transform!(df, datacols => ByRow((x...) -> x .- starting_states) => datacols)
    bound!(df, :date, first_date, last_date)

    return df
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
    @info [first_date; split_date; last_date]

    population = get_prebuilt_population(location_code)
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

function get_experiment_movement_range_config(location_code::AbstractString)
    df = get_prebuilt_movement_range(location_code)
    data_cols =
        [:all_day_bing_tiles_visited_relative_change, :all_day_ratio_single_tile_users]
    moving_average!(df, data_cols, 7)
    return MobilityConfig(df, data_cols, :ds, Day(0))
end

function get_experiment_social_proximity_config(
    location_code::AbstractString,
    temporal_lag::Day,
)
    df, data_col = get_prebuilt_social_proximity(location_code)
    moving_average!(df, data_col, 7)
    return MobilityConfig(df, data_col, :date, temporal_lag)
end

function get_experiment_eval_config(location_code::AbstractString)
    vars, labels =
        if location_code == Covid19ModelVN.LOC_CODE_VIETNAM ||
           location_code == Covid19ModelVN.LOC_CODE_UNITED_STATES
            3:6, ["infective", "recovered", "deaths", "total confirmed"]
        elseif location_code ∈ keys(Covid19ModelVN.LOC_NAMES_VN) ||
               location_code ∈ keys(Covid19ModelVN.LOC_NAMES_US)
            5:6, ["deaths", "total confirmed"]
        else
            throw("Unsupported location code '$location_code'!")
        end

    return EvalConfig([mae, mape, rmse], [7, 14, 21, 28], vars, labels)
end

function setup_experiment(
    model_type::AbstractString,
    location_code::AbstractString;
    train_range = Day(31),
    forecast_range = Day(28),
)
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
