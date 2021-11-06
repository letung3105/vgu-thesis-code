using Dates, Statistics, CairoMakie, DataFrames, DataDeps, DiffEqFlux, Covid19ModelVN

import Covid19ModelVN.JHUCSSEData,
    Covid19ModelVN.FacebookData,
    Covid19ModelVN.PopulationData,
    Covid19ModelVN.VnExpressData,
    Covid19ModelVN.VnCdcData

BASELINE_EXPERIMENTS = [
    "baseline.default.$loc" for loc ∈ [
        Covid19ModelVN.LOC_CODE_VIETNAM
        Covid19ModelVN.LOC_CODE_UNITED_STATES
        collect(keys(Covid19ModelVN.LOC_NAMES_VN))
        collect(keys(Covid19ModelVN.LOC_NAMES_US))
    ]
]

FBMOBILITY1_EXPERIMENTS = [
    "fbmobility1.default.$loc" for loc ∈ [
        Covid19ModelVN.LOC_CODE_VIETNAM
        Covid19ModelVN.LOC_CODE_UNITED_STATES
        collect(keys(Covid19ModelVN.LOC_NAMES_VN))
        collect(keys(Covid19ModelVN.LOC_NAMES_US))
    ]
]

FBMOBILITY2_EXPERIMENTS = [
    "fbmobility2.default.$loc" for loc ∈ [
        collect(keys(Covid19ModelVN.LOC_NAMES_VN))
        collect(keys(Covid19ModelVN.LOC_NAMES_US))
    ]
]

ALL_EXPERIMENTS =
    [BASELINE_EXPERIMENTS; FBMOBILITY1_EXPERIMENTS; FBMOBILITY2_EXPERIMENTS]

function experiment_covid19_dataframe(location_code::AbstractString)
    df = get_prebuilt_covid_timeseries(location_code)
    first_date, last_date = if location_code == Covid19ModelVN.LOC_CODE_VIETNAM
        # we considered 27th April 2021 to be the start of the outbreak in Vietnam
        # 4th August 2021 is when no recovered cases are recorded in JHU data
        Date(2021, 4, 27), Date(2021, 8, 4)
    elseif location_code == Covid19ModelVN.LOC_CODE_UNITED_STATES ||
           location_code ∈ keys(Covid19ModelVN.LOC_NAMES_US)
        # we considered 1st July 2021 to be the start of the outbreak in the US
        # 30th September 2021 is for getting a long enough period
        Date(2021, 7, 1), Date(2021, 9, 30)
    else
        # data for other locations don't need processing
        return df
    end

    # make the cases count starts from the first date of the considered outbreak
    bound!(df, :date, first_date - Day(1), last_date)
    datacols = names(df, Not(:date))
    starting_states = Array(df[1, datacols])
    transform!(df, datacols => ByRow((x...) -> x .- starting_states) => datacols)
    bound!(df, :date, first_date, last_date)
    return df
end

function experiment_covid19_data(loc::AbstractString, train_range::Day, forecast_range::Day)
    cols = if has_irdc(loc)
        ["infective", "recovered_total", "deaths_total", "confirmed_total"]
    elseif has_dc(loc)
        ["deaths_total", "confirmed_total"]
    else
        throw("Unsupported location code '$loc'!")
    end

    df = experiment_covid19_dataframe(loc)
    # choose the first date to be when the number of total confirmed cases passed 500
    first_date = first(subset(df, :confirmed_total => x -> x .>= 500, view = true).date)
    split_date = first_date + train_range - Day(1)
    last_date = split_date + forecast_range
    @info [first_date; split_date; last_date]

    # smooth out weekly seasonality
    moving_average!(df, cols, 7)
    conf = TimeseriesConfig(df, :date, cols)
    # split data into 2 parts for training and testing
    train_dataset, test_dataset = train_test_split(conf, first_date, split_date, last_date)
    return train_dataset, test_dataset, first_date, last_date
end

function experiment_movement_range(loc::AbstractString, first_date::Date, last_date::Date)
    df = get_prebuilt_movement_range(loc)
    cols = [:all_day_bing_tiles_visited_relative_change, :all_day_ratio_single_tile_users]
    # smooth out weekly seasonality
    moving_average!(df, cols, 7)
    return load_timeseries(TimeseriesConfig(df, :ds, cols), first_date, last_date)
end

function experiment_social_proximity(loc::AbstractString, first_date::Date, last_date::Date)
    df, col = get_prebuilt_social_proximity(loc)
    # smooth out weekly seasonality
    moving_average!(df, col, 7)
    return load_timeseries(TimeseriesConfig(df, :date, col), first_date, last_date)
end

is_seird(name) =
    name == "baseline.default" ||
    name == "fbmobility1.default" ||
    name == "fbmobility2.default"

has_irdc(loc) = loc == Covid19ModelVN.LOC_CODE_VIETNAM

has_dc(loc) =
    loc == Covid19ModelVN.LOC_CODE_UNITED_STATES ||
    loc ∈ keys(Covid19ModelVN.LOC_NAMES_VN) ||
    loc ∈ keys(Covid19ModelVN.LOC_NAMES_US)

function experiment_setup(
    experiment_name::AbstractString;
    train_range::Day = Day(32),
    forecast_range::Day = Day(28),
    social_proximity_lag::Day = Day(14),
)
    model_name, loc = rsplit(experiment_name, ".", limit = 2)
    train_dataset, test_dataset, first_date, last_date =
        experiment_covid19_data(loc, train_range, forecast_range)
    @assert size(train_dataset.data, 2) == Dates.value(train_range)
    @assert size(test_dataset.data, 2) == Dates.value(forecast_range)

    # get the initial states and available observations depending on the model type
    # and the considered location
    population = get_prebuilt_population(loc)
    u0, vars, labels = if is_seird(model_name)
        if has_irdc(loc)
            # Infective, Recovered, Deaths, Cummulative are observable
            I0 = train_dataset.data[1, 1] # infective individuals
            R0 = train_dataset.data[2, 1] # recovered individuals
            D0 = train_dataset.data[3, 1] # total deaths
            C0 = train_dataset.data[4, 1] # total confirmed
            N0 = population - D0 # effective population
            E0 = I0 * 2 # exposed individuals
            S0 = population - C0 - E0 # susceptible individuals
            # initial state
            u0 = [S0, E0, I0, R0, D0, C0, N0]
            vars = [3, 4, 5, 6]
            labels = ["infective", "recovered", "deaths", "total confirmed"]
            # return values to outer scope
            u0, vars, labels
        elseif has_dc(loc)
            # Deaths, Cummulative are observable
            D0 = train_dataset.data[1, 1] # total deaths
            C0 = train_dataset.data[2, 1] # total confirmed
            I0 = div(C0 - D0, 2) # infective individuals
            R0 = C0 - I0 - D0 # recovered individuals
            N0 = population - D0 # effective population
            E0 = I0 * 2 # exposed individuals
            S0 = population - C0 - E0 # susceptible individuals
            # initial state
            u0 = [S0, E0, I0, R0, D0, C0, N0]
            vars = [5, 6]
            labels = ["deaths", "total confirmed"]
            # return values to outer scope
            u0, vars, labels
        else
            throw("Unsupported location code '$loc'!")
        end
    else
        throw("Unsupported model '$model_name'!")
    end

    if model_name == "baseline.default"
        model = CovidModelSEIRDBaseline(u0, train_dataset.tspan)
        return model, train_dataset, test_dataset, vars, labels
    end

    movement_range_data = experiment_movement_range(loc, first_date, last_date)
    @assert size(movement_range_data, 2) == Dates.value(train_range + forecast_range)
    if model_name == "fbmobility1.default"
        model = CovidModelSEIRDFbMobility1(u0, train_dataset.tspan, movement_range_data)
        return model, train_dataset, test_dataset, vars, labels
    end

    social_proximity_data = experiment_social_proximity(
        loc,
        first_date - social_proximity_lag,
        last_date - social_proximity_lag,
    )
    @assert size(social_proximity_data, 2) == Dates.value(train_range + forecast_range)
    if model_name == "fbmobility2.default"
        model = CovidModelSEIRDFbMobility2(
            u0,
            train_dataset.tspan,
            movement_range_data,
            social_proximity_data,
        )
        return model, train_dataset, test_dataset, vars, labels
    end

    throw("Unsupported model '$model_name'")
end

struct Hyperparameters
    ζ::Float64
    sessions::AbstractVector{TrainSession}
    box_constrained::Bool
    γ_bounds::Tuple{<:Real,<:Real}
    λ_bounds::Tuple{<:Real,<:Real}
    α_bounds::Tuple{<:Real,<:Real}
end

Hyperparameters(ζ::Float64, sessions::AbstractVector{TrainSession}) =
    Hyperparameters(ζ, sessions, false, (-Inf, Inf), (-Inf, Inf), (-Inf, Inf))

function experiment_train_and_eval(
    uuid::AbstractString,
    experiment_name::AbstractString,
    hyperparams::Hyperparameters;
    snapshots_dir::AbstractString = "snapshots",
)
    if !isdir(snapshots_dir)
        mkpath(snapshots_dir)
    end

    # get model
    model, train_dataset, test_dataset, vars, labels = experiment_setup(experiment_name)
    p0 = Covid19ModelVN.initial_params(model)
    predictor = Predictor(model)
    # build loss function
    weights = exp.(collect(train_dataset.tsteps) .* hyperparams.ζ)
    lossfn = (ŷ, y) -> sum((log.(ŷ .+ 1) .- log.(y .+ 1)) .^ 2 .* weights')
    train_loss = Loss(lossfn, predictor, train_dataset, vars)

    @info "Initial loss: $(train_loss(p0))"

    minimizers = if hyperparams.box_constrained
        # optimizing the model with box constraints (not all optimizers,
        # support box-constrained optimization)
        lower_bounds = [
            hyperparams.γ_bounds[1]
            hyperparams.λ_bounds[1]
            hyperparams.α_bounds[1]
            fill(-Inf, DiffEqFlux.paramlength(model.β_ann))
        ]
        upper_bounds = [
            hyperparams.γ_bounds[2]
            hyperparams.λ_bounds[2]
            hyperparams.α_bounds[2]
            fill(Inf, DiffEqFlux.paramlength(model.β_ann))
        ]
        train_model(
            train_loss,
            p0,
            hyperparams.sessions,
            lower_bounds = lower_bounds,
            upper_bounds = upper_bounds,
            snapshots_dir = snapshots_dir,
        )
    else
        # optimizing the model without constraints
        train_model(train_loss, p0, hyperparams.sessions, snapshots_dir = snapshots_dir)
    end

    @info "Evaluating model"
    # only evaluating the last parameters
    minimizer = last(minimizers)
    eval_config = EvalConfig([mae, mape, rmse], [7, 14, 21, 28], vars, labels)
    # plot the model's forecasts againts the ground truth, and calculate to model's
    # error on the test data
    forecasts_plot, df_forecasts_errors =
        evaluate_model(eval_config, minimizer, predictor, train_dataset, test_dataset)
    # save the forecasts errors
    save_dataframe(
        df_forecasts_errors,
        joinpath(snapshots_dir, "$uuid.$experiment_name.errors.csv"),
    )
    # save the forecasts plot
    save(joinpath(snapshots_dir, "$uuid.$experiment_name.forecasts.png"), forecasts_plot)
    # get the effective reproduction number learned by the model
    R_effective_plot =
        plot_effective_reproduction_number(model, minimizer, train_dataset, test_dataset)
    save(
        joinpath(snapshots_dir, "$uuid.$experiment_name.R_effective.png"),
        R_effective_plot,
    )

    return forecasts_plot, df_forcasts_errors, R_effective_plot
end
