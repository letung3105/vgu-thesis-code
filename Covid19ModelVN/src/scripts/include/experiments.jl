using Dates, Statistics, Serialization
using OrdinaryDiffEq, DiffEqFlux
using CairoMakie
using DataFrames
using Covid19ModelVN
using JSON

import Covid19ModelVN.JHUCSSEData,
    Covid19ModelVN.FacebookData,
    Covid19ModelVN.PopulationData,
    Covid19ModelVN.VnExpressData,
    Covid19ModelVN.VnCdcData

has_irdc(loc) = loc == Covid19ModelVN.LOC_CODE_VIETNAM

has_dc(loc) =
    loc == Covid19ModelVN.LOC_CODE_UNITED_STATES ||
    loc ∈ keys(Covid19ModelVN.LOC_NAMES_VN) ||
    loc ∈ keys(Covid19ModelVN.LOC_NAMES_US)

function experiment_covid19_counts_reset!(df::AbstractDataFrame, loc::AbstractString)
    first_date, last_date = if loc == Covid19ModelVN.LOC_CODE_VIETNAM
        # we considered 27th April 2021 to be the start of the outbreak in Vietnam
        # 4th August 2021 is when no recovered cases are recorded in JHU data
        Date(2021, 4, 27), Date(2021, 8, 4)
    elseif loc == Covid19ModelVN.LOC_CODE_UNITED_STATES ||
           loc ∈ keys(Covid19ModelVN.LOC_NAMES_US)
        # we considered 1st July 2021 to be the start of the outbreak in the US
        # 30th September 2021 is for getting a long enough period
        bound!(df, :date, Date(2021, 7, 1), Date(2021, 9, 30))
        return df
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

function experiment_covid19_data(
    loc::AbstractString,
    train_range::Day,
    forecast_range::Day,
    ma7::Bool,
)
    cols = if has_irdc(loc)
        ["infective", "recovered_total", "deaths_total", "confirmed_total"]
    elseif has_dc(loc)
        ["deaths_total", "confirmed_total"]
    else
        throw("Unsupported location code '$loc'!")
    end

    df = get_prebuilt_covid_timeseries(loc)
    df[!, cols] .= Float32.(df[!, cols])
    experiment_covid19_counts_reset!(df, loc)
    # choose the first date to be when the number of total confirmed cases passed 500
    first_date = first(subset(df, :confirmed_total => x -> x .>= 500, view = true).date)
    split_date = first_date + train_range - Day(1)
    last_date = split_date + forecast_range
    # smooth out weekly seasonality
    if ma7
        moving_average!(df, cols, 7)
    end
    conf = TimeseriesConfig(df, "date", cols)
    # split data into 2 parts for training and testing
    train_dataset, test_dataset = train_test_split(conf, first_date, split_date, last_date)

    @info "Got Covid-19 data for '$loc'\n" *
          "+ First train date $first_date\n" *
          "+ Last train date $split_date\n" *
          "+ Last evaluated date $last_date"

    return train_dataset, test_dataset, first_date, last_date
end

function experiment_movement_range(
    loc::AbstractString,
    first_date::Date,
    last_date::Date,
    ma7::Bool,
)
    df = get_prebuilt_movement_range(loc)
    cols = ["all_day_bing_tiles_visited_relative_change", "all_day_ratio_single_tile_users"]
    df[!, cols] .= Float32.(df[!, cols])
    # smooth out weekly seasonality
    if ma7
        moving_average!(df, cols, 7)
    end
    return load_timeseries(TimeseriesConfig(df, "ds", cols), first_date, last_date)
end

function experiment_social_proximity(
    loc::AbstractString,
    first_date::Date,
    last_date::Date,
    ma7::Bool,
)
    df, col = get_prebuilt_social_proximity(loc)
    df[!, col] .= Float32.(df[!, col])
    # smooth out weekly seasonality
    if ma7
        moving_average!(df, col, 7)
    end
    return load_timeseries(TimeseriesConfig(df, "date", [col]), first_date, last_date)
end

function experiment_SEIRD_initial_states(loc::AbstractString, data::AbstractVector{<:Real})
    population = get_prebuilt_population(loc)
    u0, vars, labels = if has_irdc(loc)
        # Infective, Recovered, Deaths, Cummulative are observable
        I0 = data[1] # infective individuals
        R0 = data[2] # recovered individuals
        D0 = data[3] # total deaths
        C0 = data[4] # total confirmed
        N0 = population - D0 # effective population
        E0 = I0 * 2 # exposed individuals
        S0 = population - C0 - E0 # susceptible individuals
        # initial state
        u0 = Float32[S0, E0, I0, R0, D0, C0, N0]
        vars = [3, 4, 5, 6]
        labels = ["infective", "recovered", "deaths", "total confirmed"]
        # return values to outer scope
        u0, vars, labels
    elseif has_dc(loc)
        # Deaths, Cummulative are observable
        D0 = data[1] # total deaths
        C0 = data[2] # total confirmed
        I0 = div(C0 - D0, 2) # infective individuals
        R0 = C0 - I0 - D0 # recovered individuals
        N0 = population - D0 # effective population
        E0 = I0 * 2 # exposed individuals
        S0 = population - C0 - E0 # susceptible individuals
        # initial state
        u0 = Float32[S0, E0, I0, R0, D0, C0, N0]
        vars = [5, 6]
        labels = ["deaths", "total confirmed"]
        # return values to outer scope
        u0, vars, labels
    else
        throw("Unsupported location code '$loc'!")
    end
    return u0, vars, labels
end

function experiment_loss(
    tsteps::Ts,
    ζ::R,
    min::AbstractMatrix{R},
    max::AbstractMatrix{R},
) where {Ts,R<:Real}
    weights = exp.(collect(tsteps) .* ζ)
    scale = max .- min
    lossfn = function (ŷ::AbstractArray{R}, y) where {R<:Real}
        s = zero(R)
        sz = size(y)
        @inbounds for j ∈ 1:sz[2], i ∈ 1:sz[1]
            s += ((ŷ[i, j] - y[i, j]) / scale[i])^2 * weights[j]
        end
        return s
    end
    return lossfn
end

SEIRDBaselineHyperparams = @NamedTuple begin
    L2_λ::Float32
    ζ::Float32
    γ0::Float32
    λ0::Float32
    α0::Float32
    γ_bounds::Tuple{Float32,Float32}
    λ_bounds::Tuple{Float32,Float32}
    α_bounds::Tuple{Float32,Float32}
    train_range::Day
    forecast_range::Day
    ma7::Bool
end

function setup_baseline(loc::AbstractString, hyperparams::SEIRDBaselineHyperparams)
    # get data for model
    train_dataset, test_dataset = experiment_covid19_data(
        loc,
        hyperparams.train_range,
        hyperparams.forecast_range,
        hyperparams.ma7,
    )
    @assert size(train_dataset.data, 2) == Dates.value(hyperparams.train_range)
    @assert size(test_dataset.data, 2) == Dates.value(hyperparams.forecast_range)

    # initialize the model
    model = SEIRDBaseline(hyperparams.γ_bounds, hyperparams.λ_bounds, hyperparams.α_bounds)
    # get the initial states and available observations depending on the model type
    # and the considered location
    u0, vars, labels = experiment_SEIRD_initial_states(loc, train_dataset.data[:, 1])
    p0 = initparams(model, hyperparams.γ0, hyperparams.λ0, hyperparams.α0)
    train_data_min = minimum(train_dataset.data, dims = 2)
    train_data_max = maximum(train_dataset.data, dims = 2)
    lossfn =
        experiment_loss(train_dataset.tsteps, hyperparams.ζ, train_data_min, train_data_max)
    lossfn_regularized = function (ŷ, y, params)
        pnamed = namedparams(model, params)
        return lossfn(ŷ, y) + hyperparams.L2_λ * sum(abs2, pnamed.θ)
    end
    return model, u0, p0, lossfn_regularized, train_dataset, test_dataset, vars, labels
end

SEIRDFbMobility1Hyperparams = @NamedTuple begin
    L2_λ::Float32
    ζ::Float32
    γ0::Float32
    λ0::Float32
    α0::Float32
    γ_bounds::Tuple{Float32,Float32}
    λ_bounds::Tuple{Float32,Float32}
    α_bounds::Tuple{Float32,Float32}
    train_range::Day
    forecast_range::Day
    ma7::Bool
end

function setup_fbmobility1(loc::AbstractString, hyperparams::SEIRDFbMobility1Hyperparams)
    # get data for model
    train_dataset, test_dataset, first_date, last_date = experiment_covid19_data(
        loc,
        hyperparams.train_range,
        hyperparams.forecast_range,
        hyperparams.ma7,
    )
    @assert size(train_dataset.data, 2) == Dates.value(hyperparams.train_range)
    @assert size(test_dataset.data, 2) == Dates.value(hyperparams.forecast_range)

    movement_range_data =
        experiment_movement_range(loc, first_date, last_date, hyperparams.ma7)
    @assert size(movement_range_data, 2) ==
            Dates.value(hyperparams.train_range) + Dates.value(hyperparams.forecast_range)

    # initialize the model
    model = SEIRDFbMobility1(
        hyperparams.γ_bounds,
        hyperparams.λ_bounds,
        hyperparams.α_bounds,
        movement_range_data,
    )
    # get the initial states and available observations depending on the model type
    # and the considered location
    u0, vars, labels = experiment_SEIRD_initial_states(loc, train_dataset.data[:, 1])
    p0 = initparams(model, hyperparams.γ0, hyperparams.λ0, hyperparams.α0)
    train_data_min = minimum(train_dataset.data, dims = 2)
    train_data_max = maximum(train_dataset.data, dims = 2)
    lossfn =
        experiment_loss(train_dataset.tsteps, hyperparams.ζ, train_data_min, train_data_max)
    lossfn_regularized = function (ŷ, y, params)
        pnamed = namedparams(model, params)
        return lossfn(ŷ, y) + hyperparams.L2_λ * sum(abs2, pnamed.θ)
    end
    return model, u0, p0, lossfn_regularized, train_dataset, test_dataset, vars, labels
end

SEIRDFbMobility2Hyperparams = @NamedTuple begin
    L2_λ::Float32
    ζ::Float32
    γ0::Float32
    λ0::Float32
    α0::Float32
    γ_bounds::Tuple{Float32,Float32}
    λ_bounds::Tuple{Float32,Float32}
    α_bounds::Tuple{Float32,Float32}
    train_range::Day
    forecast_range::Day
    social_proximity_lag::Day
    ma7::Bool
end

function setup_fbmobility2(loc::AbstractString, hyperparams::SEIRDFbMobility2Hyperparams)
    # get data for model
    train_dataset, test_dataset, first_date, last_date = experiment_covid19_data(
        loc,
        hyperparams.train_range,
        hyperparams.forecast_range,
        hyperparams.ma7,
    )
    @assert size(train_dataset.data, 2) == Dates.value(hyperparams.train_range)
    @assert size(test_dataset.data, 2) == Dates.value(hyperparams.forecast_range)

    movement_range_data =
        experiment_movement_range(loc, first_date, last_date, hyperparams.ma7)
    @assert size(movement_range_data, 2) ==
            Dates.value(hyperparams.train_range) + Dates.value(hyperparams.forecast_range)

    social_proximity_data = experiment_social_proximity(
        loc,
        first_date - hyperparams.social_proximity_lag,
        last_date - hyperparams.social_proximity_lag,
        hyperparams.ma7,
    )
    @assert size(social_proximity_data, 2) ==
            Dates.value(hyperparams.train_range) + Dates.value(hyperparams.forecast_range)

    # build the model
    model = SEIRDFbMobility2(
        hyperparams.γ_bounds,
        hyperparams.λ_bounds,
        hyperparams.α_bounds,
        movement_range_data,
        social_proximity_data,
    )
    # get the initial states and available observations depending on the model type
    # and the considered location
    u0, vars, labels = experiment_SEIRD_initial_states(loc, train_dataset.data[:, 1])
    p0 = initparams(model, hyperparams.γ0, hyperparams.λ0, hyperparams.α0)
    train_data_min = minimum(train_dataset.data, dims = 2)
    train_data_max = maximum(train_dataset.data, dims = 2)
    lossfn =
        experiment_loss(train_dataset.tsteps, hyperparams.ζ, train_data_min, train_data_max)
    lossfn_regularized = function (ŷ, y, params)
        pnamed = namedparams(model, params)
        return lossfn(ŷ, y) + hyperparams.L2_λ * sum(abs2, pnamed.θ)
    end
    return model, u0, p0, lossfn_regularized, train_dataset, test_dataset, vars, labels
end

SEIRDFbMobility3Hyperparams = @NamedTuple begin
    L2_λ::Float32
    ζ::Float32
    γ0::Float32
    λ0::Float32
    α0::Float32
    β_bounds::Tuple{Float32,Float32}
    γ_bounds::Tuple{Float32,Float32}
    λ_bounds::Tuple{Float32,Float32}
    α_bounds::Tuple{Float32,Float32}
    train_range::Day
    forecast_range::Day
    social_proximity_lag::Day
    ma7::Bool
end

function setup_fbmobility3(loc::AbstractString, hyperparams::SEIRDFbMobility3Hyperparams)
    # get data for model
    train_dataset, test_dataset, first_date, last_date = experiment_covid19_data(
        loc,
        hyperparams.train_range,
        hyperparams.forecast_range,
        hyperparams.ma7,
    )
    @assert size(train_dataset.data, 2) == Dates.value(hyperparams.train_range)
    @assert size(test_dataset.data, 2) == Dates.value(hyperparams.forecast_range)

    movement_range_data =
        experiment_movement_range(loc, first_date, last_date, hyperparams.ma7)
    @assert size(movement_range_data, 2) ==
            Dates.value(hyperparams.train_range) + Dates.value(hyperparams.forecast_range)

    social_proximity_data = experiment_social_proximity(
        loc,
        first_date - hyperparams.social_proximity_lag,
        last_date - hyperparams.social_proximity_lag,
        hyperparams.ma7,
    )
    @assert size(social_proximity_data, 2) ==
            Dates.value(hyperparams.train_range) + Dates.value(hyperparams.forecast_range)

    # build the model
    model = SEIRDFbMobility3(
        hyperparams.β_bounds,
        hyperparams.γ_bounds,
        hyperparams.λ_bounds,
        hyperparams.α_bounds,
        movement_range_data,
        social_proximity_data,
    )
    # get the initial states and available observations depending on the model type
    # and the considered location
    u0, vars, labels = experiment_SEIRD_initial_states(loc, train_dataset.data[:, 1])
    p0 = initparams(model, hyperparams.γ0, hyperparams.λ0, hyperparams.α0)
    train_data_min = minimum(train_dataset.data, dims = 2)
    train_data_max = maximum(train_dataset.data, dims = 2)
    lossfn =
        experiment_loss(train_dataset.tsteps, hyperparams.ζ, train_data_min, train_data_max)
    lossfn_regularized = function (ŷ, y, params)
        pnamed = namedparams(model, params)
        return lossfn(ŷ, y) + hyperparams.L2_λ * sum(abs2, pnamed.θ)
    end
    return model, u0, p0, lossfn_regularized, train_dataset, test_dataset, vars, labels
end

SEIRDFbMobility4Hyperparams = @NamedTuple begin
    L2_λ::Float32
    ζ::Float32
    γ0::Float32
    λ0::Float32
    β_bounds::Tuple{Float32,Float32}
    γ_bounds::Tuple{Float32,Float32}
    λ_bounds::Tuple{Float32,Float32}
    α_bounds::Tuple{Float32,Float32}
    train_range::Day
    forecast_range::Day
    social_proximity_lag::Day
    ma7::Bool
end

function setup_fbmobility4(loc::AbstractString, hyperparams::SEIRDFbMobility4Hyperparams)
    # get data for model
    train_dataset, test_dataset, first_date, last_date = experiment_covid19_data(
        loc,
        hyperparams.train_range,
        hyperparams.forecast_range,
        hyperparams.ma7,
    )
    @assert size(train_dataset.data, 2) == Dates.value(hyperparams.train_range)
    @assert size(test_dataset.data, 2) == Dates.value(hyperparams.forecast_range)

    movement_range_data =
        experiment_movement_range(loc, first_date, last_date, hyperparams.ma7)
    @assert size(movement_range_data, 2) ==
            Dates.value(hyperparams.train_range) + Dates.value(hyperparams.forecast_range)

    social_proximity_data = experiment_social_proximity(
        loc,
        first_date - hyperparams.social_proximity_lag,
        last_date - hyperparams.social_proximity_lag,
        hyperparams.ma7,
    )
    @assert size(social_proximity_data, 2) ==
            Dates.value(hyperparams.train_range) + Dates.value(hyperparams.forecast_range)

    # build the model
    model = SEIRDFbMobility4(
        hyperparams.β_bounds,
        hyperparams.γ_bounds,
        hyperparams.λ_bounds,
        hyperparams.α_bounds,
        movement_range_data,
        social_proximity_data,
    )
    # get the initial states and available observations depending on the model type
    # and the considered location
    u0, vars, labels = experiment_SEIRD_initial_states(loc, train_dataset.data[:, 1])
    p0 = initparams(model, hyperparams.γ0, hyperparams.λ0)
    train_data_min = minimum(train_dataset.data, dims = 2)
    train_data_max = maximum(train_dataset.data, dims = 2)
    lossfn =
        experiment_loss(train_dataset.tsteps, hyperparams.ζ, train_data_min, train_data_max)
    lossfn_regularized = function (ŷ, y, params)
        pnamed = namedparams(model, params)
        return lossfn(ŷ, y) +
               hyperparams.L2_λ * (sum(abs2, pnamed.θ1) + sum(abs2, pnamed.θ2))
    end
    return model, u0, p0, lossfn_regularized, train_dataset, test_dataset, vars, labels
end

function experiment_train(
    uuid::AbstractString,
    setup::Function,
    configs::AbstractVector{TrainConfig},
    snapshots_dir::AbstractString;
    kwargs...,
)
    # get model and data
    model, u0, p0, lossfn, train_dataset, test_dataset, vars, _ = setup()
    # create a prediction model and loss function
    prob = ODEProblem(model, u0, train_dataset.tspan)
    predictor = Predictor(prob, vars)
    train_loss = Loss{true}(lossfn, predictor, train_dataset)
    test_loss = Loss{false}(rmse, predictor, test_dataset)
    # check if AD works
    dLdθ = Zygote.gradient(train_loss, p0)
    @assert !isnothing(dLdθ[1]) # gradient is computable
    @assert any(dLdθ[1] .!= 0.0) # not all gradients are 0

    minimizers =
        train_model(uuid, train_loss, test_loss, p0, configs, snapshots_dir; kwargs...)
    minimizer = last(minimizers)
    return minimizer, train_loss(minimizer)
end

function experiment_eval(
    uuid::AbstractString,
    setup::Function,
    snapshots_dir::AbstractString,
)
    # get model and data
    model, u0, _, _, train_dataset, test_dataset, vars, labels = setup()
    # create a prediction model
    prob = ODEProblem(model, u0, train_dataset.tspan)
    predictor = Predictor(prob, vars)

    eval_config = EvalConfig([mae, mape, rmse], [7, 14, 21, 28], labels)
    for fpath ∈ lookup_saved_params(snapshots_dir)
        dataname, datatype, _ = rsplit(basename(fpath), ".", limit = 3)
        if !startswith(dataname, uuid)
            continue
        end

        if datatype == "losses"
            train_losses, test_losses = Serialization.deserialize(fpath)
            fig = plot_losses(train_losses, test_losses)
            save(joinpath(snapshots_dir, "$dataname.losses.png"), fig)

        elseif datatype == "params"
            minimizer = Serialization.deserialize(fpath)
            fig_forecasts, df_errors = evaluate_model(
                eval_config,
                predictor,
                minimizer,
                train_dataset,
                test_dataset,
            )
            save(joinpath(snapshots_dir, "$dataname.forecasts.png"), fig_forecasts)
            save_dataframe(df_errors, joinpath(snapshots_dir, "$dataname.errors.csv"))

            ℜe1 = ℜe(model, u0, minimizer, train_dataset.tspan, train_dataset.tsteps)
            ℜe2 = ℜe(model, u0, minimizer, test_dataset.tspan, test_dataset.tsteps)
            fig_ℜe = plot_ℜe([ℜe1; ℜe2], train_dataset.tspan[2])
            save(joinpath(snapshots_dir, "$uuid.R_effective.png"), fig_ℜe)
        end
    end

    return nothing
end

JSON.lower(x::AbstractVector{TrainConfig}) = x

JSON.lower(x::TrainConfig{ADAM}) =
    (name = x.name, maxiters = x.maxiters, eta = x.optimizer.eta, beta = x.optimizer.beta)

JSON.lower(x::TrainConfig{BFGS{IL,L,H,T,TM}}) where {IL,L,H,T,TM} = (
    name = x.name,
    maxiters = x.maxiters,
    alphaguess = IL.name.wrapper,
    linesearch = L.name.wrapper,
    initial_invH = x.optimizer.initial_invH,
    initial_stepnorm = x.optimizer.initial_stepnorm,
    manifold = TM.name.wrapper,
)

const LK_EVALUATION = ReentrantLock()

function experiment_run(
    model_name::AbstractString,
    model_setup::Function,
    locations::AbstractVector{<:AbstractString},
    hyperparams::NamedTuple,
    train_configs::AbstractVector{<:TrainConfig};
    multithreading::Bool,
    savedir::AbstractString,
    kwargs...,
)
    minimizers = Vector{Float32}[]
    final_losses = Float32[]
    lk = ReentrantLock()

    run = function (loc)
        timestamp = Dates.format(now(), "yyyymmddHHMMSS")
        uuid = "$timestamp.$model_name.$loc"
        setup = () -> model_setup(loc, hyperparams)
        snapshots_dir = joinpath(savedir, loc)
        if !isdir(snapshots_dir)
            mkpath(snapshots_dir)
        end

        @info "Running $uuid"
        write(
            joinpath(snapshots_dir, "$uuid.hyperparams.json"),
            json((; hyperparams..., train_configs), 4),
        )
        minimizer, final_loss =
            experiment_train(uuid, setup, train_configs, snapshots_dir; kwargs...)

        # access shared arrays
        lock(lk)
        try
            push!(minimizers, minimizer)
            push!(final_losses, final_loss)
        finally
            unlock(lk)
        end

        # program crashes when multiple threads trying to plot at the same time
        lock(LK_EVALUATION)
        try
            experiment_eval(uuid, setup, snapshots_dir)
        finally
            unlock(LK_EVALUATION)
        end

        @info "Finished running $uuid"
        return nothing
    end

    if multithreading
        Threads.@threads for loc ∈ locations
            run(loc)
        end
    else
        for loc ∈ locations
            run(loc)
        end
    end

    return minimizers, final_losses
end
