using Dates, LinearAlgebra, Statistics, Serialization
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

include("trainalgs.jl")

function experiment_covid19_data(loc::AbstractString, train_range::Day, forecast_range::Day)
    df = get_prebuilt_covid_timeseries(loc)
    # derive newly confirmed from total confirmed
    df[!, :confirmed] .= df[!, :confirmed_total]
    df[2:end, :confirmed] .= diff(df[!, :confirmed_total])
    # ensure data type stability
    datacols = names(df, Not(:date))
    df[!, datacols] .= Float64.(df[!, datacols])

    if loc == Covid19ModelVN.LOC_CODE_VIETNAM || loc ∈ keys(Covid19ModelVN.LOC_NAMES_VN)
        # we considered 27th April 2021 to be the start of the outbreak in Vietnam
        bound!(df, :date, Date(2021, 4, 27), typemax(Date))
    elseif loc == Covid19ModelVN.LOC_CODE_UNITED_STATES ||
        loc ∈ keys(Covid19ModelVN.LOC_NAMES_US)
        # we considered 1st July 2021 to be the start of the outbreak in the US
        bound!(df, :date, Date(2021, 7, 1), typemax(Date))
    end

    # select data starting from when total deaths >= 1 and confirmed >= 500
    dates =
        subset(
            df,
            :deaths_total => x -> x .>= 10,
            :confirmed_total => x -> x .>= 500;
            view=true,
        ).date
    first_date = first(dates)
    split_date = first_date + train_range - Day(1)
    last_date = split_date + forecast_range

    @info("Getting Covid-19 data", loc, first_date, split_date, last_date)

    # smooth out weekly seasonality
    moving_average!(df, datacols, 7)
    # observable compartments
    conf = TimeseriesConfig(df, "date", ["deaths_total", "confirmed", "confirmed_total"])
    return conf, first_date, split_date, last_date
end

function experiment_movement_range(
    loc::AbstractString, first_date::Date, split_date::Date, last_date::Date, lag::Day
)
    first_date -= lag
    split_date -= lag
    last_date -= lag

    df = get_prebuilt_movement_range(loc)
    cols = ["all_day_bing_tiles_visited_relative_change", "all_day_ratio_single_tile_users"]
    df[!, cols] .= Float64.(df[!, cols])
    # smooth out weekly seasonality
    moving_average!(df, cols, 7)
    data = load_timeseries(TimeseriesConfig(df, "ds", cols), first_date, last_date)
    # min-max scaling
    data_train = @view data[:, 1:Dates.value(split_date - first_date)]
    data_train_min = minimum(data_train; dims=2)
    data_train_max = maximum(data_train; dims=2)
    data .= (data .- data_train_min) ./ (data_train_max .- data_train_min)
    return data
end

function experiment_social_proximity(
    loc::AbstractString, first_date::Date, split_date::Date, last_date::Date, lag::Day
)
    first_date -= lag
    split_date -= lag
    last_date -= lag

    df, col = get_prebuilt_social_proximity(loc)
    df[!, col] .= Float64.(df[!, col])
    # smooth out weekly seasonality
    moving_average!(df, col, 7)
    data = load_timeseries(TimeseriesConfig(df, "date", [col]), first_date, last_date)
    # min-max scaling
    data_train = @view data[:, 1:Dates.value(split_date - first_date)]
    data_train_min = minimum(data_train; dims=2)
    data_train_max = maximum(data_train; dims=2)
    data .= (data .- data_train_min) ./ (data_train_max .- data_train_min)
    return data
end

function experiment_SEIRD_initial_states(
    loc::AbstractString, df_first_date::AbstractDataFrame
)
    population = get_prebuilt_population(loc)
    I0 = df_first_date.confirmed[1] # infective individuals
    D0 = df_first_date.deaths_total[1] # total deaths
    C0 = df_first_date.confirmed[1] # new cases
    T0 = df_first_date.confirmed_total[1] # total cases
    R0 = T0 - I0 - D0 # recovered individuals
    N0 = population - D0 # effective population
    E0 = I0 * 5 # exposed individuals
    S0 = population - E0 - df_first_date.confirmed_total[1] # susceptible individuals
    # initial state
    u0 = Float64[S0, E0, I0, R0, D0, N0, C0, T0]
    vars = [5, 7, 8]
    labels = ["deaths", "new cases", "total cases"]
    return u0, vars, labels
end

normed_ld(a, b) = abs(norm(a) - norm(b)) / (norm(a) + norm(b))
cosine_similarity(a, b) = dot(a, b) / (norm(a) * norm(b))
cosine_distance(a, b) = (1 - cosine_similarity(a, b)) / 2

function experiment_loss_polar(w::Tuple{R,R}, ζ::R) where {R<:Real}
    lossfn = function (ŷ::AbstractArray{R}, y, tsteps) where {R<:Real}
        s = zero(R)
        sz = size(y)
        @inbounds for j in 1:sz[2]
            @views s += w[1] * normed_ld(y[:, j], ŷ[:, j])
            @views s += w[2] * cosine_distance(y[:, j], ŷ[:, j])
            s *= exp(ζ * tsteps[j])
        end
        return s
    end
    return lossfn
end

function experiment_loss_sse(
    min::AbstractVector{R}, max::AbstractVector{R}, ζ::R
) where {R<:Real}
    scale = max .- min
    lossfn = function (ŷ::AbstractArray{R}, y, tsteps) where {R<:Real}
        s = zero(R)
        sz = size(y)
        @inbounds for j in 1:sz[2], i in 1:sz[1]
            s += ((ŷ[i, j] - y[i, j]) / scale[i])^2 / sz[2] * exp(ζ * tsteps[j])
        end
        return s
    end
    return lossfn
end

function setup_baseline(
    loc::AbstractString;
    γ0::Float64,
    λ0::Float64,
    β_bounds::Tuple{Float64,Float64},
    γ_bounds::Tuple{Float64,Float64},
    λ_bounds::Tuple{Float64,Float64},
    α_bounds::Tuple{Float64,Float64},
    train_range::Day,
    forecast_range::Day,
    loss_type::Symbol,
    loss_regularization::Float64,
    loss_time_weighting::Float64,
)
    # get data for model
    dataconf, first_date, split_date, last_date = experiment_covid19_data(
        loc, train_range, forecast_range
    )
    train_dataset, test_dataset = train_test_split(
        dataconf, first_date, split_date, last_date
    )
    @assert size(train_dataset.data, 2) == Dates.value(train_range)
    @assert size(test_dataset.data, 2) == Dates.value(forecast_range)

    # initialize the model
    model = SEIRDBaseline(
        β_bounds,
        γ_bounds,
        λ_bounds,
        α_bounds,
        Float64(get_prebuilt_population(loc)),
        Float64(Dates.value(split_date - first_date)),
    )
    # get the initial states and available observations depending on the model type
    # and the considered location
    u0, vars, labels = experiment_SEIRD_initial_states(
        loc, subset(dataconf.df, :date => x -> x .== first_date; view=true)
    )
    p0 = initparams(model, γ0, λ0)

    lossfn_inner = if loss_type == :polar
        experiment_loss_polar((0.5, 0.5), loss_time_weighting)
    elseif loss_type == :sse
        min = vec(minimum(train_dataset.data; dims=2))
        max = vec(maximum(train_dataset.data; dims=2))
        experiment_loss_sse(min, max, loss_time_weighting)
    else
        error("Invalid loss function type")
    end

    lossfn = function (ŷ, y, params, tsteps)
        pnamed = namedparams(model, params)
        return lossfn_inner(ŷ, y, tsteps) +
               loss_regularization / (2 * size(y, 2)) *
               (sum(abs2, pnamed.θ1) + sum(abs2, pnamed.θ2))
    end

    return model, u0, p0, lossfn, train_dataset, test_dataset, vars, labels
end

function setup_fbmobility1(
    loc::AbstractString;
    γ0::Float64,
    λ0::Float64,
    β_bounds::Tuple{Float64,Float64},
    γ_bounds::Tuple{Float64,Float64},
    λ_bounds::Tuple{Float64,Float64},
    α_bounds::Tuple{Float64,Float64},
    train_range::Day,
    forecast_range::Day,
    movement_range_lag::Day,
    loss_type::Symbol,
    loss_regularization::Float64,
    loss_time_weighting::Float64,
)
    # get data for model
    dataconf, first_date, split_date, last_date = experiment_covid19_data(
        loc, train_range, forecast_range
    )
    train_dataset, test_dataset = train_test_split(
        dataconf, first_date, split_date, last_date
    )
    @assert size(train_dataset.data, 2) == Dates.value(train_range)
    @assert size(test_dataset.data, 2) == Dates.value(forecast_range)

    movement_range_data = experiment_movement_range(
        loc, first_date, split_date, last_date, movement_range_lag
    )
    @assert size(movement_range_data, 2) ==
        Dates.value(train_range) + Dates.value(forecast_range)

    # initialize the model
    model = SEIRDFbMobility1(
        β_bounds,
        γ_bounds,
        λ_bounds,
        α_bounds,
        Float64(get_prebuilt_population(loc)),
        Float64(Dates.value(split_date - first_date)),
        movement_range_data,
    )
    # get the initial states and available observations depending on the model type
    # and the considered location
    u0, vars, labels = experiment_SEIRD_initial_states(
        loc, subset(dataconf.df, :date => x -> x .== first_date; view=true)
    )
    p0 = initparams(model, γ0, λ0)

    lossfn_inner = if loss_type == :polar
        experiment_loss_polar((0.5, 0.5), loss_time_weighting)
    elseif loss_type == :sse
        min = vec(minimum(train_dataset.data; dims=2))
        max = vec(maximum(train_dataset.data; dims=2))
        experiment_loss_sse(min, max, loss_time_weighting)
    else
        error("Invalid loss function type")
    end

    lossfn = function (ŷ, y, params, tsteps)
        pnamed = namedparams(model, params)
        return lossfn_inner(ŷ, y, tsteps) +
               loss_regularization / (2 * size(y, 2)) *
               (sum(abs2, pnamed.θ1) + sum(abs2, pnamed.θ2))
    end

    return model, u0, p0, lossfn, train_dataset, test_dataset, vars, labels
end

function setup_fbmobility2(
    loc::AbstractString;
    γ0::Float64,
    λ0::Float64,
    β_bounds::Tuple{Float64,Float64},
    γ_bounds::Tuple{Float64,Float64},
    λ_bounds::Tuple{Float64,Float64},
    α_bounds::Tuple{Float64,Float64},
    train_range::Day,
    forecast_range::Day,
    movement_range_lag::Day,
    social_proximity_lag::Day,
    loss_type::Symbol,
    loss_regularization::Float64,
    loss_time_weighting::Float64,
)
    # get data for model
    dataconf, first_date, split_date, last_date = experiment_covid19_data(
        loc, train_range, forecast_range
    )
    train_dataset, test_dataset = train_test_split(
        dataconf, first_date, split_date, last_date
    )
    @assert size(train_dataset.data, 2) == Dates.value(train_range)
    @assert size(test_dataset.data, 2) == Dates.value(forecast_range)

    movement_range_data = experiment_movement_range(
        loc, first_date, split_date, last_date, movement_range_lag
    )
    @assert size(movement_range_data, 2) ==
        Dates.value(train_range) + Dates.value(forecast_range)

    social_proximity_data = experiment_social_proximity(
        loc, first_date, split_date, last_date, social_proximity_lag
    )
    @assert size(social_proximity_data, 2) ==
        Dates.value(train_range) + Dates.value(forecast_range)

    # build the model
    model = SEIRDFbMobility2(
        β_bounds,
        γ_bounds,
        λ_bounds,
        α_bounds,
        Float64(get_prebuilt_population(loc)),
        Float64(Dates.value(last_date - first_date)),
        movement_range_data,
        social_proximity_data,
    )
    # get the initial states and available observations depending on the model type
    # and the considered location
    u0, vars, labels = experiment_SEIRD_initial_states(
        loc, subset(dataconf.df, :date => x -> x .== first_date; view=true)
    )
    p0 = initparams(model, γ0, λ0)

    lossfn_inner = if loss_type == :polar
        experiment_loss_polar((0.5, 0.5), loss_time_weighting)
    elseif loss_type == :sse
        min = vec(minimum(train_dataset.data; dims=2))
        max = vec(maximum(train_dataset.data; dims=2))
        experiment_loss_sse(min, max, loss_time_weighting)
    else
        error("Invalid loss function type")
    end

    lossfn = function (ŷ, y, params, tsteps)
        pnamed = namedparams(model, params)
        return lossfn_inner(ŷ, y, tsteps) +
               loss_regularization / (2 * size(y, 2)) *
               (sum(abs2, pnamed.θ1) + sum(abs2, pnamed.θ2))
    end

    return model, u0, p0, lossfn, train_dataset, test_dataset, vars, labels
end

function experiment_eval(
    setup::Function,
    forecast_horizons::AbstractVector{<:Integer},
    snapshots_dir::AbstractString,
)
    # get model and data
    model, u0, _, _, train_dataset, test_dataset, vars, labels = setup()
    # create a prediction model
    prob = ODEProblem(model, u0, train_dataset.tspan)
    predictor = Predictor(prob, vars)

    eval_config = EvalConfig([mae, mape, rmse], forecast_horizons, labels)
    for fpath in lookup_saved_params(snapshots_dir)
        dataname, datatype, _ = rsplit(basename(fpath), "."; limit=3)
        if datatype == "losses"
            train_losses, test_losses = Serialization.deserialize(fpath)
            fig_losses = plot_losses(train_losses, test_losses)
            fpath_losses = joinpath(snapshots_dir, "$dataname.losses.pdf")
            @info("Generating losses plot", fpath_losses)
            save(fpath_losses, fig_losses)

        elseif datatype == "params"
            minimizer = Serialization.deserialize(fpath)
            fit = Array(predictor(minimizer, train_dataset.tspan, train_dataset.tsteps))
            pred = Array(predictor(minimizer, test_dataset.tspan, test_dataset.tsteps))
            df_fit = DataFrame(fit', eval_config.labels)
            df_pred = DataFrame(pred', eval_config.labels)
            fpath_fit = joinpath(snapshots_dir, "$dataname.fit.csv")
            fpath_pred = joinpath(snapshots_dir, "$dataname.predictions.csv")
            @info("Generating final fit and predictions", fpath_fit, fpath_pred,)
            save_dataframe(df_fit, fpath_fit)
            save_dataframe(df_pred, fpath_pred)

            fig_forecasts, df_errors, df_time_steps_errors = evaluate_model(
                eval_config, predictor, minimizer, train_dataset, test_dataset
            )
            fpath_forecasts = joinpath(snapshots_dir, "$dataname.forecasts.pdf")
            fpath_errors = joinpath(snapshots_dir, "$dataname.errors.csv")
            fpath_time_steps_errors = joinpath(
                snapshots_dir, "$dataname.time_steps_errors.csv"
            )
            @info(
                "Generating forecasts plot and errors",
                fpath_forecasts,
                fpath_errors,
                fpath_time_steps_errors,
            )
            save(fpath_forecasts, fig_forecasts)
            save_dataframe(df_errors, fpath_errors)
            save_dataframe(df_time_steps_errors, fpath_time_steps_errors)

            Re1 = Re(model, u0, minimizer, train_dataset.tspan, train_dataset.tsteps)
            Re2 = Re(model, u0, minimizer, test_dataset.tspan, test_dataset.tsteps)
            Re_ = [Re1; Re2]
            df_Re = DataFrame([Re_], ["Re"])
            fig_Re = plot_Re(Re_, train_dataset.tspan[2])
            fpath_Re = joinpath(snapshots_dir, "$dataname.R_effective.csv")
            fpath_Re_fig = joinpath(snapshots_dir, "$dataname.R_effective.pdf")
            @info("Generating effective reproduction number plot", fpath_Re, fpath_Re_fig)
            save_dataframe(df_Re, fpath_Re)
            save(fpath_Re_fig, fig_Re)

            # the fatality rate in this model changes over time
            αt1 = fatality_rate(
                model, u0, minimizer, train_dataset.tspan, train_dataset.tsteps
            )
            αt2 = fatality_rate(
                model, u0, minimizer, test_dataset.tspan, test_dataset.tsteps
            )
            αt = [αt1; αt2]
            df_αt = DataFrame([αt], ["αt"])
            fig_αt = plot_fatality_rate(αt, train_dataset.tspan[2])
            fpath_αt = joinpath(snapshots_dir, "$dataname.fatality_rate.csv")
            fpath_αt_fig = joinpath(snapshots_dir, "$dataname.fatality_rate.pdf")
            @info("Generating effective fatality rate plot", fpath_αt, fpath_αt_fig)
            save_dataframe(df_αt, fpath_αt)
            save(fpath_αt_fig, fig_αt)

        elseif datatype == "forecasts"
            fit, pred = Serialization.deserialize(fpath)
            obs_fit = Observable(fit[1])
            obs_pred = Observable(pred[1])
            fig_animation = plot_forecasts(
                eval_config, obs_fit, obs_pred, train_dataset, test_dataset
            )
            fpath_animation = joinpath(snapshots_dir, "$dataname.forecasts.mkv")
            @info("Generating fit animation", fpath_animation)
            record(
                fig_animation, fpath_animation, zip(fit, pred); framerate=60
            ) do (fit, pred)
                obs_fit[] = fit
                obs_pred[] = pred
                return contents(fig_animation[:, :])
            end
        end
    end

    return nothing
end

function experiment_run(
    model_name::AbstractString,
    model_setup::Function,
    locations::AbstractVector{<:AbstractString},
    hyperparams::NamedTuple,
    train_algorithm::Symbol,
    train_config::Dict{Symbol,Any};
    forecast_horizons::AbstractVector{<:Integer},
    savedir::AbstractString,
    show_progress::Bool,
    make_animation::Bool,
    multithreading::Bool,
)
    batch_timestamp = Dates.format(now(), "yyyymmddHHMMSS")

    lk_eval = ReentrantLock()
    minimizers = Vector{Float64}[]
    final_losses = Float64[]
    queue_eval = Tuple{String,Function,Vector{Int},String}[]

    runexp = function (loc)
        snapshots_dir = joinpath(savedir, batch_timestamp, loc)
        !isdir(snapshots_dir) && mkpath(snapshots_dir)

        timestamp = Dates.format(now(), "yyyymmddHHMMSS")
        uuid = "$timestamp.$model_name.$loc"
        setup = () -> model_setup(loc; hyperparams...)

        write(
            joinpath(snapshots_dir, "$uuid.hyperparams.json"),
            json((; hyperparams..., train_algorithm, train_config...), 4),
        )

        trainfn = if train_algorithm == :train_growing_trajectory
            train_growing_trajectory
        elseif train_algorithm == :train_growing_trajectory_two_stages
            train_growing_trajectory_two_stages
        elseif train_algorithm == :train_whole_trajectory
            train_whole_trajectory
        elseif train_algorithm == :train_whole_trajectory_two_stages
            train_whole_trajectory_two_stages
        end

        shared_progress = if multithreading && show_progress
            ProgressUnknown(; showspeed=true)
        else
            nothing
        end

        minimizer, eval_losses, _ = trainfn(
            uuid,
            setup;
            snapshots_dir,
            show_progress,
            shared_progress,
            make_animation,
            train_config...,
        )

        lock(lk_eval) do
            push!(minimizers, minimizer)
            push!(final_losses, last(eval_losses))
            return push!(queue_eval, (uuid, setup, forecast_horizons, snapshots_dir))
        end
        @info("Finish training session", uuid)
    end

    if multithreading
        Threads.@threads for loc in locations
            try
                runexp(loc)
            catch e
                e isa InterruptException && rethrow(e)
                @warn stacktrace() error = e
            end
        end
    else
        for loc in locations
            try
                runexp(loc)
            catch e
                e isa InterruptException && rethrow(e)
                @warn stacktrace() error = e
            end
        end
    end

    for (uuid, setup, forecast_horizons, snapshots_dir) in queue_eval
        experiment_eval(setup, forecast_horizons, snapshots_dir)
        @info("Finish evaluation", uuid)
    end

    return minimizers, final_losses
end
