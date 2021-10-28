export TrainSession,
    EvalConfig,
    Predictor,
    Loss,
    TrainCallback,
    TrainCallbackConfig,
    train_model,
    evaluate_model,
    calculate_forecasts_errors,
    plot_forecasts,
    mae,
    mape,
    rmse,
    rmsle

using Serialization,
    Statistics,
    ProgressMeter,
    Plots,
    DataFrames,
    CSV,
    OrdinaryDiffEq,
    DiffEqFlux,
    Covid19ModelVN

"""
Specifications for a model tranining session

# Arguments

+ `name`: Session name
+ `optimizer`: The optimizer that will run in the session
+ `maxiters`: Maximum number of iterations to run the optimizer
"""
struct TrainSession{O}
    name::AbstractString
    optimizer::O
    maxiters::Integer
end

"""
A struct that solves the underlying DiffEq problem and returns the solution when it is called

# Fields

* `problem`: the problem that will be solved
* `solver`: the numerical solver that will be used to calculate the DiffEq solution
"""
struct Predictor{S,A}
    problem::ODEProblem
    solver::S
    sensealg::A
end

"""
Construct a new default `Predictor` using the problem defined by the given model

# Argument

+ `model`: a model containing a problem that can be solved
"""
Predictor(model::AbstractCovidModel) =
    Predictor(model.problem, Tsit5(), InterpolatingAdjoint(autojacvec = ReverseDiffVJP()))

"""
Call an object of struct `CovidModelPredict` to solve the underlying DiffEq problem

# Arguments

* `params`: the set of parameters of the system
* `tspan`: the time span of the problem
* `saveat`: the collocation coordinates
"""
function (p::Predictor)(
    params::AbstractVector{<:Real},
    tspan::Tuple{<:Real,<:Real},
    saveat::Union{<:Real,AbstractVector{<:Real},StepRange,StepRangeLen},
)
    problem = remake(p.problem, p = params, tspan = tspan)
    return solve(problem, p.solver, saveat = saveat, sensealg = p.sensealg)
end

"""
A callable struct that uses `metric_fn` to calculate the loss between the output of
`predict` and `dataset`.

# Fields

* `metric_fn`: a function that computes the error between two data arrays
* `predict_fn`: the time span that the ODE solver will be run on
* `dataset`: the dataset that contains the ground truth data
* `vars`: indices of the states that will be used to calculate the loss
"""
struct Loss
    metric_fn::Function
    predict_fn::Predictor
    dataset::UDEDataset
    vars::Union{<:Integer,AbstractVector{<:Integer},OrdinalRange}
end

"""
Call an object of the `Loss` struct on a set of parameters to get the loss scalar

# Arguments

* `params`: the set of parameters of the model
"""
function (l::Loss)(params::AbstractVector{<:Real})
    sol = l.predict_fn(params, l.dataset.tspan, l.dataset.tsteps)
    if sol.retcode != :Success
        # Unstable trajectories => hard penalize
        return Inf
    end

    pred = @view sol[l.vars, :]
    if size(pred) != size(l.dataset.data)
        # Unstable trajectories / Wrong inputs
        return Inf
    end

    return l.metric_fn(pred, l.dataset.data)
end

"""
State of the callback struct

# Fields

* `iters`: number have iterations that have been run
* `progress`: the progress meter that keeps track of the process
* `train_losses`: collected training losses at each interval
* `test_losses`: collected testing losses at each interval
* `minimizer`: current best set of parameters
* `minimizer_loss`: loss value of the current best set of parameters
"""
mutable struct TrainCallbackState
    iters::Integer
    progress::Progress
    train_losses::AbstractVector{<:Real}
    test_losses::AbstractVector{<:Real}
    minimizer::AbstractVector{<:Real}
    minimizer_loss::Real
end

"""
Construct a new `TrainCallbackState` with the progress bar set to `maxiters`
and other fields set to their default values

# Arguments

+ `maxiters`: Maximum number of iterrations that the optimizer will run
"""
TrainCallbackState(maxiters::Integer) = TrainCallbackState(
    0,
    Progress(maxiters, showspeed = true),
    Float64[],
    Float64[],
    Float64[],
    Inf,
)

"""
Configuration of the callback struct

# Fields

* `test_loss_fn`: a callable for calculating the testing loss value
* `losses_plot_fpath`: file path to the saved losses figure
* `losses_plot_interval`: interval for collecting losses and plot the losses figure
* `params_save_fpath`: file path to the serialized current best set of parameters
* `params_save_interval`: interval for saving the current best set of parameters
"""
struct TrainCallbackConfig
    test_loss_fn::Union{Nothing,Loss}
    losses_plot_fpath::Union{Nothing,<:AbstractString}
    losses_plot_interval::Integer
    params_save_fpath::Union{Nothing,<:AbstractString}
    params_save_interval::Integer
end

"""
Contruct a default `TrainCallbackConfig`
"""
TrainCallbackConfig() =
    TrainCallbackConfig(nothing, nothing, typemax(Int), nothing, typemax(Int))

"""
A callable struct that is used for handling callback for `sciml_train`
"""
mutable struct TrainCallback
    state::TrainCallbackState
    config::TrainCallbackConfig
end

"""
Create a callback for `sciml_train`

# Arguments

* `maxiters`: max number of iterations the optimizer will run
* `config`: callback configurations
"""
TrainCallback(maxiters::Integer, config::TrainCallbackConfig = TrainCallbackConfig()) =
    TrainCallback(TrainCallbackState(maxiters), config)

"""
Call an object of type `TrainCallback`

# Arguments

* `params`: the model's parameters
* `train_loss`: loss from the training step
"""
function (cb::TrainCallback)(params::AbstractVector{<:Real}, train_loss::Real)
    test_loss = if !isnothing(cb.config.test_loss_fn)
        cb.config.test_loss_fn(params)
    end

    if train_loss < cb.state.minimizer_loss
        cb.state.minimizer_loss = train_loss
        cb.state.minimizer = params
    end

    cb.state.iters += 1
    if cb.state.iters % cb.config.losses_plot_interval == 0 &&
       !isnothing(cb.config.losses_plot_fpath)
        append!(cb.state.train_losses, train_loss)
        append!(cb.state.test_losses, test_loss)
        plt = plot(
            [cb.state.train_losses, cb.state.test_losses],
            labels = ["train loss" "test loss"],
            legend = :outerright,
        )
        savefig(plt, cb.config.losses_plot_fpath)
    end
    if cb.state.iters % cb.config.params_save_interval == 0 &&
       !isnothing(cb.config.params_save_fpath)
        Serialization.serialize(cb.config.params_save_fpath, cb.state.minimizer)
    end

    next!(
        cb.state.progress,
        showvalues = [:train_loss => train_loss, :test_loss => test_loss],
    )
    return false
end

"""
Find a set of paramters that minimizes the loss function defined by `train_loss_fn`, starting from
the initial set of parameters `params`.

# Arguments

+ `train_loss_fn`: a function that will be minimized
+ `test_loss_fn`: a function for evaluating the model on out-of-sample data
+ `params`: the initial set of parameters
+ `sessions`: a collection of optimizers and settings used for training the model
+ `snapshots_dir`: a directory for saving the model parameters and training losses
"""
function train_model(
    train_loss_fn::Loss,
    test_loss_fn::Loss,
    params::AbstractVector{<:Real},
    sessions::AbstractVector{TrainSession},
    snapshots_dir::AbstractString,
)
    for sess ∈ sessions
        losses_plot_fpath = get_losses_plot_fpath(snapshots_dir, sess.name)
        params_save_fpath = get_params_save_fpath(snapshots_dir, sess.name)
        cb = TrainCallback(
            sess.maxiters,
            TrainCallbackConfig(
                train_loss_fn,
                losses_plot_fpath,
                div(sess.maxiters, 20),
                params_save_fpath,
                div(sess.maxiters, 20),
            ),
        )

        @info "Running $(sess.name)"
        try
            DiffEqFlux.sciml_train(
                test_loss_fn,
                params,
                sess.optimizer,
                maxiters = sess.maxiters,
                cb = cb,
            )
        catch e
            @error e
            if isa(e, InterruptException)
                rethrow(e)
            end
        end

        params = cb.state.minimizer
        Serialization.serialize(params_save_fpath, params)
    end
    return nothing
end

"""
A struct for holding general configuration for the evaluation process

# Arguments

+ `metric_fns`: a list of metric function that will be used to compute the model errors
+ `forecast_ranges`: a list of different time ranges on which the model's prediction will be evaluated
+ `vars`: indices of the model's states that will be evaluated
+ `labels`: names of the evaluated model's states
"""
struct EvalConfig
    metric_fns::AbstractVector{Function}
    forecast_ranges::AbstractVector{<:Integer}
    vars::Union{<:Integer,AbstractVector{<:Integer},OrdinalRange}
    labels::AbstractVector{<:AbstractString}
end

"""
Evaluate the model by calculating the errors and draw plot againts ground truth data

# Arguments

+ `predict_fn`: the function that produce the model's prediction
+ `train_dataset`: ground truth data on which the model was trained
+ `test_dataset`: ground truth data that the model has not seen
+ `config`: the configuration for the evalution process
+ `snapshots_dir`: the directory of where the model's paramters are saved
"""
function evaluate_model(
    predict_fn::Predictor,
    train_dataset::UDEDataset,
    test_dataset::UDEDataset,
    config::EvalConfig,
    snapshots_dir::AbstractString,
)
    fpaths_params, uuids = lookup_saved_params(snapshots_dir)
    for (fpath_params, uuid) ∈ zip(fpaths_params, uuids)
        fit, pred = try
            minimizer = Serialization.deserialize(fpath_params)
            fit = predict_fn(minimizer, train_dataset.tspan, train_dataset.tsteps)
            pred = predict_fn(minimizer, test_dataset.tspan, test_dataset.tsteps)
            (fit, pred)
        catch e
            if isa(e, InterruptException)
                rethrow(e)
            end
            @warn e
            continue
        end

        csv_fpath = joinpath(snapshots_dir, "$uuid.evaluate.csv")
        if !isfile(csv_fpath)
            df = calculate_forecasts_errors(pred, test_dataset, config)
            CSV.write(csv_fpath, df)
        end

        fig_fpath = joinpath(snapshots_dir, "$uuid.evaluate.png")
        if !isfile(fig_fpath)
            plt = plot_forecasts(fit, pred, train_dataset, test_dataset, config)
            savefig(plt, fig_fpath)
        end
    end
end

"""
Calculate the forecast error based on the model prediction and the ground truth data
for that day

# Arguments

* `pred`: prediction made by the model
* `test_dataset`: ground truth data for the forecasted period
* `config`: configuration for evaluation
"""
function calculate_forecasts_errors(
    pred::ODESolution,
    test_dataset::UDEDataset,
    config::EvalConfig,
)
    horizons = repeat(config.forecast_ranges, inner = length(config.metric_fns))
    metrics = repeat(map(string, config.metric_fns), length(config.forecast_ranges))
    errors = reshape(
        [
            metric_fn(pred[var, 1:days], test_dataset.data[col, 1:days]) for
            metric_fn ∈ config.metric_fns, days ∈ config.forecast_ranges,
            (col, var) ∈ enumerate(config.vars)
        ],
        length(config.metric_fns) * length(config.forecast_ranges),
        length(config.vars),
    )
    df1 = DataFrame([horizons metrics], [:horizon, :metric])
    df2 = DataFrame(errors, config.labels)
    return [df1 df2]
end

"""
Plot the forecasted values produced by `predict_fn` against the ground truth data and calculated the error for each
forecasted value using `metric_fn`.

# Arguments

* `fit`: the solution returned by the model on the fit data
* `pred`: prediction made by the model
* `train_dataset`: the data that was used to train the model
* `test_dataset`: ground truth data for the forecasted period
* `config`: configuration for evaluation
"""
function plot_forecasts(
    fit::ODESolution,
    pred::ODESolution,
    train_dataset::UDEDataset,
    test_dataset::UDEDataset,
    config::EvalConfig,
)
    plts = []
    for days ∈ config.forecast_ranges,
        (col, (var, label)) ∈ enumerate(zip(config.vars, config.labels))

        plt = plot(title = "$days-day forecast", legend = :outertop, xrotation = 45)
        scatter!(
            [train_dataset.data[col, :]; test_dataset.data[col, 1:days]],
            label = label,
            fillcolor = nothing,
        )
        plot!([fit[var, :]; pred[var, 1:days]], label = "forecast $label", lw = 2)
        vline!([train_dataset.tspan[2]], color = :black, ls = :dash, label = nothing)
        push!(plts, plt)
    end

    nforecasts = length(config.forecast_ranges)
    nvariables = length(config.vars)
    return plot(
        plts...,
        layout = (nforecasts, nvariables),
        size = (300 * nvariables, 300 * nforecasts),
    )
end

"""
Calculate the mean absolute error between 2 values. Note that the input arguments must be of the same size.
The function does not check if the inputs are valid and may produces erroneous output.
"""
mae(ŷ::AbstractArray{<:Real}, y::AbstractArray{<:Real}) = mean(abs, (ŷ .- y))

"""
Calculate the mean absolute percentge error between 2 values. Note that the input arguments must be of the same size.
The function does not check if the inputs are valid and may produces erroneous output.
"""
mape(ŷ::AbstractArray{<:Real}, y::AbstractArray{<:Real}) = 100 * mean(abs, (ŷ .- y) ./ y)

"""
Calculate the root mean squared error between 2 values. Note that the input arguments must be of the same size.
The function does not check if the inputs are valid and may produces erroneous output.
"""
rmse(ŷ::AbstractArray{<:Real}, y::AbstractArray{<:Real}) = sqrt(mean(abs2, ŷ .- y))

"""
Calculate the root mean squared log error between 2 values. Note that the input arguments must be of the same size.
The function does not check if the inputs are valid and may produces erroneous output.
"""
rmsle(ŷ::AbstractArray{<:Real}, y::AbstractArray{<:Real}) =
    sqrt(mean(abs2, log.(ŷ .+ 1) .- log.(y .+ 1)))
