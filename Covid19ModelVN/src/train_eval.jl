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
    plot_effective_reproduction_number,
    mae,
    mape,
    rmse,
    rmsle

using Serialization,
    Statistics,
    ProgressMeter,
    CairoMakie,
    DataFrames,
    CSV,
    OrdinaryDiffEq,
    DiffEqFlux,
    Covid19ModelVN

"""
A struct that solves the underlying DiffEq problem and returns the solution when it is called

# Fields

* `problem`: the problem that will be solved
* `solver`: the numerical solver that will be used to calculate the DiffEq solution
* `sensealg`: sensitivity algorithm for getting the local gradient
* `abstol`: solver's absolute tolerant
* `reltol`: solver's relative tolerant
"""
struct Predictor
    problem::SciMLBase.DEProblem
    solver::SciMLBase.DEAlgorithm
    sensealg::SciMLBase.AbstractSensitivityAlgorithm
    abstol::Real
    reltol::Real
end

"""
Construct a new default `Predictor` using the problem defined by the given model

# Argument

+ `model`: a model containing a problem that can be solved
"""
Predictor(model::AbstractCovidModel) = Predictor(
    model.problem,
    Tsit5(),
    InterpolatingAdjoint(autojacvec = ReverseDiffVJP(true)),
    1e-6,
    1e-6,
)

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
    return solve(
        problem,
        p.solver,
        saveat = saveat,
        sensealg = p.sensealg,
        abstol = p.abstol,
        reltol = p.reltol,
    )
end

"""
A callable struct that uses `metric_fn` to calculate the loss between the output of
`predict` and `dataset`.

# Fields

* `metric_fn`: a function that computes the error between two data arrays
* `predict_fn`: the time span that the ODE solver will be run on
* `dataset`: the dataset that contains the ground truth data
"""
struct Loss
    metric_fn::Function
    predict_fn::Predictor
    dataset::TimeseriesDataset
    vars::AbstractVector{<:Integer}
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
    progress::ProgressUnknown
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
TrainCallbackState() = TrainCallbackState(
    0,
    ProgressUnknown(showspeed = true),
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
TrainCallback(config::TrainCallbackConfig = TrainCallbackConfig()) =
    TrainCallback(TrainCallbackState(), config)

"""
Call an object of type `TrainCallback`

# Arguments

* `params`: the model's parameters
* `train_loss`: loss from the training step
"""
function (cb::TrainCallback)(params::AbstractVector{<:Real}, train_loss::Real)
    showvalues = Pair{Symbol,Any}[
        :losses_plot_fpath=>cb.config.losses_plot_fpath,
        :params_save_fpath=>cb.config.params_save_fpath,
        :train_loss=>train_loss,
    ]
    test_loss = if !isnothing(cb.config.test_loss_fn)
        loss = cb.config.test_loss_fn(params)
        push!(showvalues, :test_loss => loss)
        loss
    end

    if train_loss < cb.state.minimizer_loss
        cb.state.minimizer_loss = train_loss
        cb.state.minimizer = params
    end

    cb.state.iters += 1
    if cb.state.iters % cb.config.losses_plot_interval == 0 &&
       !isnothing(cb.config.losses_plot_fpath)
        fig = Figure()
        ax = Axis(
            fig[1, 1],
            title = "Losses of the model after each iteration",
            xlabel = "Iterations",
        )
        append!(cb.state.train_losses, train_loss)
        scatter!(ax, cb.state.train_losses, label = "Train loss")
        if !isnothing(test_loss)
            append!(cb.state.test_losses, test_loss)
            scatter!(ax, cb.state.test_losses, label = "Test loss")
        end
        axislegend(ax, position = :lt)
        save(cb.config.losses_plot_fpath, fig)
    end

    if cb.state.iters % cb.config.params_save_interval == 0 &&
       !isnothing(cb.config.params_save_fpath)
        Serialization.serialize(cb.config.params_save_fpath, cb.state.minimizer)
    end

    next!(cb.state.progress, showvalues = showvalues)
    return false
end

"""
Specifications for a model tranining session

# Arguments

+ `name`: Session name
+ `optimizer`: The optimizer that will run in the session
+ `maxiters`: Maximum number of iterations to run the optimizer
+ `loss_samples`: Number of times to collect the training losses and testing losses
"""
struct TrainSession{Opt}
    name::AbstractString
    optimizer::Opt
    maxiters::Integer
    loss_samples::Integer
end

"""
A struct for holding general configuration for the evaluation process

# Arguments

+ `metric_fns`: a list of metric function that will be used to compute the model errors
+ `forecast_ranges`: a list of different time ranges on which the model's prediction will be evaluated
+ `labels`: names of the evaluated model's states
"""
struct EvalConfig
    metric_fns::AbstractVector{Function}
    forecast_ranges::AbstractVector{<:Integer}
    vars::AbstractVector{<:Integer}
    labels::AbstractVector{<:AbstractString}
end

"""
Find a set of paramters that minimizes the loss function defined by `train_loss_fn`, starting from
the initial set of parameters `params`.

# Arguments

+ `train_loss`: a function that will be minimized
+ `params`: the initial set of parameters
+ `sessions`: a collection of optimizers and settings used for training the model
+ `snapshots_dir`: a directory for saving the model parameters and training losses
+ `test_loss`: a function for evaluating the model on out-of-sample data
"""
function train_model(
    train_loss::Loss,
    params::AbstractVector{<:Real},
    sessions::AbstractVector{TrainSession};
    test_loss::Union{Loss,Nothing} = nothing,
    snapshots_dir::Union{AbstractString,Nothing} = nothing,
    kwargs...,
)
    minimizers = Vector{Float64}[]
    params = copy(params)
    for sess ∈ sessions
        snapshot_and_plot_interval = div(sess.maxiters, sess.loss_samples)
        losses_plot_fpath, params_save_fpath =
            isnothing(snapshots_dir) ? (nothing, nothing) :
            (
                get_losses_plot_fpath(snapshots_dir, sess.name),
                get_params_save_fpath(snapshots_dir, sess.name),
            )
        cb = TrainCallback(
            TrainCallbackConfig(
                test_loss,
                losses_plot_fpath,
                snapshot_and_plot_interval,
                params_save_fpath,
                snapshot_and_plot_interval,
            ),
        )

        @info "Running $(sess.name)"
        DiffEqFlux.sciml_train(
            train_loss,
            params,
            sess.optimizer;
            cb = cb,
            maxiters = sess.maxiters,
            kwargs...,
        )

        push!(minimizers, cb.state.minimizer)
        params .= cb.state.minimizer
        Serialization.serialize(params_save_fpath, params)
    end
    return minimizers
end

"""
Evaluate the model by calculating the errors and draw plot againts ground truth data

# Arguments

+ `config`: the configuration for the evalution process
+ `params`: the parameters used for making the predictions
+ `predict_fn`: the function that produce the model's prediction
+ `train_dataset`: ground truth data on which the model was trained
+ `test_dataset`: ground truth data that the model has not seen
"""
function evaluate_model(
    config::EvalConfig,
    params::AbstractVector{<:Real},
    predict_fn::Predictor,
    train_dataset::TimeseriesDataset,
    test_dataset::TimeseriesDataset,
)
    fit = predict_fn(params, train_dataset.tspan, train_dataset.tsteps)
    pred = predict_fn(params, test_dataset.tspan, test_dataset.tsteps)
    forecasts_plot = plot_forecasts(config, fit, pred, train_dataset, test_dataset)
    df_forecasts_errors = calculate_forecasts_errors(config, pred, test_dataset)
    return forecasts_plot, df_forecasts_errors
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
    config::EvalConfig,
    pred::SciMLBase.AbstractTimeseriesSolution,
    test_dataset::TimeseriesDataset,
)
    horizons = repeat(config.forecast_ranges, inner = length(config.metric_fns))
    metrics = repeat(map(string, config.metric_fns), length(config.forecast_ranges))
    errors = reshape(
        [
            metric_fn(pred[var, 1:days], test_dataset.data[idx, 1:days]) for
            metric_fn ∈ config.metric_fns, days ∈ config.forecast_ranges,
            (idx, var) ∈ enumerate(config.vars)
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
    config::EvalConfig,
    fit::SciMLBase.AbstractTimeseriesSolution,
    pred::SciMLBase.AbstractTimeseriesSolution,
    train_dataset::TimeseriesDataset,
    test_dataset::TimeseriesDataset,
)
    fig = Figure(
        resolution = (400 * length(config.forecast_ranges), 400 * length(config.labels)),
    )
    for (i, (var, label)) ∈ enumerate(zip(config.vars, config.labels)), (j, days) ∈ enumerate(config.forecast_ranges)
        ax = Axis(
            fig[i, j],
            title = "$days-day forecast",
            xlabel = "Days since the 500th confirmed cases",
            ylabel = "Cases",
        )
        vlines!(ax, [train_dataset.tspan[2]], color = :black, linestyle = :dash)
        scatter!([train_dataset.data[i, :]; test_dataset.data[i, 1:days]], label = label)
        scatter!([fit[var, :]; pred[var, 1:days]], label = "model's prediction")
        axislegend(ax, position = :lt)
    end
    return fig
end

"""
Get the effective reproduction number of the model and produce a plot from the data

# Arguments
* `model`: the Covid-19 model
* `minimizer`: the parameters to be used as the model's input
* `train_dataset`: the timeseries dataset for the training period
* `test_dataset`: the timeseries dataset for the testing period
"""
function plot_effective_reproduction_number(
    model::AbstractCovidModel,
    minimizer::AbstractVector{<:Real},
    train_dataset::TimeseriesDataset,
    test_dataset::TimeseriesDataset,
)
    # get the effective reproduction number learned by the model
    Re1 = effective_reproduction_number(
        model,
        minimizer,
        train_dataset.tspan,
        train_dataset.tsteps,
    )
    Re2 = effective_reproduction_number(
        model,
        minimizer,
        test_dataset.tspan,
        test_dataset.tsteps,
    )
    fig = Figure()
    ax = Axis(
        fig[1, 1],
        title = "Effective reproduction number learned by the model",
        xlabel = "Days since the 500th confirmed case",
    )
    vlines!(
        ax,
        [train_dataset.tspan[2]],
        color = :black,
        linestyle = :dash,
        label = "last training day",
    )
    scatter!(
        vec([Re1 Re2]),
        color = :red,
        linewidth = 2,
        label = "effective reproduction number",
    )
    axislegend(ax, position = :lt)
    return fig
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
