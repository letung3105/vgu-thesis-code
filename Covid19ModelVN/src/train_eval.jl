export TrainConfig,
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
    plot_ℜe,
    logit,
    hswish,
    boxconst,
    boxconst_inv,
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
+ `save_idxs`: the indices of the system's states to return
"""
struct Predictor{
    P<:SciMLBase.DEProblem,
    SO<:SciMLBase.DEAlgorithm,
    SE<:SciMLBase.AbstractSensitivityAlgorithm,
}
    problem::P
    solver::SO
    sensealg::SE
    abstol::Float64
    reltol::Float64
    save_idxs::Vector{Int}
end

"""
Construct a new default `Predictor` using the problem defined by the given model

# Argument

* `problem`: the problem that will be solved
+ `save_idxs`: the indices of the system's states to return
"""
Predictor(problem::SciMLBase.DEProblem, save_idxs::Vector{Int}) = Predictor(
    problem,
    Tsit5(),
    InterpolatingAdjoint(autojacvec = ReverseDiffVJP(true)),
    1e-6,
    1e-6,
    save_idxs,
)

"""
Call an object of struct `CovidModelPredict` to solve the underlying DiffEq problem

# Arguments

* `params`: the set of parameters of the system
* `tspan`: the time span of the problem
* `saveat`: the collocation coordinates
"""
function (p::Predictor)(
    params::VT,
    tspan::Tuple{T,T},
    saveat::TS,
) where {T<:Real,VT<:AbstractVector{T},TS}
    problem = remake(p.problem, p = params, tspan = tspan)
    return solve(
        problem,
        p.solver,
        saveat = saveat,
        sensealg = p.sensealg,
        abstol = p.abstol,
        reltol = p.reltol,
        save_idxs = p.save_idxs,
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
struct Loss{F<:Function,P<:Predictor,D<:TimeseriesDataset}
    metric_fn::F
    predict_fn::P
    dataset::D
end

"""
Call an object of the `Loss` struct on a set of parameters to get the loss scalar

# Arguments

* `params`: the set of parameters of the model
"""
function (l::Loss)(params::VT) where {VT<:AbstractVector{<:Real}}
    sol = l.predict_fn(params, l.dataset.tspan, l.dataset.tsteps)
    if sol.retcode != :Success
        # Unstable trajectories => hard penalize
        return Inf
    end
    pred = Array(sol)
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
* `train_losses`: collected testing losses at each interval
* `minimizer`: current best set of parameters
* `minimizer_loss`: loss value of the current best set of parameters
"""
mutable struct TrainCallbackState{R<:Real}
    iters::Int
    progress::ProgressUnknown
    train_losses::Vector{R}
    test_losses::Vector{R}
    minimizer::Vector{R}
    minimizer_loss::R
end

"""
Construct a new `TrainCallbackState` with the progress bar set to `maxiters`
and other fields set to their default values

# Arguments

+ `T`: type of the losses and parameters
"""
TrainCallbackState(T::Type{R}; show_progress::Bool) where {R<:Real} = TrainCallbackState{T}(
    0,
    ProgressUnknown(showspeed = true, enabled = show_progress),
    T[],
    T[],
    T[],
    typemax(T),
)

"""
Configuration of the callback struct

# Fields

* `test_loss`: loss function on the test dataset
* `losses_plot_fpath`: file path to the saved losses figure
* `losses_plot_interval`: interval for collecting losses and plot the losses figure
* `params_length`: number of parameters that the system has
* `params_save_fpath`: file path to the serialized current best set of parameters
* `params_save_interval`: interval for saving the current best set of parameters
"""
struct TrainCallbackConfig{L<:Loss}
    show_progress::Bool
    test_loss::L
    losses_plot_fpath::String
    losses_plot_interval::Int
    params_length::Int
    params_save_fpath::String
    params_save_interval::Int
end

"""
A callable struct that is used for handling callback for `sciml_train`
"""
mutable struct TrainCallback{R<:Real,L<:Loss}
    state::TrainCallbackState{R}
    config::TrainCallbackConfig{L}
end

"""
Create a callback for `sciml_train`

# Arguments

+ `T`: type of the losses and parameters
* `config`: callback configurations
"""
TrainCallback(T::Type{R}, config::TrainCallbackConfig{L}) where {R<:Real,L<:Loss} =
    TrainCallback{T,L}(TrainCallbackState(T, show_progress = config.show_progres), config)

"""
Illustrate the training andd testing losses using a twinaxis plot

# Arguments

*`train_losses`: the training losses to be plotted
*`test_losses`: the testing losses to be plotted
"""
function plot_losses(
    train_losses::AbstractVector{R},
    test_losses::AbstractVector{R},
) where {R<:Real}
    fig = Figure()
    # train losses axis
    ax1 = Axis(
        fig[1, 1],
        title = "Losses of the model after each iteration",
        xlabel = "Iterations",
        yticklabelcolor = Makie.ColorSchemes.tab10[1],
    )
    # test losses axis
    ax2 = Axis(
        fig[1, 1],
        yaxisposition = :right,
        yticklabelcolor = Makie.ColorSchemes.tab10[2],
    )
    hidespines!(ax2)
    hidexdecorations!(ax2)
    sca1 = scatter!(ax1, train_losses, color = Makie.ColorSchemes.tab10[1])
    sca2 = scatter!(ax2, test_losses, color = Makie.ColorSchemes.tab10[2])
    Legend(
        fig[1, 1],
        [sca1, sca2],
        ["Train loss", "Test loss"],
        margin = (10, 10, 10, 10),
        tellheight = false,
        tellwidth = false,
        halign = :left,
        valign = :top,
    )
    return fig
end

"""
Call an object of type `TrainCallback`

# Arguments

* `params`: the model's parameters
* `train_loss`: loss from the training step
"""
function (cb::TrainCallback)(params::AbstractVector{R}, train_loss::R) where {R<:Real}
    test_loss = cb.config.test_loss(params)
    showvalues = Pair{Symbol,Any}[
        :losses_plot_fpath=>cb.config.losses_plot_fpath,
        :params_save_fpath=>cb.config.params_save_fpath,
        :train_loss=>train_loss,
        :test_loss=>test_loss,
    ]
    next!(cb.state.progress, showvalues = showvalues)
    cb.state.iters += 1
    if train_loss < cb.state.minimizer_loss && length(params) == cb.config.params_length
        cb.state.minimizer_loss = train_loss
        cb.state.minimizer = params
    end
    if cb.state.iters % cb.config.losses_plot_interval == 0
        push!(cb.state.train_losses, train_loss)
        push!(cb.state.test_losses, test_loss)
        fig = plot_losses(cb.state.train_losses, cb.state.test_losses)
        save(cb.config.losses_plot_fpath, fig)
    end
    if cb.state.iters % cb.config.params_save_interval == 0
        Serialization.serialize(cb.config.params_save_fpath, cb.state.minimizer)
    end
    return false
end

"""
Specifications for a model tranining

# Arguments

+ `name`: name of the configurationj
+ `optimizer`: The optimizer that will run in the session
+ `maxiters`: Maximum number of iterations to run the optimizer
"""
struct TrainConfig{Opt}
    name::String
    optimizer::Opt
    maxiters::Int
end

"""
A struct for holding general configuration for the evaluation process

# Arguments

+ `metric_fns`: a list of metric function that will be used to compute the model errors
+ `forecast_ranges`: a list of different time ranges on which the model's prediction will be evaluated
+ `labels`: names of the evaluated model's states
"""
struct EvalConfig
    metric_fns::Vector{Function}
    forecast_ranges::Vector{Int}
    labels::Vector{String}
end

"""
Find a set of paramters that minimizes the loss function defined by `train_loss_fn`, starting from
the initial set of parameters `params`.

# Arguments

+ `uuid`: unique id for the training session
+ `train_loss`: a function that will be minimized
+ `test_loss`: a loss function used for evaluation
+ `p0`: the initial set of parameters
+ `configs`: a collection of optimizers and settings used for training the model
+ `loss_samples`: number of params and losses samples to take
+ `snapshots_dir`: a directory for saving the model parameters and training losses
+ `kwargs`: keyword arguments that get splatted to `sciml_train`
"""
function train_model(
    uuid::AbstractString,
    train_loss::Loss,
    test_loss::Loss,
    p0::AbstractVector{<:Real},
    configs::AbstractVector{TrainConfig},
    snapshots_dir::AbstractString;
    show_progress::Bool = true,
    loss_samples::Integer = 100,
    kwargs...,
)
    if !isdir(snapshots_dir)
        mkpath(snapshots_dir)
    end

    minimizers = Vector{typeof(p0)}()
    params = copy(p0)

    @info "Running $uuid"
    for conf ∈ configs
        sessname = "$uuid.$(conf.name)"
        losses_plot_fpath = get_losses_plot_fpath(snapshots_dir, sessname)
        params_save_fpath = get_params_save_fpath(snapshots_dir, sessname)
        save_interval = div(conf.maxiters, loss_samples)
        cb = TrainCallback(
            eltype(p0),
            TrainCallbackConfig(
                show_progress,
                test_loss,
                losses_plot_fpath,
                save_interval,
                length(p0),
                params_save_fpath,
                save_interval,
            ),
        )
        params .= try
            res = DiffEqFlux.sciml_train(
                train_loss,
                params,
                conf.optimizer;
                cb = cb,
                maxiters = conf.maxiters,
                kwargs...,
            )
            res.minimizer
        catch e
            e isa InterruptException && rethrow(e)
            @warn e
            cb.state.minimizer
        end
        push!(minimizers, params)
        Serialization.serialize(params_save_fpath, params)
    end
    return minimizers
end

"""
Evaluate the model by calculating the errors and draw plot againts ground truth data

# Arguments

+ `config`: the configuration for the evalution process
+ `predictor`: the function that produce the model's prediction
+ `params`: the parameters used for making the predictions
+ `train_dataset`: ground truth data on which the model was trained
+ `test_dataset`: ground truth data that the model has not seen
"""
function evaluate_model(
    config::EvalConfig,
    predictor::Predictor,
    params::AbstractVector{<:Real},
    train_dataset::TimeseriesDataset,
    test_dataset::TimeseriesDataset,
)
    fit = predictor(params, train_dataset.tspan, train_dataset.tsteps)
    pred = predictor(params, test_dataset.tspan, test_dataset.tsteps)
    forecasts_plot = plot_forecasts(config, fit, pred, train_dataset, test_dataset)
    df_forecasts_errors = calculate_forecasts_errors(config, pred, test_dataset)
    return forecasts_plot, df_forecasts_errors
end

"""
Calculate the forecast error based on the model prediction and the ground truth data
for each forecasting horizon

# Arguments

* `config`: configuration for evaluation
* `pred`: prediction made by the model
* `test_dataset`: ground truth data for the forecasted period
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
            metric_fn(pred[idx, 1:days], test_dataset.data[idx, 1:days]) for
            metric_fn ∈ config.metric_fns, days ∈ config.forecast_ranges,
            idx ∈ 1:length(config.labels)
        ],
        length(config.metric_fns) * length(config.forecast_ranges),
        length(config.labels),
    )
    df1 = DataFrame([horizons metrics], [:horizon, :metric])
    df2 = DataFrame(errors, config.labels)
    return [df1 df2]
end

"""
Plot the forecasted values produced by against the ground truth data.

# Arguments

* `config`: configuration for evaluation
* `fit`: the solution returned by the model on the fit data
* `pred`: prediction made by the model
* `train_dataset`: the data that was used to train the model
* `test_dataset`: ground truth data for the forecasted period
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
    for (i, label) ∈ enumerate(config.labels), (j, days) ∈ enumerate(config.forecast_ranges)
        ax = Axis(
            fig[i, j],
            title = "$days-day forecast",
            xlabel = "Days since the 500th confirmed cases",
            ylabel = "Cases",
        )
        vlines!(ax, [train_dataset.tspan[2]], color = :black, linestyle = :dash)
        scatter!(
            ax,
            [train_dataset.data[i, :]; test_dataset.data[i, 1:days]],
            label = label,
        )
        scatter!(ax, [fit[i, :]; pred[i, 1:days]], label = "model's prediction")
        axislegend(ax, position = :lt)
    end
    return fig
end

"""
Plot the effective reproduction number for the traing period and testing period

# Arguments

* `ℜe`: the effective reproduction number
* `sep`: value at which the data is splitted for training and testing
"""
function plot_ℜe(ℜe::AbstractVector{R}, sep::R) where {R<:Real}
    R_effective_plot = Figure()
    ax = Axis(
        R_effective_plot[1, 1],
        title = "Effective reproduction number learned by the model",
        xlabel = "Days since the 500th confirmed case",
    )
    vlines!(ax, [sep], color = :black, linestyle = :dash, label = "last training day")
    scatter!(ax, ℜe, color = :red, linewidth = 2, label = "effective reproduction number")
    axislegend(ax, position = :lt)
    return R_effective_plot
end

"""
Calculate the inverse of the sigmoid function
"""
logit(x::Real) = log(x / (1 - x))

"""
Transform the value of `x` to get a value that lies between `bounds[1]` and `bounds[2]`.
"""
boxconst(x::Real, bounds::Tuple{R,R}) where {R<:Real} =
    bounds[1] + (bounds[2] - bounds[1]) * sigmoid(x)

boxconst_inv(x::Real, bounds::Tuple{R,R}) where {R<:Real} =
    logit((x - bounds[1]) / (bounds[2] - bounds[1]))

"""
[1] A. Howard et al., “Searching for MobileNetV3,” arXiv:1905.02244 [cs], Nov. 2019, Accessed: Oct. 09, 2021. [Online]. Available: http://arxiv.org/abs/1905.02244
"""
hswish(x::Real) = x * (relu6(x + 3) / 6)

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
