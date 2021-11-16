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
    """
    Construct a new default `Predictor` using the problem defined by the given model

    # Argument

    * `problem`: the problem that will be solved
    + `save_idxs`: the indices of the system's states to return
    """
    Predictor(problem::SciMLBase.DEProblem, save_idxs::Vector{Int}) =
        new{typeof(problem),Tsit5,InterpolatingAdjoint}(
            problem,
            Tsit5(),
            InterpolatingAdjoint(autojacvec = ReverseDiffVJP(true)),
            1e-6,
            1e-6,
            save_idxs,
        )
end


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
struct Loss{Bool,F<:Function,P<:Predictor,D<:TimeseriesDataset}
    metric_fn::F
    predict_fn::P
    dataset::D

    """
        Loss{true}(metric_fn, predict_fn, dataset)

    Construct a new loss function againts the ground truth data where `metric_fn` accepts
    the model parameters as the third parameter (eg. for regularization)

    # Arguments

    * `metric_fn`: a function that computes the error between two data arrays
    * `predict_fn`: the time span that the ODE solver will be run on
    * `dataset`: the dataset that contains the ground truth data
    """
    Loss{true}(metric_fn, predict_fn, dataset) =
        new{true,typeof(metric_fn),typeof(predict_fn),typeof(dataset)}(
            metric_fn,
            predict_fn,
            dataset,
        )

    """
        Loss{false}(metric_fn, predict_fn, dataset)

    Construct a new loss function againts the ground truth data.
    # Arguments

    * `metric_fn`: a function that computes the error between two data arrays
    * `predict_fn`: the time span that the ODE solver will be run on
    * `dataset`: the dataset that contains the ground truth data
    """
    Loss{false}(metric_fn, predict_fn, dataset) =
        new{false,typeof(metric_fn),typeof(predict_fn),typeof(dataset)}(
            metric_fn,
            predict_fn,
            dataset,
        )
end

"""
    (l::Loss{false,F,P,D})(params::VT) where {F,P,D,VT<:AbstractVector{<:Real}}

Call an object of the `Loss` struct on a set of parameters to get the loss scalar

# Arguments

* `params`: the set of parameters of the model
"""
function (l::Loss{false,F,P,D})(params::VT) where {F,P,D,VT<:AbstractVector{<:Real}}
    sol = l.predict_fn(params, l.dataset.tspan, l.dataset.tsteps)
    if sol.retcode != :Success
        # Unstable trajectories => hard penalize
        return Inf
    end
    pred = @view sol[:, :]
    if size(pred) != size(l.dataset.data)
        # Unstable trajectories / Wrong inputs
        return Inf
    end
    return l.metric_fn(pred, l.dataset.data)
end

"""
    (l::Loss{true,F,P,D})(params::VT) where {F,P,D,VT<:AbstractVector{<:Real}}

Call an object of the `Loss` struct on a set of parameters to get the loss scalar.
The metric function accepts a keyword argument `params` for regularization

# Arguments

* `params`: the set of parameters of the model
"""
function (l::Loss{true,F,P,D})(params::VT) where {F,P,D,VT<:AbstractVector{<:Real}}
    sol = l.predict_fn(params, l.dataset.tspan, l.dataset.tsteps)
    if sol.retcode != :Success
        # Unstable trajectories => hard penalize
        return Inf
    end
    pred = @view sol[:, :]
    if size(pred) != size(l.dataset.data)
        # Unstable trajectories / Wrong inputs
        return Inf
    end
    return l.metric_fn(pred, l.dataset.data, params)
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
mutable struct TrainCallbackState{R<:Real}
    iters::Int
    progress::ProgressUnknown
    train_losses::Vector{R}
    test_losses::Vector{R}
    minimizer::Vector{R}
    minimizer_loss::R
    """
    # Arguments

    + `T`: type of the losses and parameters
    + `show_progress`: control whether to show a running progress bar
    """
    TrainCallbackState(
        T::Type{R},
        params_length::Integer,
        show_progress::Bool,
    ) where {R<:Real} = new{T}(
        0,
        ProgressUnknown(showspeed = true, enabled = show_progress),
        T[],
        T[],
        Vector{T}(undef, params_length),
        typemax(T),
    )
end


"""
# Arguments

+ `T`: type of the losses and parameters
+ `progress`: the progress meter object that will be used by the callback function
"""
TrainCallbackState(
    T::Type{R},
    params_length::Integer,
    progress::ProgressUnknown,
) where {R<:Real} = TrainCallbackState{T}(
    0,
    progress,
    T[],
    T[],
    Vector{T}(undef, params_length),
    typemax(T),
)

"""
Configuration of the callback struct

# Fields

* `test_loss`: loss function on the test dataset
* `save_interval`: interval for saving the current best set of parameters and losses
* `losses_save_fpath`: file path to the saved losses figure
* `params_save_fpath`: file path to the serialized current best set of parameters
"""
struct TrainCallbackConfig{L<:Loss}
    test_loss::L
    save_interval::Int
    losses_save_fpath::String
    params_save_fpath::String
end

"""
A callable struct that is used for handling callback for `sciml_train`
"""
struct TrainCallback{R<:Real,L<:Loss}
    state::TrainCallbackState{R}
    config::TrainCallbackConfig{L}
end

"""
Call an object of type `TrainCallback`

# Arguments

* `params`: the model's parameters
* `train_loss`: loss from the training step
"""
function (cb::TrainCallback)(params::AbstractVector{R}, train_loss::R) where {R<:Real}
    test_loss = cb.config.test_loss(params)
    showvalues = @SVector [
        :losses_save_fpath => cb.config.losses_save_fpath,
        :params_save_fpath => cb.config.params_save_fpath,
        :train_loss => train_loss,
        :test_loss => test_loss,
    ]
    next!(cb.state.progress, showvalues = showvalues)
    push!(cb.state.train_losses, train_loss)
    push!(cb.state.test_losses, test_loss)
    cb.state.iters += 1
    if train_loss < cb.state.minimizer_loss && size(params) == size(cb.state.minimizer)
        cb.state.minimizer_loss = train_loss
        cb.state.minimizer .= params
    end
    if cb.state.iters % cb.config.save_interval == 0
        Serialization.serialize(
            cb.config.losses_save_fpath,
            (cb.state.train_losses, cb.state.test_losses),
        )
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
+ `show_progress`: control whether to show the default progress bar for each training session,
this option will be ignored when `progress` is set
+ `progress`: a progress meter that will be used to monitor the training sessions
+ `kwargs`: keyword arguments that get splatted to `sciml_train`
"""
function train_model(
    uuid::AbstractString,
    train_loss::Loss,
    test_loss::Loss,
    p0::AbstractVector{<:Real},
    configs::AbstractVector{TrainConfig},
    snapshots_dir::AbstractString;
    show_progress::Bool = false,
    progress::Union{ProgressUnknown,Nothing} = nothing,
    loss_samples::Integer = 100,
    kwargs...,
)
    if !isdir(snapshots_dir)
        mkpath(snapshots_dir)
    end

    minimizers = Vector{typeof(p0)}()
    params = copy(p0)
    params_save_fpath = get_params_save_fpath(snapshots_dir, uuid)

    for conf ∈ configs
        @info "Training $uuid\nMaxiters $(conf.maxiters)\nOptimizer $(typeof(conf.optimizer).name.wrapper)"
        cb = TrainCallback(
            TrainCallbackState(
                eltype(p0),
                length(p0),
                isnothing(progress) ? show_progress : progress,
            ),
            TrainCallbackConfig(
                test_loss,
                div(conf.maxiters, loss_samples),
                get_losses_save_fpath(snapshots_dir, "$uuid.$(conf.name)"),
                params_save_fpath,
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
    evaluate_model(
        config::EvalConfig,
        predictor::Predictor,
        params::AbstractVector{<:Real},
        train_dataset::TimeseriesDataset,
        test_dataset::TimeseriesDataset,
    )::(Makie.Figure, DataFrames.DataFrame)

Evaluate the model by calculating the errors and draw plot againts ground truth data

# Returns

A 2-tuple where the first element contains the Figure object containing the model
forecasts and the second element contains the Dataframe fore the forecasts errors

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
    calculate_forecasts_errors(
        config::EvalConfig,
        pred::SciMLBase.AbstractTimeseriesSolution,
        test_dataset::TimeseriesDataset,
    )::DataFrame

Calculate the forecast error based on the model prediction and the ground truth data
for each forecasting horizon

# Returns

A dataframe containing the errors between the model prediction and the ground truth data
calculated with different metrics and forecasting horizons

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
Marker struct for our custom formatter that uses `Showoff.showoff` with the option set to
`:plain`. This is done to mitigate to error occur with `Unicode.subscript` when used on
scientific-/engineering-formated strings
"""
struct MakieShowoffPlain end

"""
    makie_log_scale_formatter(xs::AbstractVector)::Vector{String}

The format function that is used when the `MakieLogScaleFormatter` marker is set
"""
makie_showoff_plain(xs) = MakieLayout.Showoff.showoff(xs, :plain)

"""
    MakieLayout.get_ticks(l::LogTicks, scale::Union{typeof(log10), typeof(log2), typeof(log)}, ::MakieShowoffPlain, vmin, vmax)

Override Makie default function for getting ticks values and labels for log-scaled axis.
This method uses our custom formatter `MakieShowoffPlain` instead of using `Makie.Automatic`.
"""
function MakieLayout.get_ticks(
    l::LogTicks,
    scale::Union{typeof(log10),typeof(log2),typeof(log)},
    ::MakieShowoffPlain,
    vmin,
    vmax,
)
    ticks_scaled =
        MakieLayout.get_tickvalues(l.linear_ticks, identity, scale(vmin), scale(vmax))
    ticks = Makie.inverse_transform(scale).(ticks_scaled)

    labels_scaled = MakieLayout.get_ticklabels(makie_showoff_plain, ticks_scaled)
    labels = MakieLayout._logbase(scale) .* Makie.UnicodeFun.to_superscript.(labels_scaled)

    (ticks, labels)
end

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
    ax1 = Axis(
        fig[1, 1],
        title = "Losses of the model after each iteration",
        xlabel = "Iterations",
        yscale = log10,
        ytickformat = MakieShowoffPlain(),
        yticklabelcolor = Makie.ColorSchemes.tab10[1],
    )
    ax2 = Axis(
        fig[1, 1],
        yaxisposition = :right,
        yscale = log10,
        ytickformat = MakieShowoffPlain(),
        yticklabelcolor = Makie.ColorSchemes.tab10[2],
    )
    hidespines!(ax2)
    hidexdecorations!(ax2)
    ln1 = lines!(ax1, train_losses, color = Makie.ColorSchemes.tab10[1], linewidth = 3)
    ln2 = lines!(ax2, test_losses, color = Makie.ColorSchemes.tab10[2], linewidth = 3)
    Legend(
        fig[1, 1],
        [ln1, ln2],
        ["Train loss", "Test loss"],
        margin = (10, 10, 10, 10),
        tellheight = false,
        tellwidth = false,
        halign = :right,
        valign = :top,
    )
    return fig
end

function plot_losses(train_losses::AbstractVector{R}) where {R<:Real}
    fig = Figure()
    ax = Axis(
        fig[1, 1],
        title = "Losses of the model after each iteration",
        xlabel = "Iterations",
        yscale = log10,
    )
    ln = lines!(ax, train_losses, color = Makie.ColorSchemes.tab10[1], linewidth = 3)
    Legend(
        fig[1, 1],
        [ln],
        ["Train loss"],
        margin = (10, 10, 10, 10),
        tellheight = false,
        tellwidth = false,
        halign = :right,
        valign = :top,
    )
    return fig
end

"""
    plot_forecasts(
        config::EvalConfig,
        fit::SciMLBase.AbstractTimeseriesSolution,
        pred::SciMLBase.AbstractTimeseriesSolution,
        train_dataset::TimeseriesDataset,
        test_dataset::TimeseriesDataset,
    )

Plot the forecasted values produced by against the ground truth data.

# Returns

The figure object from Makie that contains the plotting definition for the model predictions

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
        lines!(
            ax,
            [train_dataset.data[i, :]; test_dataset.data[i, 1:days]],
            label = label,
            linewidth = 4,
        )
        lines!(
            ax,
            [fit[i, :]; pred[i, 1:days]],
            label = "model's prediction",
            linewidth = 4,
        )
        axislegend(ax, position = :lt)
    end
    return fig
end

"""
    plot_ℜe(ℜe::AbstractVector{R}, sep::R)::Figure where {R<:Real}

Plot the effective reproduction number for the traing period and testing period

# Returns

The figure object from Makie that contains the plotting definition for the given
effecitve reproduction number

# Arguments

* `ℜe`: the effective reproduction number
* `sep`: value at which the data is splitted for training and testing
"""
function plot_ℜe(ℜe::AbstractVector{R}, sep::R) where {R<:Real}
    fig = Figure()
    ax = Axis(fig[2, 1], xlabel = "Days since the 500th confirmed case")
    vln = vlines!(ax, [sep], color = :black, linestyle = :dash)
    ln = lines!(ax, ℜe, color = :red, linewidth = 3)
    Legend(
        fig[1, 1],
        [vln, ln],
        ["last training day", "effective reproduction number"],
        orientation = :horizontal,
        tellwidth = false,
        tellheight = true,
    )
    return fig
end

"""
    logit(x::Real)::Real

Calculate the inverse of the sigmoid function
"""
logit(x::Real) = log(x / (1 - x))

"""
    boxconst(x::Real, bounds::Tuple{R,R})::Real where {R<:Real}

Transform the value of `x` to get a value that lies between `bounds[1]` and `bounds[2]`
"""
boxconst(x::Real, bounds::Tuple{R,R}) where {R<:Real} =
    bounds[1] + (bounds[2] - bounds[1]) * sigmoid(x)

"""
    boxconst(x::Real, bounds::Tuple{R,R})::Real where {R<:Real}

Calculate the inverse of the `boxconst` function
"""
boxconst_inv(x::Real, bounds::Tuple{R,R}) where {R<:Real} =
    logit((x - bounds[1]) / (bounds[2] - bounds[1]))

"""
    hswish(x::Real)::Real

[1] A. Howard et al., “Searching for MobileNetV3,” arXiv:1905.02244 [cs], Nov. 2019, Accessed: Oct. 09, 2021. [Online]. Available: http://arxiv.org/abs/1905.02244
"""
hswish(x::Real) = x * (relu6(x + 3) / 6)

"""
    mae(ŷ::AbstractArray{<:Real}, y::AbstractArray{<:Real})::Real

Calculate the mean absolute error between 2 values. Note that the input arguments must be of the same size.
The function does not check if the inputs are valid and may produces erroneous output.
"""
mae(ŷ::AbstractArray{<:Real}, y::AbstractArray{<:Real}) = mean(abs, (ŷ .- y))

"""
    mape(ŷ::AbstractArray{<:Real}, y::AbstractArray{<:Real})::Real

Calculate the mean absolute percentge error between 2 values. Note that the input arguments must be of the same size.
The function does not check if the inputs are valid and may produces erroneous output.
"""
mape(ŷ::AbstractArray{<:Real}, y::AbstractArray{<:Real}) = 100 * mean(abs, (ŷ .- y) ./ y)

"""
    rmse(ŷ::AbstractArray{<:Real}, y::AbstractArray{<:Real})::Real

Calculate the root mean squared error between 2 values. Note that the input arguments must be of the same size.
The function does not check if the inputs are valid and may produces erroneous output.
"""
rmse(ŷ::AbstractArray{<:Real}, y::AbstractArray{<:Real}) = sqrt(mean(abs2, ŷ .- y))

"""
    rmsle(ŷ::AbstractArray{<:Real}, y::AbstractArray{<:Real})::Real

Calculate the root mean squared log error between 2 values. Note that the input arguments must be of the same size.
The function does not check if the inputs are valid and may produces erroneous output.
"""
rmsle(ŷ::AbstractArray{<:Real}, y::AbstractArray{<:Real}) =
    sqrt(mean(abs2, log.(ŷ .+ 1) .- log.(y .+ 1)))
