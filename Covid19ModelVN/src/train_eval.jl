"""
    Predictor{
        P<:SciMLBase.DEProblem,
        SO<:SciMLBase.DEAlgorithm,
        SE<:SciMLBase.AbstractSensitivityAlgorithm,
    }

A struct that solves the underlying DiffEq problem and returns the solution when it is called

# Fields

* `problem`: the problem that will be solved
* `solver`: the numerical solver that will be used to calculate the DiffEq solution
* `sensealg`: sensitivity algorithm for getting the local gradient
* `abstol`: solver's absolute tolerant
* `reltol`: solver's relative tolerant
+ `save_idxs`: the indices of the system's states to return

# Constructor

    Predictor(problem::SciMLBase.DEProblem, save_idxs::Vector{Int})

## Arguments

* `problem`: the problem that will be solved
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

    function Predictor(problem::SciMLBase.DEProblem, save_idxs::Vector{Int})
        solver = Tsit5()
        sensealg = InterpolatingAdjoint(autojacvec = ReverseDiffVJP(true))
        return new{typeof(problem),typeof(solver),typeof(sensealg)}(
            problem,
            solver,
            sensealg,
            1e-6,
            1e-6,
            save_idxs,
        )
    end
end


"""
    (p::Predictor)(params, tspan, saveat)

Call an object of struct `CovidModelPredict` to solve the underlying DiffEq problem

# Arguments

* `params`: the set of parameters of the system
* `tspan`: the time span of the problem
* `saveat`: the collocation coordinates
"""
function (p::Predictor)(params, tspan, saveat)
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
    Loss{Metric,Predict,DataIter,R<:Real}

A callable struct that uses `metric` to calculate the loss between the output of
`predict` and `dataset`.

# Fields

* `metric`: a function that computes the error between two data arrays
* `predict`: the time span that the ODE solver will be run on
* `datacycle`: the cyling iterator that go through each batch in the dataset
* `tspan`: the integration time span

# Constructor

    Loss(
        metric,
        predict,
        dataset::TimeseriesDataset,
        batchsize = length(dataset.tsteps),
    )

## Arguments

* `metric`: a function that computes the error between two data arrays
* `predict`: the time span that the ODE solver will be run on
* `dataset`: the dataset that contains the ground truth data
* `batchsize`: the size of each batch in the dataset, default to no batching

# Callable

    (l::Loss{Metric,Predict,DataCycle,R})(
        params,
    ) where {Metric<:Function,Predict<:Predictor,DataCycle<:Iterators.Stateful,R<:Real}

Call an object of the `Loss` struct on a set of parameters to get the loss scalar.
Here, the field `metric` is used with 2 parameters: the prediction and the ground
truth data.

## Arguments

* `params`: the set of parameters of the model
"""
struct Loss{Metric,Predict,DataCycle,R<:Real}
    metric::Metric
    predict::Predict
    datacycle::DataCycle
    tspan::Tuple{R,R}

    function Loss(
        metric,
        predict,
        dataset::TimeseriesDataset,
        batchsize = length(dataset.tsteps),
    )
        dataloader = timeseries_dataloader(dataset, batchsize)
        datacycle = dataloader |> Iterators.cycle |> Iterators.Stateful
        return new{typeof(metric),typeof(predict),typeof(datacycle),eltype(dataset.tspan)}(
            metric,
            predict,
            datacycle,
            dataset.tspan,
        )
    end
end

function (l::Loss{Metric,Predict,DataCycle,R})(
    params,
) where {Metric<:Function,Predict<:Predictor,DataCycle<:Iterators.Stateful,R<:Real}
    data, tsteps = popfirst!(l.datacycle)
    sol = l.predict(params, l.tspan, tsteps)
    if sol.retcode != :Success
        # Unstable trajectories => hard penalize
        return Inf
    end
    pred = @view sol[:, :]
    if size(pred) != size(data)
        # Unstable trajectories / Wrong inputs
        return Inf
    end
    return l.metric(pred, data)
end

"""
    TrainCallbackState{R<:Real}

State of the callback struct

# Fields

* `iters`: number have iterations that have been run
* `progress`: the progress meter that keeps track of the process
* `eval_losses`: collected evaluation losses at each interval
* `test_losses`: collected testing losses at each interval
* `minimizer`: current best set of parameters
* `minimizer_loss`: loss value of the current best set of parameters

# Constructor

    TrainCallbackState(
        T::Type{R},
        params_length::Integer,
        show_progress::Bool,
    ) where {R<:Real} = new{T}(

## Arguments

+ `T`: type of the losses and parameters
+ `show_progress`: control whether to show a running progress bar

# Constructor

    TrainCallbackState(
        T::Type{R},
        params_length::Integer,
        progress::ProgressUnknown,
    ) where {R<:Real} = TrainCallbackState{T}(

## Arguments

+ `T`: type of the losses and parameters
+ `progress`: the progress meter object that will be used by the callback function
"""
mutable struct TrainCallbackState{R<:Real}
    iters::Int
    progress::ProgressUnknown
    params_log::Vector{Vector{R}}
    eval_losses::Vector{R}
    test_losses::Vector{R}
    minimizer::Vector{R}
    minimizer_loss::R

    TrainCallbackState(
        T::Type{R},
        params_length::Integer,
        show_progress::Bool,
    ) where {R<:Real} = TrainCallbackState(
        T,
        params_length,
        ProgressUnknown(showspeed = true, enabled = show_progress),
    )

    TrainCallbackState(
        T::Type{R},
        params_length::Integer,
        progress::ProgressUnknown,
    ) where {R<:Real} = new{T}(
        0,
        progress,
        Vector{R}[],
        T[],
        T[],
        Vector{T}(undef, params_length),
        typemax(T),
    )
end

"""
    TrainCallbackConfig{L<:Loss}

Configuration of the callback struct

# Fields

* `eval_loss`: loss function on the train dataset
* `test_loss`: loss function on the test dataset
* `save_interval`: interval for saving the current best set of parameters and losses
* `losses_save_fpath`: file path to the saved losses figure
* `params_save_fpath`: file path to the serialized current best set of parameters
"""
struct TrainCallbackConfig{L1<:Loss,L2<:Loss}
    eval_loss::L1
    test_loss::L2
    save_interval::Int
    losses_save_fpath::String
    params_save_fpath::String
    minimizer_save_fpath::String
end

"""
    TrainCallback{R<:Real,L<:Loss}

A callable struct that is used for handling callback for `sciml_train`. The callback will
keep track of the losses, the minimizer, and show a progress that keeps track of the
training process

# Fields

* `state`: current state of the object
* `config`: callback configuration

# Callable

    (cb::TrainCallback)(params::AbstractVector{R}, train_loss::R) where {R<:Real}

# Arguments

* `params`: the model's parameters
* `train_loss`: loss from the training step
"""
struct TrainCallback{R<:Real,L<:Loss}
    state::TrainCallbackState{R}
    config::TrainCallbackConfig{L}
end

function (cb::TrainCallback)(params::AbstractVector{R}, train_loss::R) where {R<:Real}
    eval_loss = cb.config.eval_loss(params)
    test_loss = cb.config.test_loss(params)
    showvalues = @SVector [
        :losses_save_fpath => cb.config.losses_save_fpath,
        :params_save_fpath => cb.config.params_save_fpath,
        :minimizer_save_fpath => cb.config.minimizer_save_fpath,
        :train_loss => train_loss,
        :eval_loss => eval_loss,
        :test_loss => test_loss,
    ]
    next!(cb.state.progress, showvalues = showvalues)
    push!(cb.state.params_log, params)
    push!(cb.state.eval_losses, eval_loss)
    push!(cb.state.test_losses, test_loss)
    cb.state.iters += 1
    if eval_loss < cb.state.minimizer_loss && size(params) == size(cb.state.minimizer)
        cb.state.minimizer_loss = eval_loss
        cb.state.minimizer .= params
    end
    if cb.state.iters % cb.config.save_interval == 0
        Serialization.serialize(
            cb.config.losses_save_fpath,
            (cb.state.eval_losses, cb.state.test_losses),
        )
        Serialization.serialize(cb.config.params_save_fpath, cb.state.params_log)
        Serialization.serialize(cb.config.minimizer_save_fpath, cb.state.minimizer)
    end
    return false
end

"""
    EvalConfig

A struct for holding general configuration for the evaluation process

# Arguments

+ `metrics`: a list of metric function that will be used to compute the model errors
+ `forecast_ranges`: a list of different time ranges on which the model's prediction will be evaluated
+ `labels`: names of the evaluated model's states
"""
struct EvalConfig
    metrics::Vector{Function}
    forecast_ranges::Vector{Int}
    labels::Vector{String}
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
    horizons = repeat(config.forecast_ranges, inner = length(config.metrics))
    metrics = repeat(map(string, config.metrics), length(config.forecast_ranges))
    errors = reshape(
        [
            metric(pred[idx, 1:days], test_dataset.data[idx, 1:days]) for
            metric ∈ config.metrics, days ∈ config.forecast_ranges,
            idx ∈ 1:length(config.labels)
        ],
        length(config.metrics) * length(config.forecast_ranges),
        length(config.labels),
    )
    df1 = DataFrame([horizons metrics], [:horizon, :metric])
    df2 = DataFrame(errors, config.labels)
    return [df1 df2]
end

"""
    MakieShowoffPlain

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
    plot_losses(
        train_losses::AbstractVector{R},
        test_losses::AbstractVector{R},
    ) where {R<:Real}

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
    fit,
    pred,
    train_dataset::TimeseriesDataset,
    test_dataset::TimeseriesDataset,
)
    fig = Figure(
        resolution = (400 * length(config.forecast_ranges), 400 * length(config.labels)),
    )
    for (i, label) ∈ enumerate(config.labels), (j, days) ∈ enumerate(config.forecast_ranges)
        truth = @views [train_dataset.data[i, :]; test_dataset.data[i, 1:days]]
        output = [fit[i, :]; pred[i, 1:days]]
        plot_forecast!(fig[i, j], output, truth, days, train_dataset.tspan[2], label)
    end
    return fig
end

function plot_forecasts(
    config::EvalConfig,
    fit::Observable,
    pred::Observable,
    train_dataset::TimeseriesDataset,
    test_dataset::TimeseriesDataset,
)
    fig = Figure(
        resolution = (400 * length(config.forecast_ranges), 400 * length(config.labels)),
    )
    for (i, label) ∈ enumerate(config.labels), (j, days) ∈ enumerate(config.forecast_ranges)
        truth = @views [train_dataset.data[i, :]; test_dataset.data[i, 1:days]]
        output = lift(fit, pred) do x, y
            @views [x[i, :]; y[i, 1:days]]
        end
        plot_forecast!(fig[i, j], output, truth, days, train_dataset.tspan[2], label)
    end
    return fig
end

function plot_forecast!(
    gridpos::GridPosition,
    output,
    truth,
    days::Real,
    sep::Real,
    label::AbstractString,
)
    ax = Axis(
        gridpos,
        title = "$days-day forecast",
        xlabel = "Days since the 500th confirmed cases",
        ylabel = "Cases",
    )
    vlines!(ax, [sep], color = :black, linestyle = :dash)
    barplot!(ax, truth, label = label, linewidth = 4, color = Makie.ColorSchemes.tab10[1])
    lines!(
        ax,
        output,
        label = "model's prediction",
        linewidth = 4,
        color = Makie.ColorSchemes.tab10[2],
    )
    axislegend(ax, position = :lt)
    return ax
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
