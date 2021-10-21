module Helpers

export Predictor,
    Loss,
    TrainCallback,
    TrainCallbackConfig,
    TrainSession,
    mape,
    mse,
    sse,
    rmse,
    rmsle,
    train_model,
    plot_forecasts,
    lookup_saved_params

using Dates,
    Printf,
    Serialization,
    Statistics,
    Plots,
    DataFrames,
    ProgressMeter,
    OrdinaryDiffEq,
    DiffEqFlux,
    Covid19ModelVN.Datasets,
    Covid19ModelVN.Models

"""
A struct that solves the underlying DiffEq problem and returns the solution when it is called

# Fields

* `problem::ODEProblem`: the problem that will be solved
* `solver`: the numerical solver that will be used to calculate the DiffEq solution
"""
struct Predictor
    problem::ODEProblem
    solver::Any
end

Predictor(problem::ODEProblem) = Predictor(problem, Tsit5())

"""
Call an object of struct `CovidModelPredict` to solve the underlying DiffEq problem

# Arguments

* `params`: the set of parameters of the system
* `tspan`: the time span of the problem
* `saveat`: the collocation coordinates
"""
function (p::Predictor)(params, tspan, saveat)
    problem = remake(p.problem, p = params, tspan = tspan)
    return solve(problem, p.solver, saveat = saveat)
end

"""
A callable struct that uses `metric_fn` to calculate the loss between the output of
`predict` and `dataset`.

# Fields

* `metric_fn::Function`: a function that computes the error between two data arrays
* `predict_fn::Predictor`: the time span that the ODE solver will be run on
* `dataset::TimeseriesDataset`: the dataset that contains the ground truth data
* `vars::Union{Int, AbstractVector{Int}, OrdinalRange}`: indices of the states that will be used to calculate the loss
"""
struct Loss
    metric_fn::Function
    predict_fn::Predictor
    dataset::TimeseriesDataset
    vars::Union{Int,AbstractVector{Int},OrdinalRange}
end

"""
Call an object of the `Loss` struct on a set of parameters to get the loss scalar

# Arguments

* `params`: the set of parameters of the model
"""
function (l::Loss)(params)
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
Calculate the mean absolute error between 2 values. Note that the input arguments must be of the same size.
The function does not check if the inputs are valid and may produces erroneous output.
"""
mae(ŷ, y) = mean(abs, (ŷ .- y))

"""
Calculate the mean absolute percentge error between 2 values. Note that the input arguments must be of the same size.
The function does not check if the inputs are valid and may produces erroneous output.
"""
mape(ŷ, y) = 100 * mean(abs, (ŷ .- y) ./ y)

"""
Calculate the sum squared error between 2 values. Note that the input arguments must be of the same size.
The function does not check if the inputs are valid and may produces erroneous output.
"""
sse(ŷ, y) = sum(abs2, ŷ .- y)

"""
Calculate the mean squared error between 2 values. Note that the input arguments must be of the same size.
The function does not check if the inputs are valid and may produces erroneous output.
"""
mse(ŷ, y) = mean(abs2, ŷ .- y)

"""
Calculate the root mean squared error between 2 values. Note that the input arguments must be of the same size.
The function does not check if the inputs are valid and may produces erroneous output.
"""
rmse(ŷ, y) = sqrt(mean(abs2, ŷ .- y))

"""
Calculate the root mean squared log error between 2 values. Note that the input arguments must be of the same size.
The function does not check if the inputs are valid and may produces erroneous output.
"""
rmsle(ŷ, y) = sqrt(mean(abs2, log.(ŷ .+ 1) .- log.(y .+ 1)))


"""
State of the callback struct

# Fields

* `iters::Int`: number have iterations that have been run
* `progress::Progress`: the progress meter that keeps track of the process
* `train_losses::AbstractVector{<:Real}`: collected training losses at each interval
* `test_losses::AbstractVector{<:Real}`: collected testing losses at each interval
* `minimizer::AbstractVector{<:Real}`: current best set of parameters
* `minimizer_loss::Real`: loss value of the current best set of parameters
"""
mutable struct TrainCallbackState
    iters::Int
    progress::Progress
    train_losses::AbstractVector{<:Real}
    test_losses::AbstractVector{<:Real}
    minimizer::AbstractVector{<:Real}
    minimizer_loss::Real
end

TrainCallbackState(maxiters::Int) = TrainCallbackState(
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

* `test_loss_fn::Union{Nothing, Loss}`: a callable for calculating the testing loss value
* `losses_plot_fpath::Union{Nothing,AbstractString}`: file path to the saved losses figure
* `losses_plot_interval::Int`: interval for collecting losses and plot the losses figure
* `params_save_fpath::Union{Nothing,AbstractString}`: file path to the serialized current best set of parameters
* `params_save_interval::Int`: interval for saving the current best set of parameters
"""
struct TrainCallbackConfig
    test_loss_fn::Union{Nothing,Loss}
    losses_plot_fpath::Union{Nothing,AbstractString}
    losses_plot_interval::Int
    params_save_fpath::Union{Nothing,AbstractString}
    params_save_interval::Int
end

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

* `maxiters::Int`: max number of iterations the optimizer will run
* `config`: callback configurations
"""
TrainCallback(maxiters::Int, config::TrainCallbackConfig = TrainCallbackConfig()) =
    TrainCallback(TrainCallbackState(maxiters), config)

"""
Call an object of type `TrainCallback`

# Arguments

* `params::AbstractVector{<:Real}`: the model's parameters
* `train_loss::Real`: loss from the training step
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
Get default losses figure file path

# Arguments

* `fdir::AbstractString`: the root directory of the file
* `uuid::AbstractString`: the file unique identifier
"""
get_losses_plot_fpath(fdir::AbstractString, uuid::AbstractString) =
    joinpath(fdir, "$uuid.losses.png")

"""
Get default file path for saved parameters

# Arguments

* `fdir::AbstractString`: the root directory of the file
* `uuid::AbstractString`: the file unique identifier
"""
get_params_save_fpath(fdir::AbstractString, uuid::AbstractString) =
    joinpath(fdir, "$uuid.params.jls")

struct TrainSession
    name::AbstractString
    optimizer::Any
    maxiters::Int
    losses_plot_dir::AbstractString
    params_save_dir::AbstractString
end

function train_model(
    train_loss_fn::Loss,
    test_loss_fn::Loss,
    p0::AbstractVector{<:Real},
    sessions::AbstractVector{TrainSession},
)
    params = p0
    for sess in sessions
        losses_plot_fpath = get_losses_plot_fpath(sess.losses_plot_dir, sess.name)
        params_save_fpath = get_params_save_fpath(sess.params_save_dir, sess.name)
        cb = TrainCallback(
            sess.maxiters,
            TrainCallbackConfig(
                train_loss_fn,
                losses_plot_fpath,
                div(sess.maxiters, 100),
                params_save_fpath,
                div(sess.maxiters, 100),
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
Get the file paths and uuids of all the saved parameters of an experiment

# Arguments

* `snapshots_dir::AbstractString`: the directory that contains the saved parameters
* `exp_name::AbstractString`: the experiment name
"""
function lookup_saved_params(snapshots_dir::AbstractString, exp_name::AbstractString)
    exp_dir = joinpath(snapshots_dir, exp_name)
    params_files = filter(x -> endswith(x, ".jls"), readdir(exp_dir))
    fpaths = map(f -> joinpath(snapshots_dir, exp_name, f), params_files)
    uuids = map(f -> first(rsplit(f, ".", limit = 3)), params_files)
    return fpaths, uuids
end

"""
Plot the forecasted values produced by `predict_fn` against the ground truth data and calculated the error for each
forecasted value using `metric_fn`.

# Arguments

* `predict_fn::Predictor`: a function that takes the model parameters and computes the ODE solver output
* `metric_fn::Function`: a function that computes the error between two data arrays
* `train_dataset::UDEDataset`: the data that was used to train the model
* `test_dataset::UDEDataset`: ground truth data for the forecasted period
* `minimizer`: the model's paramerters that will be used to produce the forecast
* `forecast_ranges`: a range of day horizons that will be forecasted
* `vars`: the model's states that will be compared
* `cols`: the ground truth values that will be compared
* `labels`: plot labels of each of the ground truth values
"""
function plot_forecasts(
    predict_fn::Predictor,
    metric_fn::Function,
    train_dataset::TimeseriesDataset,
    test_dataset::TimeseriesDataset,
    minimizer::AbstractVector{<:Real},
    forecast_ranges::AbstractVector{Int},
    vars::Union{Int,AbstractVector{Int},OrdinalRange},
    labels::AbstractArray{<:AbstractString},
)
    fit = predict_fn(minimizer, train_dataset.tspan, train_dataset.tsteps)
    pred = predict_fn(minimizer, test_dataset.tspan, test_dataset.tsteps)

    plts = []
    for days in forecast_ranges, (col, (var, label)) in enumerate(zip(vars, labels))
        data_fit = train_dataset.data[col, :]
        data_new = test_dataset.data[col, :]

        err = metric_fn(pred[var, 1:days], data_new[1:days])
        title = @sprintf("%d-day loss = %.2f", days, err)
        plt = plot(title = title, legend = :outertop, xrotation = 45)

        scatter!([data_fit[:]; data_new[1:days]], label = label, fillcolor = nothing)
        plot!([fit[var, :]; pred[var, 1:days]], label = "forecast $label", lw = :2)
        vline!([train_dataset.tspan[2]], color = :black, ls = :dash, label = nothing)

        push!(plts, plt)
    end

    nforecasts = length(forecast_ranges)
    nvariables = length(vars)
    return plot(
        plts...,
        layout = (nforecasts, nvariables),
        size = (300 * nvariables, 300 * nforecasts),
    )
end

end
