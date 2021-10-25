module Helpers

export TimeseriesDataset,
    moving_average!,
    train_test_split,
    load_timeseries,
    save_timeseries_csv,
    Predictor,
    Loss,
    TrainCallback,
    TrainCallbackConfig,
    TrainSession,
    mae,
    mape,
    rmse

using Dates,
    Serialization,
    Statistics,
    Plots,
    DataFrames,
    ProgressMeter,
    OrdinaryDiffEq

"""
This contains the minimum required information for a timeseriese dataset that is used by UDEs

# Fields

* `data::AbstractArray{<:Real}`: an array that holds the timeseries data
* `tspan::Tuple{<:Real,<:Real}`: the first and last time coordinates of the timeseries data
* `tsteps::Union{Real,AbstractVector{<:Real},StepRange,StepRangeLen}`: collocations points
"""
struct TimeseriesDataset
    data::AbstractArray{<:Real}
    tspan::Tuple{<:Real,<:Real}
    tsteps::Union{Real,AbstractVector{<:Real},StepRange,StepRangeLen}
end

moving_average(xs, n::Int) =
    [mean(@view xs[(i >= n ? i - n + 1 : 1):i]) for i = 1:length(xs)]

moving_average!(df::DataFrame, cols, n::Int) =
    transform!(df, names(df, Cols(cols)) .=> x -> moving_average(x, n), renamecols = false)

view_dates_range(df::DataFrame, col, start_date::Date, end_date::Date) =
    view(df, (df[!, col] .>= start_date) .& (df[!, col] .<= end_date), All())

function train_test_split(
    df::DataFrame,
    data_cols,
    date_col,
    first_date::Date,
    split_date::Date,
    last_date::Date,
)
    df_train = view_dates_range(df, date_col, first_date, split_date)
    df_test = view_dates_range(df, date_col, split_date + Day(1), last_date)

    train_tspan = Float64.((0, Dates.value(split_date - first_date)))
    test_tspan = Float64.((0, Dates.value(last_date - first_date)))

    train_tsteps = train_tspan[1]:1:train_tspan[2]
    test_tsteps = (train_tspan[2]+1):1:test_tspan[2]

    train_data = Float64.(Array(df_train[!, data_cols])')
    test_data = Float64.(Array(df_test[!, data_cols])')

    train_dataset = TimeseriesDataset(train_data, train_tspan, train_tsteps)
    test_dataset = TimeseriesDataset(test_data, test_tspan, test_tsteps)

    return train_dataset, test_dataset
end

function load_timeseries(
    df::DataFrame,
    data_cols,
    date_col,
    first_date::Date,
    last_date::Date,
)
    df = view_dates_range(df, date_col, first_date, last_date)
    return Array(df[!, Cols(data_cols)])
end

function save_timeseries_csv(df, fdir, fid; recreate = false)
    fpath = joinpath(fdir, "$fid.csv")

    # file exists and don't need to be updated
    if isfile(fpath) && !recreate
        return fpath
    end
    # create containing folder if not exists
    if !isdir(fdir)
        mkpath(fdir)
    end

    CSV.write(fpath, df)
    return fpath
end


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
Calculate the root mean squared error between 2 values. Note that the input arguments must be of the same size.
The function does not check if the inputs are valid and may produces erroneous output.
"""
rmse(ŷ, y) = sqrt(mean(abs2, ŷ .- y))

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

struct TrainSession
    name::AbstractString
    optimizer::Any
    maxiters::Int
end

end
