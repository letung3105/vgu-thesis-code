export initial_params,
    AbstractCovidModel,
    CovidModelSEIRDBaseline,
    CovidModelSEIRDFbMobility1,
    CovidModelSEIRDFbMobility2,
    setup_model,
    DataConfig,
    MobilityConfig

using OrdinaryDiffEq, DiffEqFlux

abstract type AbstractCovidModel end

"""
A struct for containing the SEIRD baseline model

# Fields

* `β_ann`: an neural network that outputs the time-dependent β contact rate
* `problem`: the ODE problem to be solved
"""
struct CovidModelSEIRDBaseline <: AbstractCovidModel
    β_ann::FastChain
    problem::ODEProblem
end

"""
Construct the default SEIRD baseline model

# Arguments

* `u0`: the system initial conditions
* `tspan`: the time span in which the system is considered
"""
function CovidModelSEIRDBaseline(u0::AbstractVector{<:Real}, tspan::Tuple{<:Real,<:Real})
    # small neural network and can be trained faster on CPU
    β_ann =
        FastChain(FastDense(2, 8, relu), FastDense(8, 8, relu), FastDense(8, 1, softplus))
    # system dynamics
    function dudt!(
        du::AbstractVector{<:Real},
        u::AbstractVector{<:Real},
        p::AbstractVector{<:Real},
        t::Real,
    )
        @inbounds begin
            S, E, I, _, _, _, N = u
            γ, λ, α = abs.(@view(p[1:3]))

            # infection rate depends on time, susceptible, and infected
            β = first(β_ann([S / N; I / N], @view p[4:end]))

            du[1] = -β * S * I / N
            du[2] = β * S * I / N - γ * E
            du[3] = γ * E - λ * I
            du[4] = (1 - α) * λ * I
            du[5] = α * λ * I
            du[6] = γ * E
            du[7] = -α * λ * I
        end
        nothing
    end
    prob = ODEProblem(dudt!, u0, tspan)
    return CovidModelSEIRDBaseline(β_ann, prob)
end

"""
Get the initial set of parameters of the baselien SEIRD model with Facebook movement range
"""
initial_params(model::CovidModelSEIRDBaseline) =
    [1 / 2; 1 / 4; 0.025; DiffEqFlux.initial_params(model.β_ann)]

"""
A struct for containing the SEIRD model with Facebook movement range

# Fields

* `β_ann`: an neural network that outputs the time-dependent β contact rate
* `problem`: the ODE problem to be solved
"""
struct CovidModelSEIRDFbMobility1 <: AbstractCovidModel
    β_ann::FastChain
    problem::ODEProblem
end

"""
Construct the default SEIRD model with Facebook movement range data

# Arguments

* `u0`: the system initial conditions
* `tspan`: the time span in which the system is considered
* `movement_range_data`: the matrix for the Facebook movement range timeseries data
"""
function CovidModelSEIRDFbMobility1(
    u0::AbstractVector{<:Real},
    tspan::Tuple{<:Real,<:Real},
    movement_range_data::AbstractMatrix{<:Real},
)
    # small neural network and can be trained faster on CPU
    β_ann =
        FastChain(FastDense(4, 8, relu), FastDense(8, 8, relu), FastDense(8, 1, softplus))
    # system dynamics
    function dudt!(
        du::AbstractVector{<:Real},
        u::AbstractVector{<:Real},
        p::AbstractVector{<:Real},
        t::Real,
    )
        @inbounds begin
            S, E, I, _, _, _, N = u
            γ, λ, α = abs.(@view(p[1:3]))

            # daily mobility
            mobility = movement_range_data[Int(floor(t + 1)), :]
            # infection rate depends on time, susceptible, and infected
            β = first(β_ann([S / N; I / N; mobility...], @view p[4:end]))

            du[1] = -β * S * I / N
            du[2] = β * S * I / N - γ * E
            du[3] = γ * E - λ * I
            du[4] = (1 - α) * λ * I
            du[5] = α * λ * I
            du[6] = γ * E
            du[7] = -α * λ * I
        end
        return nothing
    end
    prob = ODEProblem(dudt!, u0, tspan)
    return CovidModelSEIRDFbMobility1(β_ann, prob)
end

"""
Get the initial set of parameters of the SEIRD model with Facebook movement range
"""
initial_params(model::CovidModelSEIRDFbMobility1) =
    [1 / 2; 1 / 4; 0.025; DiffEqFlux.initial_params(model.β_ann)]

"""
A struct for containing the SEIRD model with Facebook movement range

# Fields

* `β_ann`: an neural network that outputs the time-dependent β contact rate
* `problem`: the ODE problem to be solved
"""
struct CovidModelSEIRDFbMobility2 <: AbstractCovidModel
    β_ann::FastChain
    problem::ODEProblem
end

"""
Construct the default SEIRD model with Facebook movement range data
and social connectedness

# Arguments

* `u0`: the system initial conditions
* `tspan`: the time span in which the system is considered
* `movement_range_data`: the matrix for the Facebook movement range timeseries data
* `spc_date`: the matrix for the Social Proximity to Cases Index
"""
function CovidModelSEIRDFbMobility2(
    u0::AbstractVector{<:Real},
    tspan::Tuple{<:Real,<:Real},
    movement_range_data::AbstractMatrix{<:Real},
    spc_data::AbstractMatrix{<:Real},
)
    # small neural network and can be trained faster on CPU
    β_ann =
        FastChain(FastDense(5, 8, relu), FastDense(8, 8, relu), FastDense(8, 1, softplus))
    # system dynamics
    function dudt!(
        du::AbstractVector{<:Real},
        u::AbstractVector{<:Real},
        p::AbstractVector{<:Real},
        t::Real,
    )
        @inbounds begin
            S, E, I, _, _, _, N = u
            γ, λ, α = abs.(@view(p[1:3]))

            # daily mobility
            time_idx = Int(floor(t + 1))
            mobility = movement_range_data[time_idx, :]
            spc = spc_data[time_idx]
            # infection rate depends on time, susceptible, and infected
            β = first(β_ann([S / N; I / N; spc; mobility...], @view p[4:end]))

            du[1] = -β * S * I / N
            du[2] = β * S * I / N - γ * E
            du[3] = γ * E - λ * I
            du[4] = (1 - α) * λ * I
            du[5] = α * λ * I
            du[6] = γ * E
            du[7] = -α * λ * I
        end
        return nothing
    end
    prob = ODEProblem(dudt!, u0, tspan)
    return CovidModelSEIRDFbMobility2(β_ann, prob)
end

"""
Get the initial set of parameters of the SEIRD model with Facebook movement range
"""
initial_params(model::CovidModelSEIRDFbMobility2) =
    [1 / 2; 1 / 4; 0.025; DiffEqFlux.initial_params(model.β_ann)]

struct DataConfig
    df::AbstractDataFrame
    data_cols::Union{
        Symbol,
        <:AbstractString,
        <:AbstractVector{Symbol},
        <:AbstractVector{<:AbstractString},
    }
    date_col::Union{Symbol,<:AbstractString}
    first_date::Date
    split_date::Date
    last_date::Date
    population::Real
    initial_state_fn::Function
end

struct MobilityConfig
    df::AbstractDataFrame
    data_cols::Union{
        Symbol,
        <:AbstractString,
        <:AbstractVector{Symbol},
        <:AbstractVector{<:AbstractString},
    }
    date_col::Union{Symbol,<:AbstractString}
    temporal_lag::Day
end

function setup_model(
    model_constructor::DataType,
    data_config::DataConfig,
    movement_range_config::Union{MobilityConfig,Nothing} = nothing,
    social_proximity_config::Union{MobilityConfig,Nothing} = nothing,
)
    # separate dataframe into data arrays for train and test
    train_dataset, test_dataset = train_test_split(
        data_config.df,
        data_config.data_cols,
        data_config.date_col,
        data_config.first_date,
        data_config.split_date,
        data_config.last_date,
    )
    # get initial state
    u0 = data_config.initial_state_fn(train_dataset.data[:, 1], data_config.population)

    # use baseline model if no movement range data is given
    if isnothing(movement_range_config)
        model = model_constructor(u0, train_dataset.tspan)
        return model, train_dataset, test_dataset
    end

    # load movement range
    movement_range_data = load_timeseries(
        movement_range_config.df,
        movement_range_config.data_cols,
        movement_range_config.date_col,
        data_config.first_date - movement_range_config.temporal_lag,
        data_config.last_date - movement_range_config.temporal_lag,
    )

    # use fbmobility1 model if no social proximity data is given
    if isnothing(social_proximity_config)
        model = model_constructor(u0, train_dataset.tspan, movement_range_data)
        return model, train_dataset, test_dataset
    end

    # load social proximity
    social_proximity_data = load_timeseries(
        social_proximity_config.df,
        social_proximity_config.data_cols,
        social_proximity_config.date_col,
        data_config.first_date - social_proximity_config.temporal_lag,
        data_config.last_date - social_proximity_config.temporal_lag,
    )
    model = model_constructor(
        u0,
        train_dataset.tspan,
        movement_range_data,
        social_proximity_data,
    )
    return model, train_dataset, test_dataset
end
