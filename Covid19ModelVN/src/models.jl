export AbstractCovidModel,
    CovidModelSEIRDBaseline,
    CovidModelSEIRDFbMobility1,
    CovidModelSEIRDFbMobility2,
    initial_params,
    effective_reproduction_number,
    DataConfig,
    MobilityConfig,
    setup_model

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
            # states and params
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

function effective_reproduction_number(
    model::CovidModelSEIRDBaseline,
    params::AbstractVector{<:Real},
    tspan::Tuple{<:Real,<:Real},
    saveat::Union{<:Real,AbstractVector{<:Real},StepRange,StepRangeLen},
)
    problem = remake(model.problem, p = params, tspan = tspan)
    sol = solve(
        problem,
        Tsit5(),
        saveat = saveat,
        solver = InterpolatingAdjoint(autojacvec = ReverseDiffVJP(true)),
        abstol = 1e-6,
        reltol = 1e-6,
    )

    states = Array(sol)
    S = @view states[1, :]
    I = @view states[3, :]
    N = @view states[7, :]

    β_ann_input = [(S ./ N)'; (I ./ N)']
    βt = model.β_ann(β_ann_input, @view params[4:end])
    Rt = βt ./ params[1]
    return Rt
end

"""
A struct for containing the SEIRD model with Facebook movement range

# Fields

* `β_ann`: an neural network that outputs the time-dependent β contact rate
* `problem`: the ODE problem to be solved
"""
struct CovidModelSEIRDFbMobility1 <: AbstractCovidModel
    β_ann::FastChain
    problem::ODEProblem
    movement_range_data::AbstractMatrix{<:Real}
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
            # daily mobility
            mobility = @view movement_range_data[:, Int(floor(t + 1))]
            # states and params
            S, E, I, _, _, _, N = u
            γ, λ, α = abs.(@view(p[1:3]))
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
    return CovidModelSEIRDFbMobility1(β_ann, prob, movement_range_data)
end

"""
Get the initial set of parameters of the SEIRD model with Facebook movement range
"""
initial_params(model::CovidModelSEIRDFbMobility1) =
    [1 / 2; 1 / 4; 0.025; DiffEqFlux.initial_params(model.β_ann)]

function effective_reproduction_number(
    model::CovidModelSEIRDFbMobility1,
    params::AbstractVector{<:Real},
    tspan::Tuple{<:Real,<:Real},
    saveat::Union{<:Real,AbstractVector{<:Real},StepRange,StepRangeLen},
)
    problem = remake(model.problem, p = params, tspan = tspan)
    sol = solve(
        problem,
        Tsit5(),
        saveat = saveat,
        solver = InterpolatingAdjoint(autojacvec = ReverseDiffVJP(true)),
        abstol = 1e-6,
        reltol = 1e-6,
    )

    states = Array(sol)
    S = @view states[1, :]
    I = @view states[3, :]
    N = @view states[7, :]
    mobility = @view model.movement_range_data[:, Int.(saveat).+1]

    β_ann_input = [(S ./ N)'; (I ./ N)'; mobility]
    βt = model.β_ann(β_ann_input, @view params[4:end])
    Rt = βt ./ params[1]
    return Rt
end

"""
A struct for containing the SEIRD model with Facebook movement range

# Fields

* `β_ann`: an neural network that outputs the time-dependent β contact rate
* `problem`: the ODE problem to be solved
"""
struct CovidModelSEIRDFbMobility2 <: AbstractCovidModel
    β_ann::FastChain
    problem::ODEProblem
    movement_range_data::AbstractMatrix{<:Real}
    social_proximity_data::AbstractMatrix{<:Real}
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
    social_proximity_data::AbstractMatrix{<:Real},
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
            time_idx = Int(floor(t + 1))
            # daily mobility
            mobility = @view movement_range_data[:, time_idx]
            # daily social proximity to cases
            proximity = @view social_proximity_data[:, time_idx]
            # states and params
            S, E, I, _, _, _, N = u
            γ, λ, α = abs.(@view(p[1:3]))
            # infection rate depends on time, susceptible, and infected
            β = first(β_ann([S / N; I / N; mobility...; proximity...], @view p[4:end]))

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
    return CovidModelSEIRDFbMobility2(
        β_ann,
        prob,
        movement_range_data,
        social_proximity_data,
    )
end

"""
Get the initial set of parameters of the SEIRD model with Facebook movement range
"""
initial_params(model::CovidModelSEIRDFbMobility2) =
    [1 / 2; 1 / 4; 0.025; DiffEqFlux.initial_params(model.β_ann)]

function effective_reproduction_number(
    model::CovidModelSEIRDFbMobility2,
    params::AbstractVector{<:Real},
    tspan::Tuple{<:Real,<:Real},
    saveat::Union{<:Real,AbstractVector{<:Real},StepRange,StepRangeLen},
)
    problem = remake(model.problem, p = params, tspan = tspan)
    sol = solve(
        problem,
        Tsit5(),
        saveat = saveat,
        solver = InterpolatingAdjoint(autojacvec = ReverseDiffVJP(true)),
        abstol = 1e-6,
        reltol = 1e-6,
    )

    states = Array(sol)
    S = @view states[1, :]
    I = @view states[3, :]
    N = @view states[7, :]
    mobility = @view model.movement_range_data[:, Int.(saveat).+1]
    proximity = @view model.social_proximity_data[:, Int.(saveat).+1]

    β_ann_input = [(S ./ N)'; (I ./ N)'; mobility; proximity]
    βt = model.β_ann(β_ann_input, @view params[4:end])
    Rt = βt ./ params[1]
    return Rt
end
