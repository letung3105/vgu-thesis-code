export AbstractCovidModel, SEIRDBaseline, SEIRDFbMobility1, SEIRDFbMobility2, ℜe

using OrdinaryDiffEq, DiffEqFlux

abstract type AbstractCovidModel end

"""
A struct for containing the SEIRD baseline model

# Fields

* `β_ann`: an neural network that outputs the time-dependent β contact rate
"""
struct SEIRDBaseline
    β_ann::FastChain
end

"""
Construct the default SEIRD baseline model
"""
SEIRDBaseline() = SEIRDBaseline(
    FastChain(FastDense(2, 8, relu), FastDense(8, 8, relu), FastDense(8, 1, softplus)),
)

@inbounds function (model::SEIRDBaseline)(
    du::AbstractVector{<:Real},
    u::AbstractVector{<:Real},
    p::AbstractVector{<:Real},
    t::Real;
    γ = p[1],
    λ = p[2],
    α = p[3],
    θ = @view(p[4:4+DiffEqFlux.paramlength(model.β_ann)-1]),
)
    # states and params
    S, E, I, _, _, _, N = u
    # infection rate depends on time, susceptible, and infected
    β = first(model.β_ann([S / N; I / N], θ))
    du[1] = -β * S * I / N
    du[2] = β * S * I / N - γ * E
    du[3] = γ * E - λ * I
    du[4] = (1 - α) * λ * I
    du[5] = α * λ * I
    du[6] = γ * E
    du[7] = -α * λ * I
    return nothing
end

function ℜe(
    model::SEIRDBaseline,
    prob::ODEProblem,
    params::AbstractVector{<:Real},
    tspan::Tuple{<:Real,<:Real},
    saveat::Union{<:Real,AbstractVector{<:Real},StepRange,StepRangeLen};
    γ = params[1],
    θ = @view(params[4:4+DiffEqFlux.paramlength(model.β_ann)-1]),
)
    prob = remake(prob, p = params, tspan = tspan)
    sol = solve(
        prob,
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
    βt = model.β_ann(β_ann_input, θ)
    ℜe = βt ./ γ
    return ℜe
end

"""
A struct for containing the SEIRD model with Facebook movement range

# Fields

* `β_ann`: an neural network that outputs the time-dependent β contact rate
* `movement_range_data`: the matrix for the Facebook movement range timeseries data
"""
struct SEIRDFbMobility1 <: AbstractCovidModel
    β_ann::FastChain
    movement_range_data::AbstractMatrix{<:Real}
end

"""
Construct the default SEIRD model with Facebook movement range data

* `movement_range_data`: the matrix for the Facebook movement range timeseries data
"""
SEIRDFbMobility1(movement_range_data::AbstractMatrix{<:Real}) = SEIRDFbMobility1(
    FastChain(FastDense(4, 8, relu), FastDense(8, 8, relu), FastDense(8, 1, softplus)),
    movement_range_data,
)

@inbounds function (model::SEIRDFbMobility1)(
    du::AbstractVector{<:Real},
    u::AbstractVector{<:Real},
    p::AbstractVector{<:Real},
    t::Real;
    γ = p[1],
    λ = p[2],
    α = p[3],
    θ = @view(p[4:4+DiffEqFlux.paramlength(model.β_ann)-1]),
)
    # daily mobility
    mobility = @view model.movement_range_data[:, Int(floor(t + 1))]
    # states and params
    S, E, I, _, _, _, N = u
    # infection rate depends on time, susceptible, and infected
    β = first(model.β_ann([S / N; I / N; mobility...], θ))
    du[1] = -β * S * I / N
    du[2] = β * S * I / N - γ * E
    du[3] = γ * E - λ * I
    du[4] = (1 - α) * λ * I
    du[5] = α * λ * I
    du[6] = γ * E
    du[7] = -α * λ * I
    return nothing
end

function ℜe(
    model::SEIRDFbMobility1,
    prob::ODEProblem,
    params::AbstractVector{<:Real},
    tspan::Tuple{<:Real,<:Real},
    saveat::Union{<:Real,AbstractVector{<:Real},StepRange,StepRangeLen};
    γ = params[1],
    θ = @view(params[4:4+DiffEqFlux.paramlength(model.β_ann)-1]),
)
    prob = remake(prob, p = params, tspan = tspan)
    sol = solve(
        prob,
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
    βt = model.β_ann(β_ann_input, θ)
    ℜe = βt ./ γ
    return ℜe
end

"""
A struct for containing the SEIRD model with Facebook movement range

# Fields

* `β_ann`: an neural network that outputs the time-dependent β contact rate
* `movement_range_data`: the matrix for the Facebook movement range timeseries data
* `social_proximity_data`: the matrix for the social proximity to cases timeseries data
"""
struct SEIRDFbMobility2 <: AbstractCovidModel
    β_ann::FastChain
    movement_range_data::AbstractMatrix{<:Real}
    social_proximity_data::AbstractMatrix{<:Real}
end

SEIRDFbMobility2(
    movement_range_data::AbstractMatrix{<:Real},
    social_proximity_data::AbstractMatrix{<:Real},
) = SEIRDFbMobility2(
    FastChain(FastDense(5, 8, relu), FastDense(8, 8, relu), FastDense(8, 1, softplus)),
    movement_range_data,
    social_proximity_data,
)

@inbounds function (model::SEIRDFbMobility2)(
    du::AbstractVector{<:Real},
    u::AbstractVector{<:Real},
    p::AbstractVector{<:Real},
    t::Real;
    γ = p[1],
    λ = p[2],
    α = p[3],
    θ = @view(p[4:4+DiffEqFlux.paramlength(model.β_ann)-1]),
)
    time_idx = Int(floor(t + 1))
    # daily mobility
    mobility = @view model.movement_range_data[:, time_idx]
    # daily social proximity to cases
    proximity = @view model.social_proximity_data[:, time_idx]
    # states and params
    S, E, I, _, _, _, N = u
    # infection rate depends on time, susceptible, and infected
    β = first(model.β_ann([S / N; I / N; mobility...; proximity...], θ))
    du[1] = -β * S * I / N
    du[2] = β * S * I / N - γ * E
    du[3] = γ * E - λ * I
    du[4] = (1 - α) * λ * I
    du[5] = α * λ * I
    du[6] = γ * E
    du[7] = -α * λ * I
    return nothing
end

function ℜe(
    model::SEIRDFbMobility2,
    prob::ODEProblem,
    params::AbstractVector{<:Real},
    tspan::Tuple{<:Real,<:Real},
    saveat::Union{<:Real,AbstractVector{<:Real},StepRange,StepRangeLen};
    γ = params[1],
    θ = @view(params[4:4+DiffEqFlux.paramlength(model.β_ann)-1]),
)
    prob = remake(prob, p = params, tspan = tspan)
    sol = solve(
        prob,
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
    βt = model.β_ann(β_ann_input, θ)
    ℜe = βt ./ γ
    return ℜe
end
