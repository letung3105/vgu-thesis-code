@inbounds function SEIRD!(du, u, p, t)
    S, E, I, _, _, _, N = u
    β, γ, λ, α = p
    du[1] = -β * S * I / N
    du[2] = β * S * I / N - γ * E
    du[3] = γ * E - λ * I
    du[4] = (1 - α) * λ * I
    du[5] = α * λ * I
    du[6] = γ * E
    du[7] = -α * λ * I
    return nothing
end

"""
An abstract type for representing a Covid-19 model
"""
abstract type AbstractCovidModel end

"""
A struct for containing the SEIRD baseline model. In the model, the β parameter is
time-/covariate-dependent whose value is determined by a neural network.

# Fields

* `β_ann`: an neural network that outputs the time-dependent β contact rate
* `β_ann_paramlength`: number of parameters used by the network
* `γ_bounds`: lower and upper bounds of the γ parameter
* `λ_bounds`: lower and upper bounds of the λ parameter
* `α_bounds`: lower and upper bounds of the α parameter
"""
struct SEIRDBaseline{ANN<:FastChain,T<:Real} <: AbstractCovidModel
    β_ann::ANN
    β_ann_paramlength::Int
    γ_bounds::Tuple{T,T}
    λ_bounds::Tuple{T,T}
    α_bounds::Tuple{T,T}
end

"""
Construct the SEIRDBaseline model

# Arguments

* `γ_bounds`: lower and upper bounds of the γ parameter
* `λ_bounds`: lower and upper bounds of the λ parameter
* `α_bounds`: lower and upper bounds of the α parameter
"""
function SEIRDBaseline(
    γ_bounds::Tuple{T,T},
    λ_bounds::Tuple{T,T},
    α_bounds::Tuple{T,T},
) where {T<:Real}
    β_ann =
        FastChain(FastDense(2, 8, relu), FastDense(8, 8, relu), FastDense(8, 1, softplus))
    SEIRDBaseline(β_ann, DiffEqFlux.paramlength(β_ann), γ_bounds, λ_bounds, α_bounds)
end

@inbounds function (model::SEIRDBaseline)(du, u, p, t)
    # states and params
    S, _, I, _, _, _, N = u
    γ = boxconst(p[1], model.γ_bounds)
    λ = boxconst(p[2], model.λ_bounds)
    α = boxconst(p[3], model.α_bounds)
    θ = @view(p[4:4+model.β_ann_paramlength-1])
    # infection rate depends on time, susceptible, and infected
    β = first(model.β_ann([S / N; I / N], θ))
    SEIRD!(du, u, [β, γ, λ, α], t)
    return nothing
end

"""
Get the initial values for the trainable parameters

# Arguments

* `model`: the model that we want to get the parameterrs for
* `γ0`: initial mean incubation period
* `λ0`: initial mean infectious period
* `α0`: initial mean fatality rate
"""
initparams(model::SEIRDBaseline, γ0::R, λ0::R, α0::R) where {R<:Real} = [
    boxconst_inv(γ0, model.γ_bounds)
    boxconst_inv(λ0, model.λ_bounds)
    boxconst_inv(α0, model.α_bounds)
    DiffEqFlux.initial_params(model.β_ann)
]

"""
Get the effective reproduction rate calculated from the model

# Arguments

+ `model`: the model from which the effective reproduction number is calculated
+ `u0`: the model initial conditions
+ `params`: the model parameters
+ `tspan`: the simulated time span
+ `saveat`: the collocation points that will be saved
"""
function ℜe(
    model::SEIRDBaseline,
    u0::AbstractVector{T},
    params::AbstractVector{T},
    tspan::Tuple{T,T},
    saveat,
) where {T<:Real}
    prob = ODEProblem(model, u0, tspan, params)
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
    θ = @view(params[4:4+model.β_ann_paramlength-1])
    βt = vec(model.β_ann(β_ann_input, θ))
    γ = boxconst(params[1], model.γ_bounds)
    ℜe = βt ./ γ
    return ℜe
end

"""
A struct for containing the SEIRD model with Facebook movement range. In the model, the β
parameter is time-/covariate-dependent whose value is determined by a neural network.

# Fields

* `β_ann`: an neural network that outputs the time-dependent β contact rate
* `β_ann_paramlength`: number of parameters used by the network
* `γ_bounds`: lower and upper bounds of the γ parameter
* `λ_bounds`: lower and upper bounds of the λ parameter
* `α_bounds`: lower and upper bounds of the α parameter
* `movement_range_data`: the matrix for the Facebook movement range timeseries data
"""
struct SEIRDFbMobility1{ANN<:FastChain,T<:Real,DS<:AbstractMatrix{T}} <: AbstractCovidModel
    β_ann::ANN
    β_ann_paramlength::Int
    γ_bounds::Tuple{T,T}
    λ_bounds::Tuple{T,T}
    α_bounds::Tuple{T,T}
    movement_range_data::DS
end

"""
Construct the SEIRDFbMobility1 model

# Arguments

* `γ_bounds`: lower and upper bounds of the γ parameter
* `λ_bounds`: lower and upper bounds of the λ parameter
* `α_bounds`: lower and upper bounds of the α parameter
* `movement_range_data`: the matrix for the Facebook movement range timeseries data
"""
function SEIRDFbMobility1(
    γ_bounds::Tuple{T,T},
    λ_bounds::Tuple{T,T},
    α_bounds::Tuple{T,T},
    movement_range_data::DS,
) where {T<:Real,DS<:AbstractMatrix{T}}
    β_ann =
        FastChain(FastDense(4, 8, relu), FastDense(8, 8, relu), FastDense(8, 1, softplus))
    SEIRDFbMobility1(
        β_ann,
        DiffEqFlux.paramlength(β_ann),
        γ_bounds,
        λ_bounds,
        α_bounds,
        movement_range_data,
    )
end

@inbounds function (model::SEIRDFbMobility1)(du, u, p, t)
    # daily mobility
    mobility = @view model.movement_range_data[:, Int(floor(t + 1))]
    # states and params
    S, _, I, _, _, _, N = u
    γ = boxconst(p[1], model.γ_bounds)
    λ = boxconst(p[2], model.λ_bounds)
    α = boxconst(p[3], model.α_bounds)
    θ = @view(p[4:4+model.β_ann_paramlength-1])
    # infection rate depends on time, susceptible, and infected
    β = first(model.β_ann([S / N; I / N; mobility...], θ))
    SEIRD!(du, u, [β, γ, λ, α], t)
    return nothing
end

"""
Get the initial values for the trainable parameters

# Arguments

* `model`: the model that we want to get the parameterrs for
* `γ0`: initial mean incubation period
* `λ0`: initial mean infectious period
* `α0`: initial mean fatality rate
"""
initparams(model::SEIRDFbMobility1, γ0::R, λ0::R, α0::R) where {R<:Real} = [
    boxconst_inv(γ0, model.γ_bounds)
    boxconst_inv(λ0, model.λ_bounds)
    boxconst_inv(α0, model.α_bounds)
    DiffEqFlux.initial_params(model.β_ann)
]

"""
Get the effective reproduction rate calculated from the model

# Arguments

+ `model`: the model from which the effective reproduction number is calculated
+ `u0`: the model initial conditions
+ `params`: the model parameters
+ `tspan`: the simulated time span
+ `saveat`: the collocation points that will be saved
"""
function ℜe(
    model::SEIRDFbMobility1,
    u0::AbstractVector{T},
    params::AbstractVector{T},
    tspan::Tuple{T,T},
    saveat,
) where {T<:Real}
    prob = ODEProblem(model, u0, tspan, params)
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
    θ = @view(params[4:4+model.β_ann_paramlength-1])
    βt = vec(model.β_ann(β_ann_input, θ))
    γ = boxconst(params[1], model.γ_bounds)
    ℜe = βt ./ γ
    return ℜe
end

"""
A struct for containing the SEIRD model that uses Facebook movement range maps dataset
and Facebook social connectedness index dataset. In the model, the β parameter is
time-/covariate-dependent whose value is determined by a neural network

# Fields

* `β_ann`: an neural network that outputs the time-dependent β contact rate
* `β_ann_paramlength`: number of parameters used by the network
* `γ_bounds`: lower and upper bounds of the γ parameter
* `λ_bounds`: lower and upper bounds of the λ parameter
* `α_bounds`: lower and upper bounds of the α parameter
* `movement_range_data`: the matrix for the Facebook movement range timeseries data
* `social_proximity_data`: the matrix for the social proximity to cases timeseries data
"""
struct SEIRDFbMobility2{ANN<:FastChain,T<:Real,DS<:AbstractMatrix{T}} <: AbstractCovidModel
    β_ann::ANN
    β_ann_paramlength::Int
    γ_bounds::Tuple{T,T}
    λ_bounds::Tuple{T,T}
    α_bounds::Tuple{T,T}
    movement_range_data::DS
    social_proximity_data::DS
end

"""
Construct the SEIRDFbMobility2 model

# Arguments

* `γ_bounds`: lower and upper bounds of the γ parameter
* `λ_bounds`: lower and upper bounds of the λ parameter
* `α_bounds`: lower and upper bounds of the α parameter
* `movement_range_data`: the matrix for the Facebook movement range timeseries data
* `social_proximity_data`: the matrix for the social proximity to cases timeseries data
"""
function SEIRDFbMobility2(
    γ_bounds::Tuple{T,T},
    λ_bounds::Tuple{T,T},
    α_bounds::Tuple{T,T},
    movement_range_data::DS,
    social_proximity_data::DS,
) where {T<:Real,DS<:AbstractMatrix{T}}
    β_ann =
        FastChain(FastDense(5, 8, relu), FastDense(8, 8, relu), FastDense(8, 1, softplus))
    SEIRDFbMobility2(
        β_ann,
        DiffEqFlux.paramlength(β_ann),
        γ_bounds,
        λ_bounds,
        α_bounds,
        movement_range_data,
        social_proximity_data,
    )
end

@inbounds function (model::SEIRDFbMobility2)(du, u, p, t)
    time_idx = Int(floor(t + 1))
    # daily mobility
    mobility = @view model.movement_range_data[:, time_idx]
    # daily social proximity to cases
    proximity = @view model.social_proximity_data[:, time_idx]
    # states and params
    S, _, I, _, _, _, N = u
    γ = boxconst(p[1], model.γ_bounds)
    λ = boxconst(p[2], model.λ_bounds)
    α = boxconst(p[3], model.α_bounds)
    θ = @view(p[4:4+model.β_ann_paramlength-1])
    # infection rate depends on time, susceptible, and infected
    β = first(model.β_ann([S / N; I / N; mobility...; proximity...], θ))
    SEIRD!(du, u, [β, γ, λ, α], t)
    return nothing
end

"""
Get the initial values for the trainable parameters

# Arguments

* `model`: the model that we want to get the parameterrs for
* `γ0`: initial mean incubation period
* `λ0`: initial mean infectious period
* `α0`: initial mean fatality rate
"""
initparams(model::SEIRDFbMobility2, γ0::R, λ0::R, α0::R) where {R<:Real} = [
    boxconst_inv(γ0, model.γ_bounds)
    boxconst_inv(λ0, model.λ_bounds)
    boxconst_inv(α0, model.α_bounds)
    DiffEqFlux.initial_params(model.β_ann)
]

"""
Get the effective reproduction rate calculated from the model

# Arguments

+ `model`: the model from which the effective reproduction number is calculated
+ `u0`: the model initial conditions
+ `params`: the model parameters
+ `tspan`: the simulated time span
+ `saveat`: the collocation points that will be saved
"""
function ℜe(
    model::SEIRDFbMobility2,
    u0::AbstractVector{T},
    params::AbstractVector{T},
    tspan::Tuple{T,T},
    saveat,
) where {T<:Real}
    prob = ODEProblem(model, u0, tspan, params)
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
    θ = @view(params[4:4+model.β_ann_paramlength-1])
    βt = vec(model.β_ann(β_ann_input, θ))
    γ = boxconst(params[1], model.γ_bounds)
    ℜe = βt ./ γ
    return ℜe
end

"""
A struct for containing the SEIRD model that uses Facebook movement range maps dataset
and Facebook social connectedness index dataset. In the model, both the β and α are
time-/covariate-dependent whose values are determined by 2 separate neural networks

# Fields

* `β_ann`: an neural network that outputs the time-dependent β contact rate
* `β_ann_paramlength`: number of parameters used by the network
* `α_ann`: an neural network that outputs the time-dependent α fatality rate
* `α_ann_paramlength`: number of parameters used by the network
* `γ_bounds`: lower and upper bounds of the γ parameter
* `λ_bounds`: lower and upper bounds of the λ parameter
* `α_bounds`: lower and upper bounds of the α parameter
* `movement_range_data`: the matrix for the Facebook movement range timeseries data
* `social_proximity_data`: the matrix for the social proximity to cases timeseries data
"""
struct SEIRDFbMobility3{ANN1<:FastChain,ANN2<:FastChain,T<:Real,DS<:AbstractMatrix{T}} <:
       AbstractCovidModel
    β_ann::ANN1
    α_ann::ANN2
    β_ann_paramlength::Int
    α_ann_paramlength::Int
    γ_bounds::Tuple{T,T}
    λ_bounds::Tuple{T,T}
    α_bounds::Tuple{T,T}
    movement_range_data::DS
    social_proximity_data::DS
end

"""
Construct the SEIRDFbMobility3 model

# Arguments

* `γ_bounds`: lower and upper bounds of the γ parameter
* `λ_bounds`: lower and upper bounds of the λ parameter
* `α_bounds`: lower and upper bounds of the α parameter
* `movement_range_data`: the matrix for the Facebook movement range timeseries data
* `social_proximity_data`: the matrix for the social proximity to cases timeseries data
"""
function SEIRDFbMobility3(
    γ_bounds::Tuple{T,T},
    λ_bounds::Tuple{T,T},
    α_bounds::Tuple{T,T},
    movement_range_data::DS,
    social_proximity_data::DS,
) where {T<:Real,DS<:AbstractMatrix{T}}
    β_ann = FastChain(
        FastDense(5, 8, relu),
        FastDense(8, 8, relu),
        FastDense(8, 8, relu),
        FastDense(8, 1, softplus),
    )
    α_ann = FastChain(
        FastDense(3, 8, relu),
        FastDense(8, 8, relu),
        FastDense(8, 8, relu),
        FastDense(8, 1, x -> boxconst(x, α_bounds)),
    )
    return SEIRDFbMobility3(
        β_ann,
        α_ann,
        DiffEqFlux.paramlength(β_ann),
        DiffEqFlux.paramlength(α_ann),
        γ_bounds,
        λ_bounds,
        α_bounds,
        movement_range_data,
        social_proximity_data,
    )
end

@inbounds function (model::SEIRDFbMobility3)(du, u, p, t)
    S, _, I, R, D, _, N = u
    γ = boxconst(p[1], model.γ_bounds)
    λ = boxconst(p[2], model.λ_bounds)
    θ1 = @view(p[3:3+model.β_ann_paramlength-1])
    θ2 = @view(p[end-model.α_ann_paramlength:end])
    # daily mobility
    time_idx = Int(floor(t + 1))
    mobility = @view model.movement_range_data[:, time_idx]
    proximity = @view model.social_proximity_data[:, time_idx]
    # infection rate depends on time, susceptible, and infected
    β = first(model.β_ann([S / N; I / N; mobility...; proximity...], θ1))
    α = first(model.α_ann([I / N; R / N; D / N], θ2))
    Covid19ModelVN.SEIRD!(du, u, [β, γ, λ, α], t)
    return nothing
end

"""
Get the initial values for the trainable parameters

# Arguments

* `model`: the model that we want to get the parameterrs for
* `γ0`: initial mean incubation period
* `λ0`: initial mean infectious period
"""
initparams(model::SEIRDFbMobility3, γ0::R, λ0::R) where {R<:Real} = [
    boxconst_inv(γ0, model.γ_bounds)
    boxconst_inv(λ0, model.λ_bounds)
    DiffEqFlux.initial_params(model.β_ann)
    DiffEqFlux.initial_params(model.α_ann)
]

"""
Get the effective reproduction rate calculated from the model

# Arguments

+ `model`: the model from which the effective reproduction number is calculated
+ `u0`: the model initial conditions
+ `params`: the model parameters
+ `tspan`: the simulated time span
+ `saveat`: the collocation points that will be saved
"""
function ℜe(
    model::SEIRDFbMobility3,
    u0::AbstractVector{T},
    params::AbstractVector{T},
    tspan::Tuple{T,T},
    saveat::Ts,
) where {T<:Real,Ts}
    prob = ODEProblem(model, u0, tspan, params)
    sol = solve(
        prob,
        Tsit5(),
        saveat = saveat,
        solver = InterpolatingAdjoint(autojacvec = ReverseDiffVJP(true)),
    )
    states = Array(sol)
    S = @view states[1, :]
    I = @view states[3, :]
    N = @view states[7, :]
    mobility = @view model.movement_range_data[:, Int.(saveat).+1]
    proximity = @view model.social_proximity_data[:, Int.(saveat).+1]
    β_ann_input = [(S ./ N)'; (I ./ N)'; mobility; proximity]
    θ = @view(params[3:3+model.β_ann_paramlength-1])
    βt = vec(model.β_ann(β_ann_input, θ))
    γ = boxconst(params[1], model.γ_bounds)
    ℜe = βt ./ γ
    return ℜe
end
