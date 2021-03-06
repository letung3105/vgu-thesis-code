"""
    AbstractCovidModel

An abstract type for representing a Covid-19 model
"""
abstract type AbstractCovidModel end

"""
    SEIRDBaseline{ANN<:FastChain,T<:Real} <: AbstractCovidModel

A struct for containing the SEIRD baseline model. In the model, the β parameter is
time-/covariate-dependent whose value is determined by a neural network.

# Fields

* `β_ann`: an neural network that outputs the time-dependent β contact rate
* `α_ann`: an neural network that outputs the time-dependent α fatality rate
* `β_ann_paramlength`: number of parameters used by the network
* `α_ann_paramlength`: number of parameters used by the network
* `β_bounds`: lower and upper bounds of the β parameter
* `γ_bounds`: lower and upper bounds of the γ parameter
* `λ_bounds`: lower and upper bounds of the λ parameter
* `α_bounds`: lower and upper bounds of the α parameter
* `population`: population of the area that is being modelled
* `time_scale`: the length of time that is simulated

# Constructor

    SEIRDBaseline(
        β_bounds::Tuple{T,T},
        γ_bounds::Tuple{T,T},
        λ_bounds::Tuple{T,T},
        α_bounds::Tuple{T,T},
        population::T,
        time_scale::T,
    ) where {T<:Real}

## Arguments

* `β_bounds`: lower and upper bounds of the β parameter
* `γ_bounds`: lower and upper bounds of the γ parameter
* `λ_bounds`: lower and upper bounds of the λ parameter
* `α_bounds`: lower and upper bounds of the α parameter
* `population`: population of the area that is being modelled
* `time_scale`: the length of time that is simulated
"""
struct SEIRDBaseline{ANN1<:FastChain,ANN2<:FastChain,T<:Real} <: AbstractCovidModel
    β_ann::ANN1
    α_ann::ANN2
    β_ann_paramlength::Int
    α_ann_paramlength::Int
    β_bounds::Tuple{T,T}
    γ_bounds::Tuple{T,T}
    λ_bounds::Tuple{T,T}
    α_bounds::Tuple{T,T}
    population::T
    time_scale::T

    function SEIRDBaseline(
        β_bounds::Tuple{T,T},
        γ_bounds::Tuple{T,T},
        λ_bounds::Tuple{T,T},
        α_bounds::Tuple{T,T},
        population::T,
        time_scale::T,
    ) where {T<:Real}
        β_ann = FastChain(
            StaticDense(4, 8, mish, initW = Flux.glorot_normal),
            StaticDense(8, 8, mish, initW = Flux.glorot_normal),
            StaticDense(8, 8, mish, initW = Flux.glorot_normal),
            StaticDense(8, 1, x -> boxconst(x, β_bounds), initW = Flux.glorot_normal),
        )
        α_ann = FastChain(
            StaticDense(4, 8, mish, initW = Flux.glorot_normal),
            StaticDense(8, 1, x -> boxconst(x, α_bounds), initW = Flux.glorot_normal),
        )
        return new{typeof(β_ann),typeof(α_ann),T}(
            β_ann,
            α_ann,
            DiffEqFlux.paramlength(β_ann),
            DiffEqFlux.paramlength(α_ann),
            β_bounds,
            γ_bounds,
            λ_bounds,
            α_bounds,
            population,
            time_scale,
        )
    end
end

"""
    SEIRDFbMobility1{ANN<:FastChain,T<:Real,DS<:AbstractMatrix{T}} <: AbstractCovidModel

A struct for containing the SEIRD model with Facebook movement range. In the model, the β
parameter is time-/covariate-dependent whose value is determined by a neural network.

# Fields

* `β_ann`: an neural network that outputs the time-dependent β contact rate
* `α_ann`: an neural network that outputs the time-dependent α fatality rate
* `β_ann_paramlength`: number of parameters used by the network
* `α_ann_paramlength`: number of parameters used by the network
* `β_bounds`: lower and upper bounds of the β parameter
* `γ_bounds`: lower and upper bounds of the γ parameter
* `λ_bounds`: lower and upper bounds of the λ parameter
* `α_bounds`: lower and upper bounds of the α parameter
* `population`: population of the area that is being modelled
* `time_scale`: the length of time that is simulated
* `movement_range_data`: the matrix for the Facebook movement range timeseries data

# Constructor

    SEIRDFbMobility1(
        β_bounds::Tuple{T,T},
        γ_bounds::Tuple{T,T},
        λ_bounds::Tuple{T,T},
        α_bounds::Tuple{T,T},
        population::T,
        time_scale::T,
        movement_range_data::DS,
    ) where {T<:Real,DS<:AbstractMatrix{T}}

## Arguments

* `β_bounds`: lower and upper bounds of the β parameter
* `γ_bounds`: lower and upper bounds of the γ parameter
* `λ_bounds`: lower and upper bounds of the λ parameter
* `α_bounds`: lower and upper bounds of the α parameter
* `population`: population of the area that is being modelled
* `time_scale`: the length of time that is simulated
* `movement_range_data`: the matrix for the Facebook movement range timeseries data
"""
struct SEIRDFbMobility1{ANN1<:FastChain,ANN2<:FastChain,T<:Real,DS<:AbstractMatrix{T}} <:
       AbstractCovidModel
    β_ann::ANN1
    α_ann::ANN2
    β_ann_paramlength::Int
    α_ann_paramlength::Int
    β_bounds::Tuple{T,T}
    γ_bounds::Tuple{T,T}
    λ_bounds::Tuple{T,T}
    α_bounds::Tuple{T,T}
    population::T
    time_scale::T
    movement_range_data::DS

    function SEIRDFbMobility1(
        β_bounds::Tuple{T,T},
        γ_bounds::Tuple{T,T},
        λ_bounds::Tuple{T,T},
        α_bounds::Tuple{T,T},
        population::T,
        time_scale::T,
        movement_range_data::DS,
    ) where {T<:Real,DS<:AbstractMatrix{T}}
        β_ann = FastChain(
            StaticDense(6, 8, mish, initW = Flux.glorot_normal),
            StaticDense(8, 8, mish, initW = Flux.glorot_normal),
            StaticDense(8, 8, mish, initW = Flux.glorot_normal),
            StaticDense(8, 1, x -> boxconst(x, β_bounds), initW = Flux.glorot_normal),
        )
        α_ann = FastChain(
            StaticDense(4, 8, mish, initW = Flux.glorot_normal),
            StaticDense(8, 1, x -> boxconst(x, α_bounds), initW = Flux.glorot_normal),
        )
        return new{typeof(β_ann),typeof(α_ann),T,DS}(
            β_ann,
            α_ann,
            DiffEqFlux.paramlength(β_ann),
            DiffEqFlux.paramlength(α_ann),
            β_bounds,
            γ_bounds,
            λ_bounds,
            α_bounds,
            population,
            time_scale,
            movement_range_data,
        )
    end
end

"""
    SEIRDFbMobility2{ANN1<:FastChain,ANN2<:FastChain,T<:Real,DS<:AbstractMatrix{T}} <:

A struct for containing the SEIRD model that uses Facebook movement range maps dataset
and Facebook social connectedness index dataset. In the model, both the β and α are
time-/covariate-dependent whose values are determined by 2 separate neural networks

# Fields

* `β_ann`: an neural network that outputs the time-dependent β contact rate
* `β_ann_paramlength`: number of parameters used by the network
* `α_ann`: an neural network that outputs the time-dependent α fatality rate
* `α_ann_paramlength`: number of parameters used by the network
* `β_bounds`: lower and upper bounds of the α parameter
* `γ_bounds`: lower and upper bounds of the γ parameter
* `λ_bounds`: lower and upper bounds of the λ parameter
* `α_bounds`: lower and upper bounds of the α parameter
* `movement_range_data`: the matrix for the Facebook movement range timeseries data
* `social_proximity_data`: the matrix for the social proximity to cases timeseries data

# Constructor

    SEIRDFbMobility2(
        β_bounds::Tuple{T,T},
        γ_bounds::Tuple{T,T},
        λ_bounds::Tuple{T,T},
        α_bounds::Tuple{T,T},
        population::T,
        time_scale::T,
        movement_range_data::DS,
        social_proximity_data::DS,
    ) where {T<:Real,DS<:AbstractMatrix{T}}

## Arguments

* `β_bounds`: lower and upper bounds of the β parameter
* `γ_bounds`: lower and upper bounds of the γ parameter
* `λ_bounds`: lower and upper bounds of the λ parameter
* `α_bounds`: lower and upper bounds of the α parameter
* `population`: population of the area that is being modelled
* `time_scale`: the length of time that is simulated
* `movement_range_data`: the matrix for the Facebook movement range timeseries data
* `social_proximity_data`: the matrix for the social proximity to cases timeseries data
"""
struct SEIRDFbMobility2{ANN1<:FastChain,ANN2<:FastChain,T<:Real,DS<:AbstractMatrix{T}} <:
       AbstractCovidModel
    β_ann::ANN1
    α_ann::ANN2
    β_ann_paramlength::Int
    α_ann_paramlength::Int
    β_bounds::Tuple{T,T}
    γ_bounds::Tuple{T,T}
    λ_bounds::Tuple{T,T}
    α_bounds::Tuple{T,T}
    population::T
    time_scale::T
    movement_range_data::DS
    social_proximity_data::DS

    function SEIRDFbMobility2(
        β_bounds::Tuple{T,T},
        γ_bounds::Tuple{T,T},
        λ_bounds::Tuple{T,T},
        α_bounds::Tuple{T,T},
        population::T,
        time_scale::T,
        movement_range_data::DS,
        social_proximity_data::DS,
    ) where {T<:Real,DS<:AbstractMatrix{T}}
        β_ann = FastChain(
            StaticDense(7, 8, mish, initW = Flux.glorot_normal),
            StaticDense(8, 8, mish, initW = Flux.glorot_normal),
            StaticDense(8, 8, mish, initW = Flux.glorot_normal),
            StaticDense(8, 1, x -> boxconst(x, β_bounds), initW = Flux.glorot_normal),
        )
        α_ann = FastChain(
            StaticDense(4, 8, mish, initW = Flux.glorot_normal),
            StaticDense(8, 1, x -> boxconst(x, α_bounds), initW = Flux.glorot_normal),
        )
        return new{typeof(β_ann),typeof(α_ann),T,DS}(
            β_ann,
            α_ann,
            DiffEqFlux.paramlength(β_ann),
            DiffEqFlux.paramlength(α_ann),
            β_bounds,
            γ_bounds,
            λ_bounds,
            α_bounds,
            population,
            time_scale,
            movement_range_data,
            social_proximity_data,
        )
    end
end

"""
    initparams(model::SEIRDFbMobility2, γ0::R, λ0::R, α0::R) where {R<:Real}

Get the initial values for the trainable parameters

# Arguments

* `model`: the model that we want to get the parameterrs for
* `γ0`: initial mean incubation period
* `λ0`: initial mean infectious period
* `α0`: initial mean fatality rate
"""
function initparams(model::AbstractCovidModel, γ0::R, λ0::R) where {R<:Real}
    return [
        model.γ_bounds[1] == model.γ_bounds[2] ? γ0 : boxconst_inv(γ0, model.γ_bounds)
        model.λ_bounds[1] == model.λ_bounds[2] ? λ0 : boxconst_inv(λ0, model.λ_bounds)
        DiffEqFlux.initial_params(model.β_ann)
        DiffEqFlux.initial_params(model.α_ann)
    ]
end

"""
    namedparams(model::SEIRDFbMobility2, params::AbstractVector{<:Real})

Get a named tuple of the parameters that are used by the augmented SEIRD model

# Arguments

* `model`: the object of type the object representing the augmented model
* `params`: a vector of all the parameters used by the model
"""
function namedparams(model::AbstractCovidModel, params::AbstractVector{<:Real})
    @inbounds (
        γ=boxconst(params[1], model.γ_bounds),
        λ=boxconst(params[2], model.λ_bounds),
        θ1=@view(params[3:(3 + model.β_ann_paramlength - 1)]),
        θ2=@view(params[(end - model.α_ann_paramlength + 1):end]),
    )
end

function fatality_rate(
    model::SEIRDBaseline,
    u0::AbstractVector{T},
    params::AbstractVector{T},
    tspan::Tuple{T,T},
    saveat::Ts,
) where {T<:Real,Ts}
    sol = default_solve(model, u0, params, tspan, saveat)
    states = Array(sol)
    I = @view states[3, :]
    R = @view states[4, :]
    D = @view states[5, :]
    N = @view states[6, :]

    pnamed = namedparams(model, params)
    α_ann_input = [
        (collect(saveat) ./ model.time_scale)'
        (I ./ N)'
        (R ./ N)'
        (D ./ N)'
    ]
    αt = vec(model.α_ann(α_ann_input, pnamed.θ2))
    return αt
end

function fatality_rate(
    model::SEIRDFbMobility1,
    u0::AbstractVector{T},
    params::AbstractVector{T},
    tspan::Tuple{T,T},
    saveat::Ts,
) where {T<:Real,Ts}
    sol = default_solve(model, u0, params, tspan, saveat)
    states = Array(sol)
    I = @view states[3, :]
    R = @view states[4, :]
    D = @view states[5, :]
    N = @view states[6, :]

    pnamed = namedparams(model, params)
    α_ann_input = [
        (collect(saveat) ./ model.time_scale)'
        (I ./ N)'
        (R ./ N)'
        (D ./ N)'
    ]
    αt = vec(model.α_ann(α_ann_input, pnamed.θ2))
    return αt
end

function fatality_rate(
    model::SEIRDFbMobility2,
    u0::AbstractVector{T},
    params::AbstractVector{T},
    tspan::Tuple{T,T},
    saveat::Ts,
) where {T<:Real,Ts}
    sol = default_solve(model, u0, params, tspan, saveat)
    states = Array(sol)
    I = @view states[3, :]
    R = @view states[4, :]
    D = @view states[5, :]
    N = @view states[6, :]

    pnamed = namedparams(model, params)
    α_ann_input = [
        (collect(saveat) ./ model.time_scale)'
        (I ./ N)'
        (R ./ N)'
        (D ./ N)'
    ]
    αt = vec(model.α_ann(α_ann_input, pnamed.θ2))
    return αt
end

"""
    Re(
        model::SEIRDBaseline,
        u0::AbstractVector{T},
        params::AbstractVector{T},
        tspan::Tuple{T,T},
        saveat,
    ) where {T<:Real}

Get the effective reproduction rate calculated from the model

# Arguments

+ `model`: the model from which the effective reproduction number is calculated
+ `u0`: the model initial conditions
+ `params`: the model parameters
+ `tspan`: the simulated time span
+ `saveat`: the collocation points that will be saved
"""
function Re(
    model::SEIRDBaseline,
    u0::AbstractVector{T},
    params::AbstractVector{T},
    tspan::Tuple{T,T},
    saveat,
) where {T<:Real}
    sol = default_solve(model, u0, params, tspan, saveat)
    states = Array(sol)
    S = @view states[1, :]
    E = @view states[2, :]
    I = @view states[3, :]
    N = @view states[6, :]

    pnamed = namedparams(model, params)
    β_ann_input = [
        (collect(saveat) ./ model.time_scale)'
        (S ./ N)'
        (E ./ N)'
        (I ./ N)'
    ]

    βt = vec(model.β_ann(β_ann_input, pnamed.θ1))
    Re = βt ./ pnamed.γ
    return Re
end

"""
    Re(
        model::SEIRDFbMobility1,
        u0::AbstractVector{T},
        params::AbstractVector{T},
        tspan::Tuple{T,T},
        saveat,
    ) where {T<:Real}

Get the effective reproduction rate calculated from the model

# Arguments

+ `model`: the model from which the effective reproduction number is calculated
+ `u0`: the model initial conditions
+ `params`: the model parameters
+ `tspan`: the simulated time span
+ `saveat`: the collocation points that will be saved
"""
function Re(
    model::SEIRDFbMobility1,
    u0::AbstractVector{T},
    params::AbstractVector{T},
    tspan::Tuple{T,T},
    saveat,
) where {T<:Real}
    sol = default_solve(model, u0, params, tspan, saveat)
    states = Array(sol)
    S = @view states[1, :]
    E = @view states[2, :]
    I = @view states[3, :]
    N = @view states[6, :]

    pnamed = namedparams(model, params)
    mobility = @view model.movement_range_data[:, Int.(saveat) .+ 1]
    β_ann_input = [
        (collect(saveat) ./ model.time_scale)'
        (S ./ N)'
        (E ./ N)'
        (I ./ N)'
        mobility
    ]

    βt = vec(model.β_ann(β_ann_input, pnamed.θ1))
    Re = βt ./ pnamed.γ
    return Re
end

"""
    function Re(
        model::SEIRDFbMobility2,
        u0::AbstractVector{T},
        params::AbstractVector{T},
        tspan::Tuple{T,T},
        saveat::Ts,
    ) where {T<:Real,Ts}

Get the effective reproduction rate calculated from the model

# Arguments

+ `model`: the model from which the effective reproduction number is calculated
+ `u0`: the model initial conditions
+ `params`: the model parameters
+ `tspan`: the simulated time span
+ `saveat`: the collocation points that will be saved
"""
function Re(
    model::SEIRDFbMobility2,
    u0::AbstractVector{T},
    params::AbstractVector{T},
    tspan::Tuple{T,T},
    saveat::Ts,
) where {T<:Real,Ts}
    sol = default_solve(model, u0, params, tspan, saveat)
    states = Array(sol)
    S = @view states[1, :]
    E = @view states[2, :]
    I = @view states[3, :]
    N = @view states[6, :]

    pnamed = namedparams(model, params)
    mobility = @view model.movement_range_data[:, Int.(saveat) .+ 1]
    proximity = @view model.social_proximity_data[:, Int.(saveat) .+ 1]
    β_ann_input = [
        (collect(saveat) ./ model.time_scale)'
        (S ./ N)'
        (E ./ N)'
        (I ./ N)'
        mobility
        proximity
    ]

    βt = vec(model.β_ann(β_ann_input, pnamed.θ1))
    Re = βt ./ pnamed.γ
    return Re
end

"""
    SEIRD!(du, u, p, t)

The default SEIRD! dynamics function used for OrdinaryDiffEq.jl

# Arguments

+ `du`: the derivative of state `u` at time step `t` that has to be calculated
+ `u`: the system's states at time `t`
+ `p`: the system's parameters
+ `t`: the current time steps

# Model states

The system contains 6 compartments S, E, I, R, D, and N. The states and derivatives
of the compartments can be access using index 1..7 with the same order as listed.

+ **S**: Compartment for susceptible individuals
+ **E**: Compartment for exposed individuals
+ **I**: Compartment for infectious individuals
+ **R**: Compartment for recovered individuals
+ **D**: Compartment for total deaths
+ **N**: Compartment for the effective population

# Model parameters

The system accepts 4 parameters β, γ, λ, and α. The parameters can be access using index
1..4 with the same order as listed.

+ **β**: The contact rate
+ **γ**: Inverse of the mean incubation period
+ **λ**: Inverse of the mean infectious period
+ **α**: The fatality rate
"""
function SEIRD!(du, u, p, t)
    @inbounds begin
        S, E, I, _, _, N, C, _ = u
        β, γ, λ, α = p
        du[1] = -β * S * I / N
        du[2] = β * S * I / N - γ * E
        du[3] = γ * E - λ * I
        du[4] = (1 - α) * λ * I
        du[5] = α * λ * I
        du[6] = -α * λ * I
        du[7] = -C + γ * E
        du[8] = γ * E
    end
    return nothing
end

"""
    default_solve(model, u0, params, tspan, saveat)

Solve the augmented SEIRD model with sensible settings

# Arguments

* `model`: object of the augmented SEIRD model
* `u0`: the system initial conditions
* `params`: the system parameters
* `tspan`: the integration time span
* `tsteps`: the timesteps to be saved
"""
function default_solve(model, u0, params, tspan, saveat)
    return solve(
        ODEProblem(model, u0, tspan, params),
        Tsit5();
        saveat=saveat,
        solver=InterpolatingAdjoint(; autojacvec=ReverseDiffVJP(true)),
        abstol=1e-6,
        reltol=1e-6,
    )
end

"""
    (model::SEIRDBaseline)(du, u, p, t)

The augmented SEIRD! dynamics that integrate a neural network for estimating
the contact rate

# Arguments

+ `du`: the derivative of state `u` at time step `t` that has to be calculated
+ `u`: the system's states at time `t`
+ `p`: the system's parameters
+ `t`: the current time steps
"""
function (model::SEIRDBaseline)(du, u, p, t)
    @inbounds begin
        # states and params
        S, E, I, R, D, N, _, _ = u
        pnamed = namedparams(model, p)
        # infection rate depends on time, susceptible, and infected
        β = first(
            model.β_ann(SVector{4}(t / model.time_scale, S / N, E / N, I / N), pnamed.θ1)
        )
        α = first(
            model.α_ann(SVector{4}(t / model.time_scale, I / N, R / N, D / N), pnamed.θ2)
        )
        SEIRD!(du, u, SVector{4}(β, pnamed.γ, pnamed.λ, α), t)
    end
    return nothing
end

"""
    (model::SEIRDFbMobility1)(du, u, p, t)

The augmented SEIRD! dynamics that integrate a neural network for estimating
the contact rate. Facebook's Movement Range Dataset is used to inform the model
to improve predicting performance

# Arguments

+ `du`: the derivative of state `u` at time step `t` that has to be calculated
+ `u`: the system's states at time `t`
+ `p`: the system's parameters
+ `t`: the current time steps
"""
function (model::SEIRDFbMobility1)(du, u, p, t)
    @inbounds begin
        time_idx = Int(floor(t + 1))
        # states and params
        S, E, I, R, D, N, _, _ = u
        pnamed = namedparams(model, p)
        # infection rate depends on time, susceptible, and infected
        β = first(
            model.β_ann(
                SVector{6}(
                    t / model.time_scale,
                    S / N,
                    E / N,
                    I / N,
                    model.movement_range_data[1, time_idx],
                    model.movement_range_data[2, time_idx],
                ),
                pnamed.θ1,
            ),
        )
        α = first(
            model.α_ann(SVector{4}(t / model.time_scale, I / N, R / N, D / N), pnamed.θ2)
        )
        SEIRD!(du, u, SVector{4}(β, pnamed.γ, pnamed.λ, α), t)
    end
    return nothing
end

"""
    (model::SEIRDFbMobility2)(du, u, p, t)

The augmented SEIRD! dynamics that integrate a neural network for estimating
the contact rate. Facebook's Movement Range Dataset and the Social Proximity
to Cases Index are used to inform the model to improve predicting performance.
The β and α parameters are transformed with sigmoid to prevent the model from
over shooting their values.

# Arguments

+ `du`: the derivative of state `u` at time step `t` that has to be calculated
+ `u`: the system's states at time `t`
+ `p`: the system's parameters
+ `t`: the current time steps
"""
function (model::SEIRDFbMobility2)(du, u, p, t)
    @inbounds begin
        time_idx = Int(floor(t + 1))
        # states and params
        S, E, I, R, D, N, _, _ = u
        pnamed = namedparams(model, p)
        # infection rate depends on time, susceptible, and infected
        β = first(
            model.β_ann(
                SVector{7}(
                    t / model.time_scale,
                    S / N,
                    E / N,
                    I / N,
                    model.movement_range_data[1, time_idx],
                    model.movement_range_data[2, time_idx],
                    model.social_proximity_data[1, time_idx],
                ),
                pnamed.θ1,
            ),
        )
        α = first(
            model.α_ann(SVector{4}(t / model.time_scale, I / N, R / N, D / N), pnamed.θ2)
        )
        SEIRD!(du, u, SVector{4}(β, pnamed.γ, pnamed.λ, α), t)
    end
    return nothing
end
