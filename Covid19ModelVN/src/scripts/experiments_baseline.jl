include("experiments.jl")

using OrdinaryDiffEq, DiffEqFlux, CairoMakie

function setup_baseline(
    loc::AbstractString,
    γ_bounds::Tuple{<:Real,<:Real},
    λ_bounds::Tuple{<:Real,<:Real},
    α_bounds::Tuple{<:Real,<:Real};
    train_range::Day = Day(32),
    forecast_range::Day = Day(28),
)
    train_dataset, test_dataset = experiment_covid19_data(loc, train_range, forecast_range)
    @assert size(train_dataset.data, 2) == Dates.value(train_range)
    @assert size(test_dataset.data, 2) == Dates.value(forecast_range)

    # build the model
    model! = SEIRDBaseline()
    # get the initial states and available observations depending on the model type
    # and the considered location
    u0, vars, labels = experiment_SEIRD_initial_states(loc, train_dataset.data[:, 1])
    # augmented dynamic
    dudt!(du, u, p, t) = model!(
        du,
        u,
        p,
        t;
        γ = boxconst(p[1], γ_bounds),
        λ = boxconst(p[2], λ_bounds),
        α = boxconst(p[3], α_bounds),
    )
    # define problem and train model
    prob = ODEProblem(dudt!, u0, train_dataset.tspan)
    predictor = Predictor(prob, vars)
    loss = experiment_loss(predictor, train_dataset, 0.01)
    return model!, prob, predictor, loss, train_dataset, test_dataset, labels
end

function experiment_baseline(
    loc::AbstractString;
    γ_bounds::Tuple{<:Real,<:Real} = (1 / 5, 1 / 2),
    λ_bounds::Tuple{<:Real,<:Real} = (1 / 21, 1 / 14),
    α_bounds::Tuple{<:Real,<:Real} = (0.0, 0.06),
    name::AbstractString = "baseline",
    savedir::AbstractString,
)
    snapshots_dir = joinpath(savedir, loc)
    uuid = Dates.format(now(), "yyyymmddHHMMSS")
    sessname = "$uuid.$name.$loc"
    # get model and data
    model!, prob, predictor, loss, train_dataset, test_dataset, labels =
        setup_baseline(loc, γ_bounds, λ_bounds, α_bounds)
    # get initial parameters
    p0 = [
        logit((1 / 3 - γ_bounds[1]) / (γ_bounds[2] - γ_bounds[1]))
        logit((1 / 14 - λ_bounds[1]) / (λ_bounds[2] - λ_bounds[1]))
        logit((0.025 - α_bounds[1]) / (α_bounds[2] - α_bounds[1]))
        DiffEqFlux.initial_params(model!.β_ann)
    ]
    # parameters estimation
    minimizers = train_model(
        loss,
        p0,
        TrainSession[
            TrainSession("$sessname.adam", ADAM(0.01), 500, 100),
            TrainSession("$sessname.bfgs", BFGS(initial_stepnorm = 0.01), 100, 100),
        ],
        snapshots_dir = snapshots_dir,
    )
    # evaluation with estimated parameters
    minimizer = first(minimizers)
    return experiment_evaluate(
        sessname,
        model!,
        prob,
        predictor,
        minimizer,
        train_dataset,
        test_dataset,
        labels,
        snapshots_dir = snapshots_dir,
    )
end

let
    γ_bounds::Tuple{<:Real,<:Real} = (1 / 5, 1 / 2)
    λ_bounds::Tuple{<:Real,<:Real} = (1 / 21, 1 / 14)
    α_bounds::Tuple{<:Real,<:Real} = (0.0, 0.06)
    loc = "hcm"
    # get model and data
    model!, prob, predictor, loss, train_dataset, test_dataset, labels =
        setup_baseline(loc, γ_bounds, λ_bounds, α_bounds)
    # get initial parameters
    p0 = [
        logit((1 / 3 - γ_bounds[1]) / (γ_bounds[2] - γ_bounds[1]))
        logit((1 / 14 - λ_bounds[1]) / (λ_bounds[2] - λ_bounds[1]))
        logit((0.025 - α_bounds[1]) / (α_bounds[2] - α_bounds[1]))
        DiffEqFlux.initial_params(model!.β_ann)
    ]
    Zygote.gradient(loss, p0)
end

for loc ∈ [
    Covid19ModelVN.LOC_CODE_VIETNAM
    Covid19ModelVN.LOC_CODE_UNITED_STATES
    collect(keys(Covid19ModelVN.LOC_NAMES_VN))
    collect(keys(Covid19ModelVN.LOC_NAMES_US))
]
    experiment_baseline(loc, savedir = "snapshots")
end
