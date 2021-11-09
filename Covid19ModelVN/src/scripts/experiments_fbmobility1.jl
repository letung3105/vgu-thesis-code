include("experiments.jl")

using OrdinaryDiffEq, DiffEqFlux, CairoMakie

SEIRDFbMobility1Hyperparams = @NamedTuple begin
    ζ::Float64
    γ0::Float64
    λ0::Float64
    α0::Float64
    γ_bounds::Tuple{Float64,Float64}
    λ_bounds::Tuple{Float64,Float64}
    α_bounds::Tuple{Float64,Float64}
    train_range::Day
    forecast_range::Day
end

function setup_fbmobility1(loc::AbstractString, hyperparams::SEIRDFbMobility1Hyperparams)
    # get data for model
    train_dataset, test_dataset, first_date, last_date =
        experiment_covid19_data(loc, hyperparams.train_range, hyperparams.forecast_range)
    @assert size(train_dataset.data, 2) == Dates.value(hyperparams.train_range)
    @assert size(test_dataset.data, 2) == Dates.value(hyperparams.forecast_range)

    movement_range_data = experiment_movement_range(loc, first_date, last_date)
    @assert size(movement_range_data, 2) ==
            Dates.value(hyperparams.train_range) + Dates.value(hyperparams.forecast_range)

    # initialize the model
    model = SEIRDFbMobility1(movement_range_data)
    # augmented dynamic
    dudt(du, u, p, t) = model(
        du,
        u,
        p,
        t;
        γ = boxconst(p[1], hyperparams.γ_bounds),
        λ = boxconst(p[2], hyperparams.λ_bounds),
        α = boxconst(p[3], hyperparams.α_bounds),
    )
    # get the initial states and available observations depending on the model type
    # and the considered location
    u0, vars, labels = experiment_SEIRD_initial_states(loc, train_dataset.data[:, 1])
    # create a prediction model and loss function
    prob = ODEProblem(dudt, u0, train_dataset.tspan)
    predictor = Predictor(prob, vars)
    loss = experiment_loss(predictor, train_dataset, hyperparams.ζ)
    # get initial parameters
    p0 = [
        boxconst_inv(hyperparams.γ0, hyperparams.γ_bounds)
        boxconst_inv(hyperparams.λ0, hyperparams.λ_bounds)
        boxconst_inv(hyperparams.α0, hyperparams.α_bounds)
        DiffEqFlux.initial_params(model.β_ann)
    ]
    # function for getting the effective reproduction number
    ℜe_augmented = function (minimizer)
        γ = boxconst(minimizer[1], hyperparams.γ_bounds)
        ℜe1 = ℜe(model, prob, minimizer, train_dataset.tspan, train_dataset.tsteps; γ)
        ℜe2 = ℜe(model, prob, minimizer, test_dataset.tspan, test_dataset.tsteps; γ)
        return [ℜe1; ℜe2]
    end
    return predictor, loss, p0, train_dataset, test_dataset, labels, ℜe_augmented
end

let
    savedir = "snapshots/default"
    hyperparams = (
        ζ = 0.01,
        γ0 = 1 / 3,
        λ0 = 1 / 14,
        α0 = 0.025,
        γ_bounds = (1 / 5, 1 / 2),
        λ_bounds = (1 / 21, 1 / 7),
        α_bounds = (0.0, 0.06),
        train_range = Day(32),
        forecast_range = Day(28),
    )
    configs = TrainConfig[
        TrainConfig("500ADAM", ADAM(), 500),
        TrainConfig("500LBFGS", LBFGS(), 500),
    ]

    for loc ∈ [
        Covid19ModelVN.LOC_CODE_VIETNAM
        Covid19ModelVN.LOC_CODE_UNITED_STATES
        collect(keys(Covid19ModelVN.LOC_NAMES_VN))
        collect(keys(Covid19ModelVN.LOC_NAMES_US))
    ]
        timestamp = Dates.format(now(), "yyyymmddHHMMSS")
        plt1, plt2, df_errors = experiment_run(
            "$timestamp.fbmobility1.$loc",
            configs,
            () -> setup_fbmobility1(loc, hyperparams),
            snapshots_dir = joinpath(savedir, loc),
        )
        display(plt1)
        display(plt2)
        display(df_errors)
    end
end
