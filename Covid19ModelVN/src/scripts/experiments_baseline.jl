include("experiments.jl")

SEIRDBaselineHyperparams = @NamedTuple begin
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

function setup_baseline(loc::AbstractString, hyperparams::SEIRDBaselineHyperparams)
    # get data for model
    train_dataset, test_dataset =
        experiment_covid19_data(loc, hyperparams.train_range, hyperparams.forecast_range)
    @assert size(train_dataset.data, 2) == Dates.value(hyperparams.train_range)
    @assert size(test_dataset.data, 2) == Dates.value(hyperparams.forecast_range)

    # initialize the model
    model = SEIRDBaseline(hyperparams.γ_bounds, hyperparams.λ_bounds, hyperparams.α_bounds)
    # get the initial states and available observations depending on the model type
    # and the considered location
    u0, vars, labels = experiment_SEIRD_initial_states(loc, train_dataset.data[:, 1])
    p0 = initparams(model, hyperparams.γ0, hyperparams.λ0, hyperparams.α0)
    lossfn = experiment_loss(train_dataset.tsteps, hyperparams.ζ)
    return model, u0, p0, lossfn, train_dataset, test_dataset, vars, labels
end

experiment_run(
    "baseline",
    setup_baseline,
    [
        Covid19ModelVN.LOC_CODE_VIETNAM
        Covid19ModelVN.LOC_CODE_UNITED_STATES
        collect(keys(Covid19ModelVN.LOC_NAMES_VN))
        collect(keys(Covid19ModelVN.LOC_NAMES_US))
    ],
    (
        ζ = 0.01,
        γ0 = 1 / 3,
        λ0 = 1 / 14,
        α0 = 0.025,
        γ_bounds = (1 / 5, 1 / 2),
        λ_bounds = (1 / 21, 1 / 7),
        α_bounds = (0.0, 0.06),
        train_range = Day(32),
        forecast_range = Day(28),
    ),
    TrainConfig[TrainConfig("500ADAM", ADAM(), 500), TrainConfig("500LBFGS", LBFGS(), 500)],
    savedir = "snapshots/default",
)
