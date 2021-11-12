include("experiments.jl")

SEIRDFbMobility3Hyperparams = @NamedTuple begin
    ζ::Float64
    γ0::Float64
    λ0::Float64
    γ_bounds::Tuple{Float64,Float64}
    λ_bounds::Tuple{Float64,Float64}
    α_bounds::Tuple{Float64,Float64}
    train_range::Day
    forecast_range::Day
    social_proximity_lag::Day
end

function setup_fbmobility3(loc::AbstractString, hyperparams::SEIRDFbMobility3Hyperparams)
    # get data for model
    train_dataset, test_dataset, first_date, last_date =
        experiment_covid19_data(loc, hyperparams.train_range, hyperparams.forecast_range)
    @assert size(train_dataset.data, 2) == Dates.value(hyperparams.train_range)
    @assert size(test_dataset.data, 2) == Dates.value(hyperparams.forecast_range)

    movement_range_data = experiment_movement_range(loc, first_date, last_date)
    @assert size(movement_range_data, 2) ==
            Dates.value(hyperparams.train_range) + Dates.value(hyperparams.forecast_range)

    social_proximity_data = experiment_social_proximity(
        loc,
        first_date - hyperparams.social_proximity_lag,
        last_date - hyperparams.social_proximity_lag,
    )
    @assert size(social_proximity_data, 2) ==
            Dates.value(hyperparams.train_range) + Dates.value(hyperparams.forecast_range)

    # build the model
    model = SEIRDFbMobility3(
        hyperparams.γ_bounds,
        hyperparams.λ_bounds,
        hyperparams.α_bounds,
        movement_range_data,
        social_proximity_data,
    )
    # get the initial states and available observations depending on the model type
    # and the considered location
    u0, vars, labels = experiment_SEIRD_initial_states(loc, train_dataset.data[:, 1])
    p0 = initparams(model, hyperparams.γ0, hyperparams.λ0)
    lossfn = experiment_loss(train_dataset.tsteps, hyperparams.ζ)
    return model, u0, p0, lossfn, train_dataset, test_dataset, vars, labels
end

experiment_run(
    "fbmobility3",
    setup_fbmobility3,
    [
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
        social_proximity_lag = Day(14),
    ),
    TrainConfig[TrainConfig("500ADAM", ADAM(), 500), TrainConfig("500LBFGS", LBFGS(), 500)],
    savedir = "snapshots/default",
)
