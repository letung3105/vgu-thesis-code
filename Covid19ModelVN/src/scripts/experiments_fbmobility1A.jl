include("experiments.jl")

experiment_run(
    "fbmobility1A",
    setup_fbmobility1,
    [
        Covid19ModelVN.LOC_CODE_VIETNAM
        Covid19ModelVN.LOC_CODE_UNITED_STATES
        collect(keys(Covid19ModelVN.LOC_NAMES_VN))
        collect(keys(Covid19ModelVN.LOC_NAMES_US))
    ],
    (
        ζ = 0.005,
        γ0 = 1 / 3,
        λ0 = 1 / 14,
        α0 = 0.025,
        γ_bounds = (1 / 5, 1 / 2),
        λ_bounds = (1 / 21, 1 / 7),
        α_bounds = (0.0, 0.06),
        train_range = Day(32),
        forecast_range = Day(28),
    ),
    TrainConfig[
        TrainConfig("ADAM", ADAM(0.01), 500),
        TrainConfig("BFGS", BFGS(initial_stepnorm = 0.01), 500),
    ],
    savedir = "snapshots/default",
)
