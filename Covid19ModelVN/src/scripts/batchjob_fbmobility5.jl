include("cmd.jl")

using Hyperopt
using Statistics

ho = let
    savedir = "snapshots/fbmobility5/hyperopt"
    locations = [
        collect(keys(Covid19ModelVN.LOC_NAMES_VN))
        collect(keys(Covid19ModelVN.LOC_NAMES_US))
    ]
    @thyperopt for i ∈ 50, # number of samples
        sampler ∈ RandomSampler(), # choose an parameters sampler
        ζ ∈ [exp10.(-3:-1); exp10.(-3:-1) .* 5],
        adam_lr ∈ [exp10.(-4:-2); exp10.(-4:-2) .* 5],
        adam_maxiters ∈ exp10.(2:4) .* 5,
        bfgs_initial_stepnorm ∈ exp10.(-3:-2),
        bfgs_maxiters ∈ exp10.(2:4)

        _, final_losses = experiment_run(
            "fbmobility5",
            setup_fbmobility5,
            locations,
            (
                ζ = ζ,
                γ0 = 1 / 3,
                λ0 = 1 / 14,
                β_bounds = (0.0, 1.336),
                γ_bounds = (1 / 5, 1 / 2),
                λ_bounds = (1 / 21, 1 / 7),
                α_bounds = (0.0, 0.06),
                train_range = Day(32),
                forecast_range = Day(28),
                social_proximity_lag = Day(14),
            ),
            TrainConfig[
                TrainConfig("ADAM", ADAM(adam_lr), Int(adam_maxiters)),
                TrainConfig(
                    "BFGS",
                    BFGS(initial_stepnorm = bfgs_initial_stepnorm),
                    Int(bfgs_maxiters),
                ),
            ];
            savedir,
        )
        mean(final_losses)
    end
end
