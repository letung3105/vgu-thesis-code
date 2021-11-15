include("experiments.jl")

using Hyperopt
using Statistics

let
    savedir = "snapshots/fbmobility4/hyperopt"
    locations = [
        collect(keys(Covid19ModelVN.LOC_NAMES_VN))
        collect(keys(Covid19ModelVN.LOC_NAMES_US))
    ]

    ho = @thyperopt for i ∈ 100, # number of samples
        sampler ∈ Hyperband(R = 50, η = 3, inner = RandomSampler()),
        ζ ∈ [exp10.(-3:-1); exp10.(-3:-1) .* 5],
        adam_lr ∈ [exp10.(-4:-2); exp10.(-4:-2) .* 5],
        adam_maxiters ∈ exp10(3) .* (2:4:10),
        bfgs_initial_stepnorm ∈ exp10.(-3:-2),
        bfgs_maxiters ∈ exp10(2) .* (2:4:10)

        minimizers, final_losses = experiment_run(
            "fbmobility4",
            setup_fbmobility4,
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
        mean(final_losses), minimizers
    end
    show(ho)

    fig = Figure(resolution = (600, 400 * length(ho.params)))
    for (paramid, param) in enumerate(ho.params)
        ax = Axis(fig[paramid, 1], xlabel = string(param), ylabel = "Loss")
        scatter!(ax, map(h -> h[paramid], ho.history), Float64.(ho.results))
    end
    display(fig)
end

let
    savedir = "snapshots/fbmobility4/testalllocations"
    # locations = [
    #     collect(keys(Covid19ModelVN.LOC_NAMES_VN))
    #     collect(keys(Covid19ModelVN.LOC_NAMES_US))
    # ]
    locations = ["losangeles_ca"]
    _, final_losses = experiment_run(
        "fbmobility4",
        setup_fbmobility4,
        locations,
        (
            ζ = 0.001,
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
            TrainConfig("ADAM", ADAM(0.001), Int(10000)),
            TrainConfig("BFGS", BFGS(initial_stepnorm = 0.01), Int(500)),
        ];
        savedir,
        show_progress = true
    )
end
