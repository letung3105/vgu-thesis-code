include("include/experiments.jl")

using Hyperopt
using Statistics
using ProgressMeter

let
    savedir = "snapshots/fbmobility4/hyperopt"
    locations = [
        collect(keys(Covid19ModelVN.LOC_NAMES_VN))
        collect(keys(Covid19ModelVN.LOC_NAMES_US))
    ]

    samples = 30
    progress = Progress(samples, showspeed = true)
    ho = @hyperopt for i in samples, # number of samples
        sampler in Hyperband(R = 50, η = 3, inner = RandomSampler()),
        adam_lr in [exp10.(-4:-2); exp10.(-4:-2) .* 5],
        bfgs_initial_stepnorm in exp10.(-3:-2),
        L2_λ in exp10.(-5:-2),
        ζ in [-exp10.(-3:-2); 0.0; exp10.(-3:-2)],
        batchsize in [0, 8, 16]

        minimizers, final_losses = experiment_run(
            "fbmobility4",
            setup_fbmobility4,
            locations,
            (
                L2_λ = L2_λ,
                ζ = 0.0,
                γ0 = 1 / 3,
                λ0 = 1 / 14,
                β_bounds = (0.0, 1.336),
                γ_bounds = (1 / 5, 1 / 2),
                λ_bounds = (1 / 21, 1 / 7),
                α_bounds = (0.0, 0.06),
                train_range = Day(32),
                forecast_range = Day(28),
                social_proximity_lag = Day(14),
                ma7 = true,
                batchsize = batchsize,
            ),
            TrainConfig[
                TrainConfig("ADAM", ADAM(adam_lr), Int(10000)),
                TrainConfig(
                    "BFGS",
                    BFGS(initial_stepnorm = bfgs_initial_stepnorm),
                    Int(500),
                ),
            ];
            forecast_horizons = [7, 14, 21, 28],
            multithreading = true,
            savedir,
        )
        next!(progress)
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
