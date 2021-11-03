# Activate the environment for running the script
if isfile("Project.toml") && isfile("Manifest.toml")
    import Pkg
    Pkg.activate(".")
end

include("experiments.jl")

const SNAPSHOTS_DIR = "snapshots"

struct Hyperparameters
    ζ::Float64
    adam_maxiters::Int
    adam_learning_rate::Float64
    bfgs_maxiters::Int
    bfgs_initial_stepnorm::Float64
end

function run_experiments(
    names::AbstractVector{<:AbstractString},
    hyperparams::Hyperparameters;
    snapshots_dir::AbstractString = SNAPSHOTS_DIR,
)
    for name ∈ names
        model_type, location_code = rsplit(name, ".", limit = 2)
        timestamp = Dates.format(now(), "yyyymmddHHMMSS")
        train_sessions = [
            TrainSession(
                "$timestamp.$model_type.adam",
                ADAM(hyperparams.adam_learning_rate),
                hyperparams.adam_maxiters,
                100,
            ),
            TrainSession(
                "$timestamp.$model_type.bfgs",
                BFGS(initial_stepnorm = hyperparams.bfgs_initial_stepnorm),
                hyperparams.bfgs_maxiters,
                100,
            ),
        ]
        eval_config = get_experiment_eval_config(location_code)

        model, train_dataset, test_dataset = setup_experiment(model_type, location_code)
        predict_fn = Predictor(model)
        p0 = Covid19ModelVN.initial_params(model)

        weights = exp.(collect(train_dataset.tsteps) .* hyperparams.ζ)
        loss(ŷ, y) = sum(mean((log.(ŷ .+ 1) .- log.(y .+ 1)) .^ 2 .* weights', dims = 2))
        train_loss_fn = Loss(loss, predict_fn, train_dataset, eval_config.vars)
        @info "Initial training loss: $(train_loss_fn(p0))"

        experiment_dir = joinpath(snapshots_dir, location_code)
        if !isdir(experiment_dir)
            mkpath(experiment_dir)
        end
        @info "Snapshot directory '$experiment_dir'"

        @info "Start training"
        sessions_params =
            train_model(train_sessions, train_loss_fn, p0, snapshots_dir = experiment_dir)

        @info "Ploting evaluations"
        for (sess, params) in zip(train_sessions, sessions_params)
            eval_res =
                evaluate_model(eval_config, params, predict_fn, train_dataset, test_dataset)

            csv_fpath = joinpath(experiment_dir, "$(sess.name).evaluate.csv")
            if !isfile(csv_fpath)
                CSV.write(csv_fpath, eval_res.df_errors)
            end

            fig_fpath = joinpath(experiment_dir, "$(sess.name).evaluate.png")
            if !isfile(fig_fpath)
                savefig(eval_res.fig_forecasts, fig_fpath)
            end
        end
    end
end

let
    hyperparams = [
        Hyperparameters(0.1, 500, 0.01, 1000, 0.01),
        Hyperparameters(0.2, 500, 0.01, 1000, 0.01),
        Hyperparameters(0.1, 500, 0.001, 1000, 0.001),
        Hyperparameters(0.2, 500, 0.001, 1000, 0.001),
    ]
    for (i, hyper) in enumerate(hyperparams)
        run_experiments(
            vec([
                "baseline.default.$loc" for loc ∈ [
                    "vietnam",
                    "unitedstates",
                    "hcm",
                    "binhduong",
                    "dongnai",
                    "longan",
                    "losangeles_ca",
                    "cook_il",
                    "harris_tx",
                    "maricopa_az",
                ]
            ]),
            hyper,
            snapshots_dir = joinpath("SNAPSHOTS_DIR", "batch$i"),
        )
    end
end
