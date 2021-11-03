# Activate the environment for running the script
if isfile("Project.toml") && isfile("Manifest.toml")
    import Pkg
    Pkg.activate(".")
end

include("experiments.jl")

const SNAPSHOTS_DIR = "snapshots"

function run_experiments(
    names::AbstractVector{<:AbstractString},
    hyperparams::Hyperparameters;
    snapshots_dir::AbstractString = SNAPSHOTS_DIR,
)
    for name ∈ names
        model_type, location_code = rsplit(name, ".", limit = 2)
        model, train_dataset, test_dataset = setup_experiment(model_type, location_code)
        experiment_train_and_eval(
            model,
            model_type,
            location_code,
            hyperparams,
            train_dataset,
            test_dataset,
            snapshots_dir = snapshots_dir,
        )
    end
end

let
    run_experiments(
        vec([
            "fbmobility2.default.$loc" for loc ∈ [
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
        Hyperparameters(0.1, 500, 0.01, 1000, 0.01),
        snapshots_dir = joinpath(SNAPSHOTS_DIR, "testrun01"),
    )
end
