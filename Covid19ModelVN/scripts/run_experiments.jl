# Activate the environment for running the script
if isfile("Project.toml") && isfile("Manifest.toml")
    import Pkg
    Pkg.activate(".")
end

include("experiments.jl")

const SNAPSHOTS_DIR = "snapshots"

for model ∈ ["baseline.default", "fbmobility1.default", "fbmobility2.default"],
    loc ∈ [keys(Covid19ModelVN.LOC_NAMES_US)...; keys(Covid19ModelVN.LOC_NAMES_VN)...]

    experiment_train_and_eval(
        "$model.$loc",
        Hyperparameters(0.1, 500, 0.01, 100, 0.01),
        snapshots_dir = joinpath(SNAPSHOTS_DIR, loc),
    )
end
