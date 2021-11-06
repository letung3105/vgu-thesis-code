# Activate the environment for running the script
if isfile("Project.toml") && isfile("Manifest.toml")
    import Pkg
    Pkg.activate(".")
end

include("experiments.jl")

const SNAPSHOTS_DIR = "snapshots"

function run01(experiment_name::AbstractString)
    uuid = Dates.format(now(), "yyyymmddHHMMSS")
    hyperparams = Hyperparameters(
        0.1,
        # 2-stage training procedure without constraints
        TrainSession[
            TrainSession(
                "$uuid.$experiment_name.adam",
                ADAM(0.01),
                500, # maxiters
                100, # number of snapshots that are taken at evenly spaced intervals
            ),
            TrainSession(
                "$uuid.$experiment_name.bfgs",
                BFGS(initial_stepnorm = 0.01),
                100, # maxiters
                100, # number of snapshots that are taken at evenly spaced intervals
            ),
        ],
    )

    experiment_train_and_eval(
        uuid,
        experiment_name,
        hyperparams,
        snapshots_dir = joinpath(
            SNAPSHOTS_DIR,
            "500adam-learningrate0.01-100bfgs-initstepnorm0.01",
            loc,
        ),
    )
end

batchrun01(experiment_names) =
    for experiment_name ∈ experiment_names
        run01(experiment_name)
    end

function run02(experiment_name::AbstractString)
    uuid = Dates.format(now(), "yyyymmddHHMMSS")
    hyperparams = Hyperparameters(
        0.1,
        # single-stage training procedure with constraints on parameters' lower bounds
        # and upper bounds
        TrainSession[TrainSession(
            "$uuid.$experiment_name.bfgs",
            BFGS(initial_stepnorm = 0.01),
            100,
            100,
        )],
        true, # use box constraints
        (1 / 5, 1 / 2), # mean incubation period is in the range of 2 to 5 days
        (1 / 21, 1 / 7), # mean infectious period is in the range of 7 to 21 days
        (0.01, 0.06), # fatality rate is in the range of 1% to 6%
    )

    _, loc = rsplit(experiment_name, ".", limit = 2)
    experiment_train_and_eval(
        uuid,
        experiment_name,
        hyperparams,
        snapshots_dir = joinpath(
            SNAPSHOTS_DIR,
            "100bfgs-initstepnorm0.01-boxconstrained",
            loc,
        ),
    )
end

batchrun02(experiment_names) =
    for experiment_name ∈ experiment_names
        run02(experiment_name)
    end

function run03(experiment_name::AbstractString)
    uuid = Dates.format(now(), "yyyymmddHHMMSS")
    hyperparams = Hyperparameters(
        0.1,
        # single-stage training procedure with constraints on parameters' lower bounds
        # and upper bounds
        TrainSession[TrainSession(
            "$uuid.$experiment_name.bfgs",
            BFGS(initial_stepnorm = 0.001),
            200,
            100,
        )],
        true, # use box constraints
        (1 / 5, 1 / 2), # mean incubation period is in the range of 2 to 5 days
        (1 / 21, 1 / 7), # mean infectious period is in the range of 7 to 21 days
        (0.01, 0.06), # fatality rate is in the range of 1% to 6%
    )

    experiment_train_and_eval(
        uuid,
        experiment_name,
        hyperparams,
        snapshots_dir = joinpath(
            SNAPSHOTS_DIR,
            "BFGS-maxiters-200-initstepnorm-0.001-boxconstrained",
            loc,
        ),
    )
end

batchrun03(experiment_names) =
    for experiment_name ∈ experiment_names
        run03(experiment_name)
    end

function run04(experiment_name::AbstractString)
    uuid = Dates.format(now(), "yyyymmddHHMMSS")
    hyperparams = Hyperparameters(
        0,
        # single-stage training procedure with constraints on parameters' lower bounds
        # and upper bounds
        TrainSession[TrainSession(
            "$uuid.$experiment_name.bfgs",
            BFGS(initial_stepnorm = 0.01),
            100,
            100,
        )],
        true, # use box constraints
        (1 / 5, 1 / 2), # mean incubation period is in the range of 2 to 5 days
        (1 / 21, 1 / 7), # mean infectious period is in the range of 7 to 21 days
        (0.01, 0.06), # fatality rate is in the range of 1% to 6%
    )

    _, loc = rsplit(experiment_name, ".", limit = 2)
    experiment_train_and_eval(
        uuid,
        experiment_name,
        hyperparams,
        snapshots_dir = joinpath(
            SNAPSHOTS_DIR,
            "100bfgs-initstepnorm0.01-boxconstrained-no-weights",
            loc,
        ),
    )
end

batchrun04(experiment_names) =
    for experiment_name ∈ experiment_names
        run04(experiment_name)
    end


batchrun04([FBMOBILITY1_EXPERIMENTS; FBMOBILITY2_EXPERIMENTS])
