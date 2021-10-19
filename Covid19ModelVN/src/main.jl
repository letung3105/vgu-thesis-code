# Activate the environment for running the script
if isfile("Project.toml") && isfile("Manifest.toml")
    import Pkg
    Pkg.activate(".")
end

using Dates, DiffEqFlux, Covid19ModelVN.Cmds, Covid19ModelVN.Models, Covid19ModelVN.Helpers

const DEFAULT_DATASETS_DIR = "datasets"
const DEFAULT_SNAPSHOTS_DIR = "snapshots"

function train_and_evaluate_experiment_preset_vietnam(
    exp_name::AbstractString;
    fb_movement_range_fpath::AbstractString,
    datasets_dir::AbstractString = DEFAULT_DATASETS_DIR,
    snapshots_dir::AbstractString = DEFAULT_SNAPSHOTS_DIR,
)
    model, train_dataset, test_dataset =
        setup_experiment_preset_vietnam(exp_name, datasets_dir, fb_movement_range_fpath)

    predict_fn = Predictor(model.problem)
    train_loss_fn = Loss(mse, predict_fn, train_dataset, 3:6)
    test_loss_fn = Loss(mse, predict_fn, test_dataset, 3:6)
    p0 = get_model_initial_params(model)

    @info train_loss_fn(p0)
    @info test_loss_fn(p0)

    # create containing folder if not exists
    exp_dir = joinpath(snapshots_dir, exp_name)
    if !isdir(exp_dir)
        mkpath(exp_dir)
    end

    timestamp = Dates.format(now(), "yyyymmddHHMMSS")
    sessions = [
        TrainSession("$timestamp.adam", ADAM(1e-2), 1000, exp_dir, exp_dir),
        TrainSession(
            "$timestamp.bfgs",
            BFGS(initial_stepnorm = 1e-2),
            1000,
            exp_dir,
            exp_dir,
        ),
    ]

    @info "Start training"
    train_model(train_loss_fn, test_loss_fn, p0, sessions)

    @info "Ploting evaluations"
    fpaths_params, uuids = lookup_params_snapshots(snapshots_dir, exp_name)
    for (fpath_params, uuid) in zip(fpaths_params, uuids)
        fig_fpath = joinpath(snapshots_dir, exp_name, "$uuid.evaluate.forecasts.mape.png")
        if !isfile(fig_fpath)
            plt = plot_forecasts(
                predict_fn,
                mape,
                train_dataset,
                test_dataset,
                Serialization.deserialize(fpath_params),
                [7, 14, 28],
                3:6,
                1:4,
                ["infective" "recoveries" "deaths" "total cases"],
            )
            savefig(
                plt,
                joinpath(snapshots_dir, exp_name, "$uuid.evaluate.forecasts.mape.png"),
            )
        end
    end

    return nothing
end

experiment_names = [
    "baseline.default.vietnam",
    "fbmobility.default.vietnam",
    "fbmobility.4daydelay.vietnam",
    "fbmobility.ma7movementrange.default.vietnam",
    "fbmobility.ma7movementrange.default.vietnam",
]

for exp_name in experiment_names
    train_and_evaluate_experiment_preset_vietnam(
        exp_name,
        fb_movement_range_fpath = joinpath(
            DEFAULT_DATASETS_DIR,
            "facebook",
            "movement-range-data-2021-10-09",
            "movement-range-2021-10-09.txt",
        ),
    )
end