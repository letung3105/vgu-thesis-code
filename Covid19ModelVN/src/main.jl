# Activate the environment for running the script
if isfile("Project.toml") && isfile("Manifest.toml")
    import Pkg
    Pkg.activate(".")
end

using Covid19ModelVN.Cmds, Covid19ModelVN.Models, Covid19ModelVN.Helpers

const DEFAULT_DATASETS_DIR = "datasets"
const DEFAULT_SNAPSHOTS_DIR = "snapshots"

function train_and_evaluate_experiment_preset_vietnam(
    exp_name::AbstractString;
    fb_movement_range_fpath::AbstractString,
    datasets_dir::AbstractString=DEFAULT_DATASETS_DIR,
    snapshots_dir::AbstractString=DEFAULT_SNAPSHOTS_DIR,
)
    df_cases_timeseries, df_fb_movement_range = setup_experiment_data_vietnam(
        datasets_dir,
        fb_movement_range_fpath,
    )
    model, train_dataset, test_dataset = setup_experiment_preset_vietnam(
        exp_name,
        df_cases_timeseries,
        df_fb_movement_range,
    )
    predict_fn = Predictor(model.problem)
    train_loss_fn = Loss(sse, predict_fn, train_dataset, 3:6)
    test_loss_fn = Loss(sse, predict_fn, test_dataset, 3:6)
    p0 = get_model_initial_params(model)

    @info train_loss_fn(p0)
    @info test_loss_fn(p0)

    @info "Start training"
    train_model_default_2steps(
        exp_name, train_loss_fn, test_loss_fn, p0,
        snapshots_dir=snapshots_dir,
        adam_maxiters=500,
        bfgs_maxiters=100,
    )

    @info "Ploting evaluations"
    evaluate_model_default(
        exp_name, predict_fn, train_dataset, test_dataset,
        snapshots_dir=snapshots_dir,
    )

    return nothing
end

train_and_evaluate_experiment_preset_vietnam(
    "baseline.default.vietnam",
    fb_movement_range_fpath=joinpath(
        DEFAULT_DATASETS_DIR,
        "facebook",
        "movement-range-data-2021-10-09",
        "movement-range-2021-10-09.txt",
    ),
)