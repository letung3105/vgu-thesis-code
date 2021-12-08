include("include/cmd.jl")

let
    args_model = [
        "--beta_bounds",
        0.2 / 4,
        6.68 / 4,
        "--gamma0",
        1 / 4,
        "--gamma_bounds",
        1 / 4,
        1 / 4,
        "--lambda0",
        1 / 14,
        "--lambda_bounds",
        1 / 14,
        1 / 14,
        "--alpha_bounds",
        0.005,
        0.05,
        "--train_days=60",
        "--movement_range_lag_days=0",
        "--social_proximity_lag_days=0",
        "--loss_type=sse",
        "--loss_regularization=0.0001",
    ]

    args_train = [
        "train_whole_trajectory_two_stages",
        "--lr=0.01",
        "--lr_limit=0.0001",
        "--lr_decay_rate=0.5",
        "--lr_decay_step=1000",
        "--maxiters_first=10000",
        "--maxiters_second=1000",
        "--minibatching=15",
    ]

    runcmd(
        string.([
            args_model...,
            "--locations",
            "cook_il",
            "--savedir=testsnapshots/testhyperparams",
            "--show_progress",
            "fbmobility2",
            args_train...,
        ]),
    )
end

runcmd(
    string.([
        args_model...,
        "--locations",
        union(keys(Covid19ModelVN.LOC_NAMES_US), keys(Covid19ModelVN.LOC_NAMES_VN))...,
        "--savedir=testsnapshots/batchrun01",
        "--multithreading",
        "fbmobility2",
        args_train...,
    ]),
)
