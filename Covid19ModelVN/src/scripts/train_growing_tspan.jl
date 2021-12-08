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
        "--train_days=28",
        "--movement_range_lag_days=0",
        "--social_proximity_lag_days=0",
        "--loss_type=sse",
        "--loss_regularization=0.0001",
    ]

    args_train = [
        "train_growing_trajectory_two_stages",
        "--lr=0.05",
        "--lr_limit=0.0001",
        "--lr_decay_rate=0.5",
        "--lr_decay_step=1000",
        "--maxiters_initial=1000",
        "--maxiters_growth=1000",
        "--maxiters_second=1000",
        "--tspan_size_initial=7",
        "--tspan_size_growth=7",
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
        # "--loss_regularization=0.001",
    ]

    args_train = [
        "train_growing_trajectory_two_stages",
        "--lr=0.05",
        "--lr_limit=0.0001",
        "--lr_decay_rate=0.5",
        "--lr_decay_step=1000",
        "--maxiters_initial=500",
        "--maxiters_growth=500",
        "--maxiters_second=1000",
        "--tspan_size_initial=6",
        "--tspan_size_growth=6",
    ]

    for model in ["baseline", "fbmobility1"]
        runcmd(
            string.([
                args_model...,
                "--locations",
                Covid19ModelVN.LOC_CODE_VIETNAM,
                Covid19ModelVN.LOC_CODE_UNITED_STATES,
                keys(Covid19ModelVN.LOC_NAMES_US)...,
                keys(Covid19ModelVN.LOC_NAMES_VN)...,
                "--savedir=testsnapshots/batch60days",
                "--multithreading",
                model,
                args_train...,
            ]),
        )
    end

    for model in ["fbmobility2"]
        runcmd(
            string.([
                args_model...,
                "--locations",
                keys(Covid19ModelVN.LOC_NAMES_US)...,
                keys(Covid19ModelVN.LOC_NAMES_VN)...,
                "--savedir=testsnapshots/batch60days",
                "--multithreading",
                model,
                args_train...,
            ]),
        )
    end
end

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
        "--train_days=45",
        "--movement_range_lag_days=0",
        "--social_proximity_lag_days=0",
        "--loss_type=sse",
        # "--loss_regularization=0.001",
    ]

    args_train = [
        "train_growing_trajectory_two_stages",
        "--lr=0.05",
        "--lr_limit=0.0001",
        "--lr_decay_rate=0.5",
        "--lr_decay_step=1000",
        "--maxiters_initial=500",
        "--maxiters_growth=500",
        "--maxiters_second=1000",
        "--tspan_size_initial=5",
        "--tspan_size_growth=5",
    ]

    for model in ["baseline", "fbmobility1"]
        runcmd(
            string.([
                args_model...,
                "--locations",
                Covid19ModelVN.LOC_CODE_VIETNAM,
                Covid19ModelVN.LOC_CODE_UNITED_STATES,
                keys(Covid19ModelVN.LOC_NAMES_US)...,
                keys(Covid19ModelVN.LOC_NAMES_VN)...,
                "--savedir=testsnapshots/batch45days",
                "--multithreading",
                model,
                args_train...,
            ]),
        )
    end

    for model in ["fbmobility2"]
        runcmd(
            string.([
                args_model...,
                "--locations",
                keys(Covid19ModelVN.LOC_NAMES_US)...,
                keys(Covid19ModelVN.LOC_NAMES_VN)...,
                "--savedir=testsnapshots/batch45days",
                "--multithreading",
                model,
                args_train...,
            ]),
        )
    end
end

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
        "--train_days=32",
        "--movement_range_lag_days=0",
        "--social_proximity_lag_days=0",
        "--loss_type=sse",
        # "--loss_regularization=0.001",
    ]

    args_train = [
        "train_growing_trajectory_two_stages",
        "--lr=0.05",
        "--lr_limit=0.0001",
        "--lr_decay_rate=0.5",
        "--lr_decay_step=1000",
        "--maxiters_initial=500",
        "--maxiters_growth=500",
        "--maxiters_second=1000",
        "--tspan_size_initial=4",
        "--tspan_size_growth=4",
    ]

    for model in ["baseline", "fbmobility1"]
        runcmd(
            string.([
                args_model...,
                "--locations",
                Covid19ModelVN.LOC_CODE_VIETNAM,
                Covid19ModelVN.LOC_CODE_UNITED_STATES,
                keys(Covid19ModelVN.LOC_NAMES_US)...,
                keys(Covid19ModelVN.LOC_NAMES_VN)...,
                "--savedir=testsnapshots/batch32days",
                "--multithreading",
                model,
                args_train...,
            ]),
        )
    end

    for model in ["fbmobility2"]
        runcmd(
            string.([
                args_model...,
                "--locations",
                keys(Covid19ModelVN.LOC_NAMES_US)...,
                keys(Covid19ModelVN.LOC_NAMES_VN)...,
                "--savedir=testsnapshots/batch32days",
                "--multithreading",
                model,
                args_train...,
            ]),
        )
    end
end
