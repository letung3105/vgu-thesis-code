include("include/cmd.jl")

# runcmd(
#     string.([
#         "--train_days=60",
#         "--loss_regularization=0.0001",
#         "--loss_time_weighting=-0.001",
#         "--locations=cook_il",
#         "--savedir=testsnapshots/testhyperparams",
#         "--show_progress",
#         "fbmobility2",
#         "train_growing_trajectory",
#         "--lr=0.05",
#         "--lr_limit=0.00001",
#         "--lr_decay_rate=0.3",
#         "--lr_decay_step=1000",
#         "--maxiters_initial=500",
#         "--maxiters_growth=500",
#         "--tspan_size_initial=6",
#         "--tspan_size_growth=6",
#     ]),
# )


let
    args_model = [
        "--train_days=60",
        "--loss_regularization=0.0001",
        "--loss_time_weighting=-0.001",
    ]

    args_train = [
        "train_growing_trajectory_two_stages",
        "--lr=0.05",
        "--lr_limit=0.00001",
        "--lr_decay_rate=0.3",
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
        "--train_days=45",
        "--loss_regularization=0.0001",
        "--loss_time_weighting=-0.001",
    ]

    args_train = [
        "train_growing_trajectory_two_stages",
        "--lr=0.05",
        "--lr_limit=0.00001",
        "--lr_decay_rate=0.3",
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
        "--train_days=32",
        "--loss_regularization=0.0001",
        "--loss_time_weighting=-0.001",
    ]

    args_train = [
        "train_growing_trajectory_two_stages",
        "--lr=0.05",
        "--lr_limit=0.00001",
        "--lr_decay_rate=0.3",
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
