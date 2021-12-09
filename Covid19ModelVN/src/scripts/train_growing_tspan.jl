include("include/cmd.jl")

# 28 days --> 1800
# 42 days --> 850
# 56 days --> 500

runcmd(
    string.([
        "--train_days=42",
        "--loss_regularization=0.0001",
        "--loss_time_weighting=-0.001",
        "--locations=dongnai",
        "--savedir=testsnapshots/testhyperparams",
        "--show_progress",
        "fbmobility2",
        "train_growing_trajectory_two_stages",
        "--lr=0.05",
        "--lr_limit=0.00001",
        "--lr_decay_rate=0.5",
        "--lr_decay_step=1000",
        "--maxiters_initial=850",
        "--maxiters_growth=850",
        "--maxiters_second=1000",
        "--tspan_size_initial=7",
        "--tspan_size_growth=7",
    ]),
)


let
    args_model = [
        "--train_days=56",
        "--loss_regularization=0.0001",
        "--loss_time_weighting=-0.001",
    ]

    args_train = [
        "train_growing_trajectory_two_stages",
        "--lr=0.05",
        "--lr_limit=0.00001",
        "--lr_decay_rate=0.5",
        "--lr_decay_step=1000",
        "--maxiters_initial=500",
        "--maxiters_growth=500",
        "--maxiters_second=1000",
        "--tspan_size_initial=7",
        "--tspan_size_growth=7",
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
                "--savedir=testsnapshots/batch56days",
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
                "--savedir=testsnapshots/batch56days",
                "--multithreading",
                model,
                args_train...,
            ]),
        )
    end
end

let
    args_model = [
        "--train_days=42",
        "--loss_regularization=0.0001",
        "--loss_time_weighting=-0.001",
    ]

    args_train = [
        "train_growing_trajectory_two_stages",
        "--lr=0.05",
        "--lr_limit=0.00001",
        "--lr_decay_rate=0.5",
        "--lr_decay_step=1000",
        "--maxiters_initial=850",
        "--maxiters_growth=850",
        "--maxiters_second=1000",
        "--tspan_size_initial=7",
        "--tspan_size_growth=7",
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
                "--savedir=testsnapshots/batch42days",
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
                "--savedir=testsnapshots/batch42days",
                "--multithreading",
                model,
                args_train...,
            ]),
        )
    end
end

let
    args_model = [
        "--train_days=28",
        "--loss_regularization=0.0001",
        "--loss_time_weighting=-0.001",
    ]

    args_train = [
        "train_growing_trajectory_two_stages",
        "--lr=0.05",
        "--lr_limit=0.00001",
        "--lr_decay_rate=0.5",
        "--lr_decay_step=1000",
        "--maxiters_initial=1800",
        "--maxiters_growth=1800",
        "--maxiters_second=1000",
        "--tspan_size_initial=7",
        "--tspan_size_growth=7",
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
                "--savedir=testsnapshots/batch28days",
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
                "--savedir=testsnapshots/batch28days",
                "--multithreading",
                model,
                args_train...,
            ]),
        )
    end
end
