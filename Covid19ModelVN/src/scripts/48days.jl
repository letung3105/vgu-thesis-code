include("include/cmd.jl")

let
    args_model = [
        "--beta_bounds",
        0.2 / 4,
        6.68 / 4,
        "--train_days=48",
        "--loss_regularization=0.00001",
        "--loss_time_weighting=-0.001",
    ]

    args_train = [
        "train_growing_trajectory_two_stages",
        "--lr=0.05",
        "--lr_limit=0.00001",
        "--lr_decay_rate=0.5",
        "--lr_decay_step=1000",
        "--maxiters_initial=1000",
        "--maxiters_growth=0",
        "--maxiters_second=1000",
        "--tspan_size_initial=4",
        "--tspan_size_growth=4",
    ]

    for i in 1:10
        runcmd(
            string.([
                args_model...,
                "--locations",
                keys(Covid19ModelVN.LOC_NAMES_US)...,
                keys(Covid19ModelVN.LOC_NAMES_VN)...,
                "--savedir=testsnapshots/batch48days",
                "--multithreading",
                "fbmobility2",
                args_train...,
            ]),
        )

        for model in ["baseline", "fbmobility1"]
            runcmd(
                string.([
                    args_model...,
                    "--locations",
                    Covid19ModelVN.LOC_CODE_VIETNAM,
                    Covid19ModelVN.LOC_CODE_UNITED_STATES,
                    keys(Covid19ModelVN.LOC_NAMES_VN)...,
                    keys(Covid19ModelVN.LOC_NAMES_US)...,
                    "--savedir=testsnapshots/batch48days",
                    "--multithreading",
                    model,
                    args_train...,
                ]),
            )
        end
    end
end
