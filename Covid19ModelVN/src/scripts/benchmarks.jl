include("include/cmd.jl")

using BenchmarkTools

let
    models = [
        ("baseline", get_baseline_hyperparams, setup_baseline),
        ("fbmobility1", get_fbmobility1_hyperparams, setup_fbmobility1),
        ("fbmobility2", get_fbmobility2_hyperparams, setup_fbmobility2),
        ("fbmobility3", get_fbmobility3_hyperparams, setup_fbmobility3),
        ("fbmobility4", get_fbmobility4_hyperparams, setup_fbmobility4),
    ]

    for (name, gethyperparams, setup) ∈ models
        @info("Benchmarking model", name)

        parsed_args =
            parse_commandline([name, "--locations=hcm", "--", "train_whole_trajectory"])
        hyperparams = gethyperparams(parsed_args)

        model, u0, p0, lossfn, train_dataset, _, vars, _ =
            setup(parsed_args[:locations][1]; hyperparams...)
        prob = ODEProblem(model, u0, train_dataset.tspan)
        predictor = Predictor(prob, vars)

        loss1 = let
            l = experiment_loss_polar((0.5f0, 0.5f0))
            Loss(lossfn, predictor, train_dataset)
        end
        loss2 = let
            l = experiment_loss_sse(
                vec(minimum(train_dataset.data, dims = 2)),
                vec(maximum(train_dataset.data, dims = 2)),
            )
            Loss(l, predictor, train_dataset)
        end
        loss3 = let
            l = experiment_loss_ssle()
            Loss(l, predictor, train_dataset)
        end

        display(@benchmark Zygote.gradient($loss1, $p0))
        println()
        display(@benchmark Zygote.gradient($loss2, $p0))
        println()
        display(@benchmark Zygote.gradient($loss3, $p0))
        println()
    end
end
