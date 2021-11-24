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

        model, u0, p0, lossfn, train_dataset, test_dataset, vars, labels =
            setup(parsed_args[:locations][1]; hyperparams...)
        prob = ODEProblem(model, u0, train_dataset.tspan)
        predictor = Predictor(prob, vars)
        loss1 = Loss(lossfn, predictor, train_dataset)
        loss2 = Loss(mae, predictor, test_dataset)

        sol = predictor(p0, train_dataset.tspan, train_dataset.tsteps)
        pred = @view sol[:, :]
        du = similar(u0)

        @code_warntype model(du, u0, p0, 0)
        @code_warntype lossfn(pred, train_dataset.data)
        @code_warntype predictor(p0, train_dataset.tspan, train_dataset.tsteps)
        @code_warntype loss1(p0)
        @code_warntype loss2(p0)

        @show Zygote.gradient(loss1, p0)

        display(@benchmark $model($du, $u0, $p0, 0))
        println()
        display(@benchmark $lossfn($pred, $train_dataset.data))
        println()
        display(@benchmark $predictor($p0, $train_dataset.tspan, $train_dataset.tsteps))
        println()
        display(@benchmark $loss1($p0))
        println()
        display(@benchmark $loss2($p0))
        println()
    end
end
