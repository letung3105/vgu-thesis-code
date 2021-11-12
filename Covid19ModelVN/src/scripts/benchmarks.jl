# Activate the environment for running the script
if isfile("Project.toml") && isfile("Manifest.toml")
    import Pkg
    Pkg.activate(".")
end

include("experiments.jl")

using OrdinaryDiffEq, DiffEqFlux
using Covid19ModelVN
using BenchmarkTools

import DiffEqFlux.ForwardDiff


function benchmarks_compute_loss_gradient_with_different_sensealg_zygote(
    experiment_name,
    solver;
    abstol = 1e-6,
    reltol = 1e-6,
)
    @info "Solver = $solver | abstol = $abstol | reltol = $reltol"

    model, train_dataset, _, vars, _ = experiment_setup(experiment_name)
    params = Covid19ModelVN.initial_params(model)

    benchmark_zygote_gradient = function (sensealg)
        predictor = Predictor(model.problem, solver, sensealg, abstol, reltol, vars)
        loss = Loss(rmse, predictor, train_dataset)
        display(@benchmark Zygote.gradient($loss, $params))
    end

    @info "Compute gradient with sensealg = ForwardDiffSensitivity()"
    benchmark_zygote_gradient(ForwardDiffSensitivity())

    @info "Compute gradient with sensealg = ForwardSensitivity()"
    benchmark_zygote_gradient(ForwardSensitivity())

    @info "Compute gradient with sensealg = BacksolveAdjoint(autojacvec = ReverseDiffVJP(false))"
    benchmark_zygote_gradient(BacksolveAdjoint(autojacvec = ReverseDiffVJP(false)))

    @info "Compute gradient with sensealg = InterpolatingAdjoint(autojacvec = ReverseDiffVJP(false))"
    benchmark_zygote_gradient(InterpolatingAdjoint(autojacvec = ReverseDiffVJP(false)))

    @info "Compute gradient with sensealg = BacksolveAdjoint(autojacvec = ReverseDiffVJP(true))"
    benchmark_zygote_gradient(BacksolveAdjoint(autojacvec = ReverseDiffVJP(true)))

    @info "Compute gradient with sensealg = InterpolatingAdjoint(autojacvec = ReverseDiffVJP(true))"
    benchmark_zygote_gradient(InterpolatingAdjoint(autojacvec = ReverseDiffVJP(true)))

    return nothing
end

benchmarks_compute_loss_gradient_with_different_sensealg_zygote(
    "baseline.default.hcm",
    Tsit5(),
)
benchmarks_compute_loss_gradient_with_different_sensealg_zygote(
    "fbmobility1.default.hcm",
    Tsit5(),
)
benchmarks_compute_loss_gradient_with_different_sensealg_zygote(
    "fbmobility2.default.hcm",
    Tsit5(),
)
