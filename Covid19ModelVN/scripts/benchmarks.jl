# Activate the environment for running the script
if isfile("Project.toml") && isfile("Manifest.toml")
    import Pkg
    Pkg.activate(".")
end

include("experiments.jl")

using Covid19ModelVN

import Covid19ModelVN.JHUCSSEData,
    Covid19ModelVN.FacebookData,
    Covid19ModelVN.PopulationData,
    Covid19ModelVN.VnExpressData,
    Covid19ModelVN.VnCdcData

using OrdinaryDiffEq, DiffEqFlux
using BenchmarkTools

function benchmarks_compute_loss_gradient_with_different_sensealg(
    exp_name,
    metric_fn;
    solver,
    abstol = 1e-6,
    reltol = 1e-6,
)
    model, train_dataset, _ = setup_experiment_vietnam(exp_name)
    p0 = Covid19ModelVN.initial_params(model)

    @info "Solver = $solver | abstol = $abstol | reltol = $reltol"

    predict_fn1 = Predictor(model.problem, solver, ForwardDiffSensitivity(), abstol, reltol)
    lossfn1 = Loss(metric_fn, predict_fn1, train_dataset, 3:6)
    @info "Compute gradient with sensealg = ForwardDiffSensitivity()"
    display(@benchmark gradient($lossfn1, $p0))

    predict_fn2 = Predictor(model.problem, solver, ForwardSensitivity(), abstol, reltol)
    lossfn2 = Loss(metric_fn, predict_fn2, train_dataset, 3:6)
    @info "Compute gradient with sensealg = ForwardSensitivity()"
    display(@benchmark gradient($lossfn2, $p0))

    predict_fn3 = Predictor(
        model.problem,
        solver,
        InterpolatingAdjoint(autojacvec = ReverseDiffVJP()),
        abstol,
        reltol,
    )
    lossfn3 = Loss(metric_fn, predict_fn3, train_dataset, 3:6)
    @info "Compute gradient with sensealg = InterpolatingAdjoint(autojacvec = ReverseDiffVJP())"
    display(@benchmark gradient($lossfn3, $p0))

    predict_fn4 = Predictor(
        model.problem,
        solver,
        BacksolveAdjoint(autojacvec = ReverseDiffVJP()),
        abstol,
        reltol,
    )
    lossfn4 = Loss(metric_fn, predict_fn4, train_dataset, 3:6)
    @info "Compute gradient with sensealg = BacksolveAdjoint(autojacvec = ReverseDiffVJP())"
    display(@benchmark gradient($lossfn4, $p0))

    predict_fn5 = Predictor(
        model.problem,
        solver,
        InterpolatingAdjoint(autojacvec = ReverseDiffVJP(true)),
        abstol,
        reltol,
    )
    lossfn5 = Loss(metric_fn, predict_fn5, train_dataset, 3:6)
    @info "Compute gradient with sensealg = InterpolatingAdjoint(autojacvec = ReverseDiffVJP(true))"
    display(@benchmark gradient($lossfn5, $p0))

    predict_fn6 = Predictor(
        model.problem,
        solver,
        BacksolveAdjoint(autojacvec = ReverseDiffVJP(true)),
        abstol,
        reltol,
    )
    lossfn6 = Loss(metric_fn, predict_fn6, train_dataset, 3:6)
    @info "Compute gradient with sensealg = BacksolveAdjoint(autojacvec = ReverseDiffVJP(true))"
    display(@benchmark gradient($lossfn6, $p0))

    return nothing
end

benchmarks_compute_loss_gradient_with_different_sensealg(
    "baseline.default.vietnam",
    rmsle,
    solver = Vern7(),
)

benchmarks_compute_loss_gradient_with_different_sensealg(
    "baseline.default.vietnam",
    rmsle,
    solver = Tsit5(),
)
