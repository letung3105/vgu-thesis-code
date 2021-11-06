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
    metric_fn,
    solver;
    abstol = 1e-6,
    reltol = 1e-6,
)
    model, train_dataset, _ = experiment_setup("fbmobility2.default.hcm")
    p0 = Covid19ModelVN.initial_params(model)

    @info "Solver = $solver | abstol = $abstol | reltol = $reltol"

    predict_fn1 =
        Predictor(model.problem, solver, ForwardDiffSensitivity(), abstol, reltol)
    lossfn1 = Loss(metric_fn, predict_fn1, train_dataset, [5,6])
    @info "Compute gradient with sensealg = ForwardDiffSensitivity()"
    display(@benchmark Zygote.gradient($lossfn1, $p0))

    predict_fn2 =
        Predictor(model.problem, solver, ForwardSensitivity(), abstol, reltol)
    lossfn2 = Loss(metric_fn, predict_fn2, train_dataset, [5, 6])
    @info "Compute gradient with sensealg = ForwardSensitivity()"
    display(@benchmark Zygote.gradient($lossfn2, $p0))

    predict_fn3 = Predictor(
        model.problem,
        solver,
        BacksolveAdjoint(autojacvec = ReverseDiffVJP(false)),
        abstol,
        reltol,
    )
    lossfn3 = Loss(metric_fn, predict_fn3, train_dataset, [5, 6])
    @info "Compute gradient with sensealg = BacksolveAdjoint(autojacvec = ReverseDiffVJP(false))"
    display(@benchmark Zygote.gradient($lossfn3, $p0))

    predict_fn4 = Predictor(
        model.problem,
        solver,
        InterpolatingAdjoint(autojacvec = ReverseDiffVJP(false)),
        abstol,
        reltol,
    )
    lossfn4 = Loss(metric_fn, predict_fn4, train_dataset, [5, 6])
    @info "Compute gradient with sensealg = InterpolatingAdjoint(autojacvec = ReverseDiffVJP(false))"
    display(@benchmark Zygote.gradient($lossfn4, $p0))

    predict_fn5 = Predictor(
        model.problem,
        solver,
        BacksolveAdjoint(autojacvec = ReverseDiffVJP(true)),
        abstol,
        reltol,
    )
    lossfn5 = Loss(metric_fn, predict_fn5, train_dataset, [5, 6])
    @info "Compute gradient with sensealg = BacksolveAdjoint(autojacvec = ReverseDiffVJP(true))"
    display(@benchmark Zygote.gradient($lossfn5, $p0))

    predict_fn6 = Predictor(
        model.problem,
        solver,
        InterpolatingAdjoint(autojacvec = ReverseDiffVJP(true)),
        abstol,
        reltol,
    )
    lossfn6 = Loss(metric_fn, predict_fn6, train_dataset, [5, 6])
    @info "Compute gradient with sensealg = InterpolatingAdjoint(autojacvec = ReverseDiffVJP(true))"
    display(@benchmark Zygote.gradient($lossfn6, $p0))

    return nothing
end

function benchmarks_compute_loss_gradient_with_different_sensealg_forwarddiff(
    metric_fn,
    solver;
    abstol = 1e-6,
    reltol = 1e-6,
)
    model, train_dataset, _ = experiment_setup("fbmobility2.default.hcm")
    p0 = Covid19ModelVN.initial_params(model)

    @info "Solver = $solver | abstol = $abstol | reltol = $reltol"

    predict_fn1 =
        Predictor(model.problem, solver, ForwardDiffSensitivity(), abstol, reltol)
    lossfn1 = Loss(metric_fn, predict_fn1, train_dataset, [5,6])
    @info "Compute gradient with sensealg = ForwardDiffSensitivity()"
    display(@benchmark ForwardDiff.gradient($lossfn1, $p0))

    predict_fn2 =
        Predictor(model.problem, solver, ForwardSensitivity(), abstol, reltol)
    lossfn2 = Loss(metric_fn, predict_fn2, train_dataset, [5, 6])
    @info "Compute gradient with sensealg = ForwardSensitivity()"
    display(@benchmark ForwardDiff.gradient($lossfn2, $p0))

    predict_fn3 = Predictor(
        model.problem,
        solver,
        BacksolveAdjoint(autojacvec = ReverseDiffVJP(false)),
        abstol,
        reltol,
    )
    lossfn3 = Loss(metric_fn, predict_fn3, train_dataset, [5, 6])
    @info "Compute gradient with sensealg = BacksolveAdjoint(autojacvec = ReverseDiffVJP(false))"
    display(@benchmark ForwardDiff.gradient($lossfn3, $p0))

    predict_fn4 = Predictor(
        model.problem,
        solver,
        InterpolatingAdjoint(autojacvec = ReverseDiffVJP(false)),
        abstol,
        reltol,
    )
    lossfn4 = Loss(metric_fn, predict_fn4, train_dataset, [5, 6])
    @info "Compute gradient with sensealg = InterpolatingAdjoint(autojacvec = ReverseDiffVJP(false))"
    display(@benchmark ForwardDiff.gradient($lossfn4, $p0))

    predict_fn5 = Predictor(
        model.problem,
        solver,
        BacksolveAdjoint(autojacvec = ReverseDiffVJP(true)),
        abstol,
        reltol,
    )
    lossfn5 = Loss(metric_fn, predict_fn5, train_dataset, [5, 6])
    @info "Compute gradient with sensealg = BacksolveAdjoint(autojacvec = ReverseDiffVJP(true))"
    display(@benchmark ForwardDiff.gradient($lossfn5, $p0))

    predict_fn6 = Predictor(
        model.problem,
        solver,
        InterpolatingAdjoint(autojacvec = ReverseDiffVJP(true)),
        abstol,
        reltol,
    )
    lossfn6 = Loss(metric_fn, predict_fn6, train_dataset, [5, 6])
    @info "Compute gradient with sensealg = InterpolatingAdjoint(autojacvec = ReverseDiffVJP(true))"
    display(@benchmark ForwardDiff.gradient($lossfn6, $p0))

    return nothing
end

# benchmarks_compute_loss_gradient_with_different_sensealg_zygote(rmsle, Vern7())

# benchmarks_compute_loss_gradient_with_different_sensealg_zygote(rmsle, Vern9())

benchmarks_compute_loss_gradient_with_different_sensealg_zygote(rmsle, Tsit5())

# benchmarks_compute_loss_gradient_with_different_sensealg_forwarddiff(rmsle, Vern7())

# benchmarks_compute_loss_gradient_with_different_sensealg_forwarddiff(rmsle, Vern9())

# benchmarks_compute_loss_gradient_with_different_sensealg_forwarddiff(rmsle, Tsit5())