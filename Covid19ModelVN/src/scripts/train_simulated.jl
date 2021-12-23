include("include/experiments.jl")

let
    pop = 9e6
    I0 = 500.0
    E0 = I0 * 2.0
    R0 = 0.0
    D0 = 0.0
    N0 = pop
    C0 = 0.0
    T0 = 500.0
    S0 = pop - E0 - T0

    u0 = [S0, E0, I0, R0, D0, N0, C0, T0]
    p0 = [0.2, 1.0 / 2.0, 1.0 / 14.0, 0.025]

    tspan = (0.0, 30.0)
    tsteps = tspan[1]:1.0:tspan[2]

    observables = [1, 2, 3, 4, 5, 6, 7]

    prob = ODEProblem(Covid19ModelVN.SEIRD!, u0, tspan)
    sol = solve(prob, Tsit5(); p=p0, saveat=tsteps)

    data = @view sol[observables, :]
    dataset = TimeseriesDataset(data, tspan, tsteps)

    model = SEIRDBaseline(
        (0.2 / 4, 6.68 / 4),
        (1.0 / 5.0, 1.0 / 2.0),
        (1.0 / 21.0, 1.0 / 7.0),
        (0.005, 0.05),
        pop,
        tspan[2]
    )
    params = initparams(model, 1.0 / 3.0, 1.0 / 20.0)
    prob = ODEProblem(model, u0, tspan)

    predictor = Predictor(prob, observables)
    loss_regularization  = 0.00001;
    lossfn_inner = experiment_loss_sse(
        vec(minimum(data; dims=2)), vec(maximum(data; dims=2)), -0.05
    )
    lossfn = function (ŷ, y, params, tsteps)
        pnamed = namedparams(model, params)
        return lossfn_inner(ŷ, y, tsteps) +
               loss_regularization / (2 * size(y, 2)) *
               (sum(abs2, pnamed.θ1) + sum(abs2, pnamed.θ2))
    end
    loss_batch = Loss{true}(lossfn, predictor, dataset, 10)
    loss_whole = Loss{true}(lossfn, predictor, dataset)

    progress = ProgressUnknown(; showspeed=true)
    minimizer = copy(params)
    loss_min = Inf
    cb = function (params, train_loss)
        l = loss_whole(params)
        if l < loss_min
            loss_min = l
            minimizer .= params
        end
        next!(progress; showvalues=[:train_loss => train_loss, :eval_loss => l])
        return false
    end

    DiffEqFlux.sciml_train(loss_batch, params, ADAM(5e-3); maxiters=5000, cb=cb)
    DiffEqFlux.sciml_train(
        loss_whole, minimizer, BFGS(; initial_stepnorm=1e-2); maxiters=10000, cb=cb
    )

    pnamed = namedparams(model, minimizer)
    @show pnamed.γ
    @show pnamed.λ

    pred = predictor(minimizer, tspan, tsteps)
    fig = Figure(; resolution=(600, 300 * length(observables)))
    for (i, obs) in enumerate(observables)
        ax = Axis(fig[i, 1])
        lines!(ax, data[obs, :])
        lines!(ax, pred[obs, :])
    end
    fig
end
