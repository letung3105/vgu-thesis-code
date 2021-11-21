using Covid19ModelVN
using DiffEqFlux

function growing_fit(
    uuid,
    setup;
    snapshots_dir,
    forecast_horizons,
    lr,
    lr_decay_rate,
    lr_decay_step,
    lr_limit,
    weight_decay,
    maxiters_initial,
    maxiters_growth,
    tspan_size_initial,
    tspan_size_growth,
    showprogress,
)
    model, u0, params, lossfn, train_dataset, test_dataset, vars, labels = setup()
    prob = ODEProblem(model, u0, train_dataset.tspan)
    predictor = Predictor(prob, vars)
    evalloss = Loss(lossfn, predictor, train_dataset)
    testloss = Loss(mae, predictor, test_dataset)

    eval_config = EvalConfig([mae, mape, rmse], forecast_horizons, labels)
    cb_videostream, vstream = make_video_stream_callback(
        predictor,
        params,
        train_dataset,
        test_dataset,
        eval_config,
    )
    cb_log = TrainCallback(
        TrainCallbackState(eltype(params), length(params), showprogress),
        TrainCallbackConfig(
            evalloss,
            testloss,
            100,
            get_losses_save_fpath(snapshots_dir, uuid),
            get_params_save_fpath(snapshots_dir, uuid),
        ),
    )
    cb = function (p, l)
        cb_videostream(p)
        cb_log(p, l)
    end

    maxiters = maxiters_initial
    tspan_size_max = length(train_dataset.tsteps)
    for k = tspan_size_initial:tspan_size_growth:tspan_size_max
        train_dataset_batch = TimeseriesDataset(
            @view(train_dataset.data[:, 1:k]),
            (train_dataset.tspan[1], train_dataset.tspan[1] + k - 1),
            @view(train_dataset.tsteps[train_dataset.tsteps.<k])
        )
        @info "Train on tspan = $(train_dataset_batch.tspan) with tsteps = $(train_dataset_batch.tsteps)"

        trainloss = Loss(lossfn, predictor, train_dataset_batch)
        # NOTE: order must be WeightDecay --> ADAM --> ExpDecay
        opt = Flux.Optimiser(
            WeightDecay(weight_decay),
            ADAM(lr),
            ExpDecay(lr, lr_decay_rate, lr_decay_step, lr_limit),
        )
        res = DiffEqFlux.sciml_train(trainloss, params, opt; maxiters, cb)
        params .= res.minimizer
        maxiters += maxiters_growth
    end

    fpath_vstream = joinpath(snapshots_dir, "$uuid.mp4")
    save(fpath_vstream, vstream)

    return params, cb_log.state.eval_losses, cb_log.state.test_losses
end

function whole_fit(
    uuid,
    setup;
    snapshots_dir,
    forecast_horizons,
    lr,
    lr_decay_rate,
    lr_decay_step,
    lr_limit,
    weight_decay,
    maxiters,
    minibatching,
    showprogress,
)
    model, u0, params, lossfn, train_dataset, test_dataset, vars, labels = setup()
    prob = ODEProblem(model, u0, train_dataset.tspan)
    predictor = Predictor(prob, vars)
    evalloss = Loss(lossfn, predictor, train_dataset)
    testloss = Loss(mae, predictor, test_dataset)

    eval_config = EvalConfig([mae, mape, rmse], forecast_horizons, labels)
    cb_videostream, vstream = make_video_stream_callback(
        predictor,
        params,
        train_dataset,
        test_dataset,
        eval_config,
    )
    cb_log = TrainCallback(
        TrainCallbackState(eltype(params), length(params), showprogress),
        TrainCallbackConfig(
            evalloss,
            testloss,
            100,
            get_losses_save_fpath(snapshots_dir, uuid),
            get_params_save_fpath(snapshots_dir, uuid),
        ),
    )
    cb = function (p, l)
        cb_videostream(p)
        cb_log(p, l)
    end

    trainloss = if minibatching != 0
        Loss(lossfn, predictor, train_dataset, minibatching)
    else
        Loss(lossfn, predictor, train_dataset)
    end
    # NOTE: order must be WeightDecay --> ADAM --> ExpDecay
    opt = Flux.Optimiser(
        WeightDecay(weight_decay),
        ADAM(lr),
        ExpDecay(lr, lr_decay_rate, lr_decay_step, lr_limit),
    )
    res = DiffEqFlux.sciml_train(trainloss, params, opt; maxiters, cb)

    fpath_vstream = joinpath(snapshots_dir, "$uuid.mp4")
    save(fpath_vstream, vstream)

    return res.minimizer, cb_log.state.eval_losses, cb_log.state.test_losses
end
