include("include/cmd.jl")

using ProgressMeter

function make_video_stream_callback(
    predictor,
    params,
    train_dataset,
    test_dataset,
    eval_config,
)
    model_fit = Node(predictor(params, train_dataset.tspan, train_dataset.tsteps))
    model_pred = Node(predictor(params, test_dataset.tspan, test_dataset.tsteps))
    fig = plot_forecasts(eval_config, model_fit, model_pred, train_dataset, test_dataset)
    vs = VideoStream(fig, framerate = 60)
    cb = function (params)
        model_fit[] = predictor(params, train_dataset.tspan, train_dataset.tsteps)
        model_pred[] = predictor(params, test_dataset.tspan, test_dataset.tsteps)
        autolimits!.(contents(fig[:, :]))
        recordframe!(vs)
    end
    return cb, vs
end

function growing_fit(
    uuid,
    setup;
    snapshots_dir,
    forecast_horizons,
    η = 1e-1,
    η_decay_rate = 0.5,
    η_decay_step = 100,
    η_limit = 1e-4,
    weight_decay = 1e-4,
    maxiters_initial = 100,
    maxiters_growth = 100,
    batch_initial = 10,
    batch_growth = 10,
    showprogress = false,
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

    # NOTE: order must be WeightDecay --> ADAM --> ExpDecay
    opt = Flux.Optimiser(
        WeightDecay(weight_decay),
        ADAM(),
        ExpDecay(η, η_decay_rate, η_decay_step, η_limit),
    )

    maxiters = maxiters_initial
    batch_max = length(train_dataset.tsteps)
    for k = batch_initial:batch_growth:batch_max
        cb = function (p, l)
            cb_videostream(p)
            cb_log(p, l)
        end

        train_dataset_batch = TimeseriesDataset(
            @view(train_dataset.data[:, 1:k]),
            (train_dataset.tspan[1], train_dataset.tspan[1] + k - 1),
            @view(train_dataset.tsteps[train_dataset.tsteps.<k])
        )
        @info "Train on tspan = $(train_dataset_batch.tspan) with tsteps = $(train_dataset.tsteps)"

        trainloss = Loss(lossfn, predictor, train_dataset_batch)
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
    η = 1e-1,
    η_decay_rate = 0.5,
    η_decay_step = 100,
    η_limit = 1e-4,
    weight_decay = 1e-4,
    maxiters = 1000,
    minibatching = 0,
    showprogress = false,
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
        ADAM(),
        ExpDecay(η, η_decay_rate, η_decay_step, η_limit),
    )
    res = DiffEqFlux.sciml_train(trainloss, params, opt; maxiters, cb)

    fpath_vstream = joinpath(snapshots_dir, "$uuid.mp4")
    save(fpath_vstream, vstream)

    return res.minimizer, cb_log.state.eval_losses, cb_log.state.test_losses
end

let loc = "losangeles_ca"
    model = "fbmobility4"

    timestamp = Dates.format(now(), "yyyymmddHHMMSS")
    uuid = "$timestamp.$model.$loc"

    parsed_args = parse_commandline(["--locations=$loc", "--", model])
    _, gethyper, model_setup = setupcmd(parsed_args)
    hyperparams = gethyper(parsed_args)
    setup = () -> model_setup(loc, hyperparams)
    forecast_horizons = parsed_args[:forecast_horizons]

    snapshots_dir = joinpath("snapshots", loc)
    !isdir(snapshots_dir) && mkpath(snapshots_dir)

    growing_fit(
        uuid,
        setup;
        snapshots_dir,
        forecast_horizons,
        maxiters_initial = 400,
        maxiters_growth = 0,
        batch_initial = 8,
        batch_growth = 8,
        showprogress = true,
    )
    experiment_eval(uuid, setup, forecast_horizons, snapshots_dir)
end


let loc = "losangeles_ca"
    model = "fbmobility4"

    timestamp = Dates.format(now(), "yyyymmddHHMMSS")
    uuid = "$timestamp.$model.$loc"

    parsed_args = parse_commandline(["--locations=$loc", "--", model])
    _, gethyper, model_setup = setupcmd(parsed_args)
    hyperparams = gethyper(parsed_args)
    setup = () -> model_setup(loc, hyperparams)
    forecast_horizons = parsed_args[:forecast_horizons]

    snapshots_dir = joinpath("snapshots", loc)
    !isdir(snapshots_dir) && mkpath(snapshots_dir)

    whole_fit(
        uuid,
        setup;
        snapshots_dir,
        forecast_horizons,
        η = 1e-1,
        η_decay_rate = 0.5,
        η_decay_step = 100,
        η_limit = 1e-4,
        maxiters = 1000,
        weight_decay = 1e-4,
        minibatching = 8,
        showprogress = true,
    )
    experiment_eval(uuid, setup, forecast_horizons, snapshots_dir)
end
