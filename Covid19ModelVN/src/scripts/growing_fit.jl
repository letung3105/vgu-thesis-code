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
    lr = 1e-2,
    batch_initial = 10,
    batch_growth = 10,
    maxiters = 100,
    maxiters_growth = 100,
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
    cb_general = TrainCallback(
        TrainCallbackState(eltype(params), length(params), showprogress),
        TrainCallbackConfig(
            evalloss,
            testloss,
            100,
            get_losses_save_fpath(snapshots_dir, uuid),
            get_params_save_fpath(snapshots_dir, uuid),
        ),
    )

    batch_max = length(train_dataset.tsteps)
    for k = batch_initial:batch_growth:batch_max
        cb = function (p, l)
            cb_videostream(p)
            cb_general(p, l)
        end

        train_dataset_batch = TimeseriesDataset(
            @view(train_dataset.data[:, 1:k]),
            (train_dataset.tspan[1], train_dataset.tspan[1] + k - 1),
            @view(train_dataset.tsteps[train_dataset.tsteps .< k])
        )
        @info "Train on tspan = $(train_dataset_batch.tspan) with tsteps = $(train_dataset.tsteps)"

        trainloss = Loss(lossfn, predictor, train_dataset_batch)
        res = DiffEqFlux.sciml_train(trainloss, params, ADAM(lr); maxiters, cb)
        params .= res.minimizer
        maxiters += maxiters_growth
    end

    fpath_vstream = joinpath(snapshots_dir, "$uuid.mp4")
    save(fpath_vstream, vstream)

    return params, cb_general.state.eval_losses, cb_general.state.test_losses
end

# for loc in union(keys(Covid19ModelVN.LOC_NAMES_US), keys(Covid19ModelVN.LOC_NAMES_VN))
# for loc in keys(Covid19ModelVN.LOC_NAMES_US)
let loc = "losangeles_ca"
    model = "fbmobility4"

    timestamp = Dates.format(now(), "yyyymmddHHMMSS")
    uuid = "$timestamp.$model.$loc"

    parsed_args =
        parse_commandline(["--locations=$loc", "--zeta=0", "--", model])
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
        lr = 0.01,
        batch_initial = 8,
        batch_growth = 8,
        maxiters = 400,
        maxiters_growth = 0,
        showprogress = true,
    )
    experiment_eval(uuid, setup, forecast_horizons, snapshots_dir)
end

# for loc in keys(Covid19ModelVN.LOC_NAMES_US)
let loc = "cook_il"
    model = "fbmobility4"

    timestamp = Dates.format(now(), "yyyymmddHHMMSS")
    uuid = "$timestamp.$model.$loc"

    runcmd([
        "--locations=$loc",
        "--savedir=snapshots/test",
        "--adam_maxiters=500",
        "--bfgs_maxiters=0",
        "--zeta=0",
        "--train_batchsize=8",
        "--show_progress",
        model,
    ])
end

let loc = "cook_il"
    model = "fbmobility4"

    timestamp = Dates.format(now(), "yyyymmddHHMMSS")
    uuid = "$timestamp.$model.$loc"

    parsed_args =
        parse_commandline(["--locations=$loc", "--zeta=0", "--", model])
    _, gethyper, model_setup = setupcmd(parsed_args)
    hyperparams = gethyper(parsed_args)
    setup = () -> model_setup(loc, hyperparams)
    model, u0, params, lossfn, train_dataset, test_dataset, vars, labels = setup()
    prob = ODEProblem(model, u0, train_dataset.tspan)
    predictor = Predictor(prob, vars)
    evalloss = Loss(lossfn, predictor, train_dataset)
    testloss = Loss(mae, predictor, test_dataset)

    y = train_dataset.data
    ŷ = train_dataset.data

    min = vec(minimum(y, dims = 2))
    max = vec(maximum(y, dims = 2))
    scale = max .- min
    lossfn = experiment_loss((0.5, 0.5))
    lossfn(ŷ, y)
end