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
    evalloss = Loss{true}(lossfn, predictor, train_dataset)
    testloss = Loss{false}(mae, predictor, test_dataset)

    eval_config = EvalConfig([mae, mape, rmse], forecast_horizons, labels)
    cb_videostream, vstream = make_video_stream_callback(
        predictor,
        params,
        train_dataset,
        test_dataset,
        eval_config,
    )
    cb_general = TrainCallback(
        TrainCallbackState(
            eltype(params),
            length(params),
            showprogress,
        ),
        TrainCallbackConfig(
            evalloss,
            testloss,
            100,
            get_losses_save_fpath(snapshots_dir, uuid),
            get_params_save_fpath(snapshots_dir, uuid),
        )
    )

    batch_max = length(train_dataset.tsteps)
    for k = batch_initial:batch_growth:batch_max
        cb = function (p, l)
            cb_videostream(p)
            cb_general(p, l)
        end

        train_dataset_batch = TimeseriesDataset(
            @view(train_dataset.data[:, 1:k]),
            train_dataset.tspan,
            @view(train_dataset.tsteps[1:k])
        )
        trainloss = Loss{true}(lossfn, predictor, train_dataset_batch)

        res = DiffEqFlux.sciml_train(trainloss, params, ADAM(lr); maxiters, cb)
        params .= res.minimizer
        maxiters += maxiters_growth
    end

    fpath_vstream = joinpath(snapshots_dir, "$uuid.mp4")
    save(fpath_vstream, vstream)

    return params, cb_general.state.eval_losses, cb_general.state.test_losses
end

# for loc in union(keys(Covid19ModelVN.LOC_NAMES_US), keys(Covid19ModelVN.LOC_NAMES_VN))
let loc = "hcm"
    model = "fbmobility4"

    timestamp = Dates.format(now(), "yyyymmddHHMMSS")
    uuid = "$timestamp.$model.$loc"

    parsed_args = parse_commandline([
        "--locations=$loc",
        "--zeta=0",
        "--L2_lambda=0",
        "--",
        model,
    ])
    _, gethyper, model_setup = setupcmd(parsed_args)
    hyperparams = gethyper(parsed_args)
    setup = () -> model_setup(loc, hyperparams);
    forecast_horizons = parsed_args[:forecast_horizons]

    snapshots_dir = joinpath("snapshots", loc)
    !isdir(snapshots_dir) && mkpath(snapshots_dir)

    growing_fit(
        uuid,
        setup;
        snapshots_dir,
        forecast_horizons,
        lr = 0.01,
        batch_initial = 4,
        batch_growth = 4,
        maxiters = 300,
        maxiters_growth = 0,
        showprogress = true,
    )
    experiment_eval(uuid, setup, forecast_horizons, snapshots_dir)
end

let
    loc = "losangeles_ca"
    model = "fbmobility4"

    parsed_args = parse_commandline([
        "--locations=$loc",
        "--zeta=0",
        "--L2_lambda=0",
        "--train_days=32",
        model,
    ])
    _, gethyper, model_setup = setupcmd(parsed_args)
    hyperparams = gethyper(parsed_args)
    minimizer = Serialization.deserialize("snapshots/losangeles_ca/20211119135242.fbmobility4.losangeles_ca.params.jls")
    model, u0, params, lossfn, train_dataset, test_dataset, vars, labels =
        model_setup(loc, hyperparams)

    αt = fatality_rate(
        model,
        u0,
        minimizer,
        train_dataset.tspan,
        train_dataset.tsteps,
    )
    lines(αt)
end