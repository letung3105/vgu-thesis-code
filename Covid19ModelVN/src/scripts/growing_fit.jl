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
    return vs, cb
end

function make_losses_record_callback(evalloss, testloss)
    evallosses = Float32[]
    testlosses = Float32[]
    cb = function (params)
        l1 = evalloss(params)
        l2 = testloss(params)
        push!(evallosses, l1)
        push!(testlosses, l2)
    end
    return evallosses, testlosses, cb
end

function make_snapshot_callback(fpath, interval, p0)
    minloss = Inf32
    minparams = copy(p0)
    iters = 0
    cb = function (params, loss)
        if loss < minloss && size(params) == size(minparams)
            minloss = loss
            minparams .= params
        end
        iters += 1
        if iters % interval == 0
            Serialization.serialize(fpath, minparams)
        end
    end
    return cb
end

function growing_fit(
    uuid,
    model_setup,
    location,
    hyperparams;
    lr = 1e-2,
    batch_initial = 10,
    batch_growth = 10,
    maxiters = 100,
    maxiters_growth = 100,
    forecast_horizons,
    savedir,
)
    snapshots_dir = joinpath(savedir, location)
    if !isdir(snapshots_dir)
        mkpath(snapshots_dir)
    end

    model, u0, params, lossfn, train_dataset, test_dataset, vars, labels =
        model_setup(location, hyperparams)
    prob = ODEProblem(model, u0, train_dataset.tspan)
    predictor = Predictor(prob, vars)
    evalloss = Loss{true}(lossfn, predictor, train_dataset)
    testloss = Loss{false}(mae, predictor, test_dataset)

    eval_config = EvalConfig([mae, mape, rmse], forecast_horizons, labels)
    vs, cb_videostream = make_video_stream_callback(
        predictor,
        params,
        train_dataset,
        test_dataset,
        eval_config,
    )
    evallosses, testlosses, cb_losses = make_losses_record_callback(evalloss, testloss)

    batch_max = length(train_dataset.tsteps)
    for k = batch_initial:batch_growth:batch_max
        cb_snapshot = make_snapshot_callback(get_params_save_fpath(snapshots_dir, uuid), 100, params)
        progress = Progress(maxiters, showspeed = true)
        cb = function (p, l)
            cb_losses(p)
            cb_videostream(p)
            cb_snapshot(p, l)
            next!(
                progress,
                showvalues = [
                    :batchsize => k,
                    :train_loss => l,
                    :eval_loss => last(evallosses),
                    :test_loss => last(testlosses),
                ],
            )
            false
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

    fig_losses = plot_losses(evallosses, testlosses)
    ℜe1 = ℜe(model, u0, params, train_dataset.tspan, train_dataset.tsteps)
    ℜe2 = ℜe(model, u0, params, test_dataset.tspan, test_dataset.tsteps)
    fig_ℜe = plot_ℜe([ℜe1; ℜe2], train_dataset.tspan[2])
    fig_forecasts, df_errors =
        evaluate_model(eval_config, predictor, params, train_dataset, test_dataset)

    fpath_videostream = joinpath(snapshots_dir, "$uuid.mp4")
    fpath_losses = joinpath(snapshots_dir, "$uuid.losses.png")
    fpath_ℜe = joinpath(snapshots_dir, "$uuid.R_effective.png")
    fpath_forecasts = joinpath(snapshots_dir, "$uuid.forecasts.png")
    fpath_errors = joinpath(snapshots_dir, "$uuid.errors.csv")

    save(fpath_videostream, vs)
    save(fpath_losses, fig_losses)
    save(fpath_ℜe, fig_ℜe)
    save(fpath_forecasts, fig_forecasts)
    save_dataframe(df_errors, fpath_errors)

    return fig_losses, fig_forecasts, fig_ℜe, df_errors
end

for loc in keys(Covid19ModelVN.LOC_NAMES_US)
    timestamp = Dates.format(now(), "yyyymmddHHMMSS")
    # loc = "harris_tx"
    model = "fbmobility4"
    uuid = "$timestamp.$model.$loc"

    parsed_args = parse_commandline([
        "--locations=$loc",
        "--zeta=0",
        "--L2_lambda=0.00001",
        "--train_days=32",
        model,
    ])
    _, gethyper, model_setup = setupcmd(parsed_args)
    hyperparams = gethyper(parsed_args)

    fig_losses, fig_forecast, fig_ℜe, df_errors = growing_fit(
        uuid,
        model_setup,
        loc,
        hyperparams;
        batch_initial = 8,
        batch_growth = 8,
        maxiters = 300,
        maxiters_growth = 300,
        forecast_horizons = parsed_args[:forecast_horizons],
        savedir = "snapshots"
    )
    display(fig_losses)
    display(fig_forecast)
    display(fig_ℜe)
end
