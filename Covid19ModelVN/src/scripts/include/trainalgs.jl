using Covid19ModelVN
using DiffEqFlux

function setup_model_training(
    model::AbstractCovidModel,
    u0::AbstractVector{<:Real},
    lossfn::Function,
    train_dataset::TimeseriesDataset,
    test_dataset::TimeseriesDataset,
    vars::AbstractVector{<:Integer},
)
    prob = ODEProblem(model, u0, train_dataset.tspan)
    predictor = Predictor(prob, vars)
    eval_loss = Loss(lossfn, predictor, train_dataset)
    test_loss = Loss(mae, predictor, test_dataset)
    return predictor, eval_loss, test_loss
end

function setup_training_callback(
    uuid,
    predictor,
    eval_loss,
    test_loss,
    params,
    train_dataset,
    test_dataset,
    labels,
    forecast_horizons,
    snapshots_dir,
    show_progress,
    make_animation,
)
    cb_log = TrainCallback(
        TrainCallbackState(eltype(params), length(params), show_progress),
        TrainCallbackConfig(
            eval_loss,
            test_loss,
            100,
            get_losses_save_fpath(snapshots_dir, uuid),
            get_params_save_fpath(snapshots_dir, uuid),
        ),
    )

    eval_config = EvalConfig([mae, mape, rmse], forecast_horizons, labels)
    cb_animation = ForecastsAnimationCallback(
        predictor,
        params,
        train_dataset,
        test_dataset,
        eval_config;
        framerate = 60,
    )

    cb = if make_animation
        function (p, l)
            cb_animation(p)
            cb_log(p, l)
            return false
        end
    else
        function (p, l)
            cb_log(p, l)
            return false
        end
    end

    return cb, cb_log, cb_animation
end

function train_growing_trajectory(
    uuid::AbstractString,
    setup::Function;
    snapshots_dir::AbstractString,
    forecast_horizons::AbstractVector{<:Integer},
    lr::Real,
    lr_decay_rate::Real,
    lr_decay_step::Integer,
    lr_limit::Real,
    weight_decay::Real,
    maxiters_initial::Integer,
    maxiters_growth::Integer,
    tspan_size_initial::Integer,
    tspan_size_growth::Integer,
    show_progress::Bool,
    make_animation::Bool,
)
    model, u0, params, lossfn, train_dataset, test_dataset, vars, labels = setup()
    predictor, eval_loss, test_loss =
        setup_model_training(model, u0, lossfn, train_dataset, test_dataset, vars)

    cb, cb_log, cb_animation = setup_training_callback(
        uuid,
        predictor,
        eval_loss,
        test_loss,
        params,
        train_dataset,
        test_dataset,
        labels,
        forecast_horizons,
        snapshots_dir,
        show_progress,
        make_animation,
    )

    maxiters = maxiters_initial
    tspan_size_max = length(train_dataset.tsteps)
    for k = tspan_size_initial:tspan_size_growth:tspan_size_max
        train_dataset_batch = TimeseriesDataset(
            @view(train_dataset.data[:, 1:k]),
            (train_dataset.tspan[1], train_dataset.tspan[1] + k - 1),
            @view(train_dataset.tsteps[train_dataset.tsteps.<k])
        )
        @info "Train on tspan = $(train_dataset_batch.tspan) with tsteps = $(train_dataset_batch.tsteps)"

        train_loss = Loss(lossfn, predictor, train_dataset_batch)
        # NOTE: order must be WeightDecay --> ADAM --> ExpDecay
        opt = Flux.Optimiser(
            WeightDecay(weight_decay),
            ADAM(lr),
            ExpDecay(lr, lr_decay_rate, lr_decay_step, lr_limit),
        )
        res = DiffEqFlux.sciml_train(train_loss, params, opt; maxiters, cb)
        params .= res.minimizer
        maxiters += maxiters_growth
    end

    if make_animation
        fpath_vstream = joinpath(snapshots_dir, "$uuid.mp4")
        save(fpath_vstream, cb_animation.vs)
    end

    return params, cb_log.state.eval_losses, cb_log.state.test_losses
end

function train_whole_trajectory(
    uuid::AbstractString,
    setup::Function;
    snapshots_dir::AbstractString,
    forecast_horizons::AbstractVector{<:Integer},
    lr::Real,
    lr_decay_rate::Real,
    lr_decay_step::Integer,
    lr_limit::Real,
    weight_decay::Real,
    maxiters::Integer,
    minibatching::Integer,
    show_progress::Bool,
)
    model, u0, params, lossfn, train_dataset, test_dataset, vars, labels = setup()
    predictor, eval_loss, test_loss =
        setup_model_training(model, u0, lossfn, train_dataset, test_dataset, vars)

    cb, cb_log, cb_animation = setup_training_callback(
        uuid,
        predictor,
        eval_loss,
        test_loss,
        params,
        train_dataset,
        test_dataset,
        labels,
        forecast_horizons,
        snapshots_dir,
        show_progress,
        true,
    )

    train_loss = if minibatching != 0
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
    res = DiffEqFlux.sciml_train(train_loss, params, opt; maxiters, cb)

    if make_animation
        fpath_vstream = joinpath(snapshots_dir, "$uuid.mp4")
        save(fpath_vstream, cb_animation.vs)
    end

    return res.minimizer, cb_log.state.eval_losses, cb_log.state.test_losses
end
