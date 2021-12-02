using Covid19ModelVN
using DiffEqFlux
using ProgressMeter

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
    eval_loss = Loss{true}(lossfn, predictor, train_dataset)
    test_loss = Loss{false}(mae, predictor, test_dataset)
    return predictor, eval_loss, test_loss
end

function setup_training_callback(
    uuid::AbstractString,
    predictor::Predictor,
    eval_loss::Loss,
    test_loss::Loss,
    params::AbstractVector{<:Real},
    train_dataset::TimeseriesDataset,
    test_dataset::TimeseriesDataset,
    snapshots_dir::AbstractString,
    show_progress::Bool,
    shared_progress::Union{ProgressUnknown,Nothing},
    make_animation::Bool,
)
    cb_log = LogCallback(
        LogCallbackState(
            eltype(params),
            length(params),
            isnothing(shared_progress) ? show_progress : shared_progress,
        ),
        LogCallbackConfig(
            eval_loss,
            test_loss,
            get_losses_save_fpath(snapshots_dir, uuid),
            get_params_save_fpath(snapshots_dir, uuid),
        ),
    )

    cb_animation = ForecastsCallback(
        ForecastsCallbackState(eltype(params)),
        ForecastsCallbackConfig(
            predictor,
            train_dataset,
            test_dataset,
            get_forecasts_save_fpath(snapshots_dir, uuid),
        ),
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
    show_progress::Bool,
    shared_progress::Union{ProgressUnknown,Nothing},
    make_animation::Bool,
    lr::Real,
    lr_decay_rate::Real,
    lr_decay_step::Integer,
    lr_limit::Real,
    maxiters_initial::Integer,
    maxiters_growth::Integer,
    tspan_size_initial::Integer,
    tspan_size_growth::Integer,
)
    model, u0, params, lossfn, train_dataset, test_dataset, vars, _ = setup()
    predictor, eval_loss, test_loss =
        setup_model_training(model, u0, lossfn, train_dataset, test_dataset, vars)

    cb, cb_log, _ = setup_training_callback(
        uuid,
        predictor,
        eval_loss,
        test_loss,
        params,
        train_dataset,
        test_dataset,
        snapshots_dir,
        show_progress,
        shared_progress,
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
        @info(
            "Grew fitting time span",
            uuid,
            tspan = repr(train_dataset_batch.tspan),
            tsteps = repr(train_dataset_batch.tsteps),
        )

        @info("Training with ADAM optimizer", uuid)
        # NOTE: order must be WeightDecay --> ADAM --> ExpDecay
        res = DiffEqFlux.sciml_train(
            Loss{true}(lossfn, predictor, train_dataset_batch),
            params,
            Flux.Optimiser(ADAM(lr), ExpDecay(lr, lr_decay_rate, lr_decay_step, lr_limit));
            maxiters,
            cb,
        )
        params .= res.minimizer
        maxiters += maxiters_growth
    end

    return params, cb_log.state.eval_losses, cb_log.state.test_losses
end

function train_growing_trajectory_two_stages(
    uuid::AbstractString,
    setup::Function;
    snapshots_dir::AbstractString,
    show_progress::Bool,
    shared_progress::Union{ProgressUnknown,Nothing},
    make_animation::Bool,
    lr::Real,
    lr_decay_rate::Real,
    lr_decay_step::Integer,
    lr_limit::Real,
    maxiters_initial::Integer,
    maxiters_growth::Integer,
    maxiters_second::Integer,
    tspan_size_initial::Integer,
    tspan_size_growth::Integer,
)
    model, u0, params, lossfn, train_dataset, test_dataset, vars, _ = setup()
    predictor, eval_loss, test_loss =
        setup_model_training(model, u0, lossfn, train_dataset, test_dataset, vars)

    cb, cb_log, _ = setup_training_callback(
        uuid,
        predictor,
        eval_loss,
        test_loss,
        params,
        train_dataset,
        test_dataset,
        snapshots_dir,
        show_progress,
        shared_progress,
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
        @info(
            "Grew fitting time span",
            uuid,
            tspan = repr(train_dataset_batch.tspan),
            tsteps = repr(train_dataset_batch.tsteps),
        )

        @info("Training with ADAM optimizer", uuid)
        # NOTE: order must be WeightDecay --> ADAM --> ExpDecay
        res = DiffEqFlux.sciml_train(
            Loss{true}(lossfn, predictor, train_dataset_batch),
            params,
            Flux.Optimiser(ADAM(lr), ExpDecay(lr, lr_decay_rate, lr_decay_step, lr_limit));
            maxiters,
            cb,
        )
        params .= res.minimizer
        maxiters += maxiters_growth
    end


    @info("Training with BFGS optimizer", uuid)
    res = DiffEqFlux.sciml_train(
        Loss{true}(lossfn, predictor, train_dataset),
        params,
        BFGS(initial_stepnorm = 1e-2);
        maxiters = maxiters_second,
        allow_f_increases = true,
        cb,
    )

    return res.minimizer, cb_log.state.eval_losses, cb_log.state.test_losses
end

function train_whole_trajectory(
    uuid::AbstractString,
    setup::Function;
    snapshots_dir::AbstractString,
    show_progress::Bool,
    shared_progress::Union{ProgressUnknown,Nothing},
    make_animation::Bool,
    lr::Real,
    lr_decay_rate::Real,
    lr_decay_step::Integer,
    lr_limit::Real,
    maxiters::Integer,
    minibatching::Integer,
)
    model, u0, params, lossfn, train_dataset, test_dataset, vars, _ = setup()
    predictor, eval_loss, test_loss =
        setup_model_training(model, u0, lossfn, train_dataset, test_dataset, vars)

    cb, cb_log, _ = setup_training_callback(
        uuid,
        predictor,
        eval_loss,
        test_loss,
        params,
        train_dataset,
        test_dataset,
        snapshots_dir,
        show_progress,
        shared_progress,
        make_animation,
    )

    @info("Training with ADAM optimizer", uuid)
    # NOTE: order must be WeightDecay --> ADAM --> ExpDecay
    DiffEqFlux.sciml_train(
        if minibatching != 0
            Loss(lossfn, predictor, train_dataset, minibatching)
        else
            Loss(lossfn, predictor, train_dataset)
        end,
        params,
        Flux.Optimiser(ADAM(lr), ExpDecay(lr, lr_decay_rate, lr_decay_step, lr_limit));
        maxiters,
        cb,
    )

    return cb_log.state.minimizer, cb_log.state.eval_losses, cb_log.state.test_losses
end

function train_whole_trajectory_two_stages(
    uuid::AbstractString,
    setup::Function;
    snapshots_dir::AbstractString,
    show_progress::Bool,
    shared_progress::Union{ProgressUnknown,Nothing},
    make_animation::Bool,
    lr::Real,
    lr_decay_rate::Real,
    lr_decay_step::Integer,
    lr_limit::Real,
    maxiters_first::Integer,
    maxiters_second::Integer,
    minibatching::Integer,
)
    model, u0, params, lossfn, train_dataset, test_dataset, vars, _ = setup()
    predictor, eval_loss, test_loss =
        setup_model_training(model, u0, lossfn, train_dataset, test_dataset, vars)

    cb, cb_log, _ = setup_training_callback(
        uuid,
        predictor,
        eval_loss,
        test_loss,
        params,
        train_dataset,
        test_dataset,
        snapshots_dir,
        show_progress,
        shared_progress,
        make_animation,
    )

    @info("Training with ADAM optimizer", uuid)
    # NOTE: order must be WeightDecay --> ADAM --> ExpDecay
    DiffEqFlux.sciml_train(
        if minibatching != 0
            Loss{true}(lossfn, predictor, train_dataset, minibatching)
        else
            Loss{true}(lossfn, predictor, train_dataset)
        end,
        params,
        Flux.Optimiser(ADAM(lr), ExpDecay(lr, lr_decay_rate, lr_decay_step, lr_limit));
        maxiters = maxiters_first,
        cb,
    )

    @info("Training with BFGS optimizer", uuid)
    DiffEqFlux.sciml_train(
        Loss{true}(lossfn, predictor, train_dataset),
        cb_log.state.minimizer,
        BFGS(initial_stepnorm = 1e-2);
        maxiters = maxiters_second,
        allow_f_increases = true,
        cb,
    )

    return cb_log.state.minimizer, cb_log.state.eval_losses, cb_log.state.test_losses
end
