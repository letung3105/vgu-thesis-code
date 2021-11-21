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

function train_growing_trajectory(
    uuid::AbstractString,
    setup::Function;
    snapshots_dir::AbstractString,
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
)
    model, u0, params, lossfn, train_dataset, test_dataset, vars, labels = setup()
    predictor, eval_loss, test_loss =
        setup_model_training(model, u0, lossfn, train_dataset, test_dataset, vars)

    cb = TrainCallback(
        TrainCallbackState(eltype(params), length(params), show_progress),
        TrainCallbackConfig(
            eval_loss,
            test_loss,
            100,
            get_losses_save_fpath(snapshots_dir, uuid),
            get_params_save_fpath(snapshots_dir, uuid),
            get_minimizer_save_fpath(snapshots_dir, uuid),
        ),
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

    return params, cb.state.eval_losses, cb.state.test_losses
end

function train_whole_trajectory(
    uuid::AbstractString,
    setup::Function;
    snapshots_dir::AbstractString,
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

    cb = TrainCallback(
        TrainCallbackState(eltype(params), length(params), show_progress),
        TrainCallbackConfig(
            eval_loss,
            test_loss,
            100,
            get_losses_save_fpath(snapshots_dir, uuid),
            get_params_save_fpath(snapshots_dir, uuid),
            get_minimizer_save_fpath(snapshots_dir, uuid),
        ),
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

    return res.minimizer, cb.state.eval_losses, cb.state.test_losses
end
