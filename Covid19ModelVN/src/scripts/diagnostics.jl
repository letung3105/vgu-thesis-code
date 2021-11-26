using BenchmarkTools
using Distributions
using OrdinaryDiffEq.EnsembleAnalysis

include("include/cmd.jl")

function plot_movement_range(loc, data::AbstractMatrix{<:Real})
    fig = Figure()
    ax = Axis(fig[1, 1], title = "Movement Range Data for '$loc'")
    ln1 = lines!(ax, data[1, :])
    ln2 = lines!(ax, data[2, :])
    Legend(
        fig[2, 1],
        [ln1, ln2],
        ["relative change in movement", "stay put index"],
        tellwidth = false,
        tellheight = true,
        orientation = :horizontal,
    )
    return fig
end

function plot_social_proximity(loc, data::AbstractMatrix{<:Real})
    fig = Figure()
    ax = Axis(fig[1, 1], title = "Social Proximity to Cases for '$loc'")
    lines!(ax, data[1, :], label = "social proximity to cases")
    axislegend(ax, position = :lt)
    return fig
end

function visualize_data(loc; train_range = Day(32), forecast_range = Day(28))
    train_dataset, test_dataset, first_date, last_date =
        experiment_covid19_data(loc, train_range, forecast_range)

    fig = Figure()
    ncompartments = size(train_dataset.data, 1)
    for i ∈ 1:ncompartments
        ax = Axis(fig[1, i])
        @views data = [train_dataset.data[i, :]; test_dataset.data[i, :]]
        lines!(ax, data)
    end
    display(fig)

    movement_range_data = try
        experiment_movement_range(loc, first_date, last_date)
    catch e
        @warn e
    end
    if !isnothing(movement_range_data)
        fig = plot_movement_range(loc, movement_range_data)
        display(fig)
    end

    social_proximity_data = try
        experiment_social_proximity(loc, first_date, last_date)
    catch e
        @warn e
    end
    if !isnothing(social_proximity_data)
        fig = plot_social_proximity(loc, social_proximity_data)
        display(fig)
    end
end

function visualize_data_all_locations()
    for loc ∈ [
        Covid19ModelVN.LOC_CODE_VIETNAM
        Covid19ModelVN.LOC_CODE_UNITED_STATES
        collect(keys(Covid19ModelVN.LOC_NAMES_VN))
        collect(keys(Covid19ModelVN.LOC_NAMES_US))
    ]
        visualize_data(loc)
    end
end

function check_model_methods(loc, model)
    @info("Testing model", loc, model)

    # setup model at for a location with the default settings
    parsed_args =
        parse_commandline(["--locations=$loc", "--", model, "train_growing_trajectory"])
    loc = parsed_args[:locations][1]
    _, get_hyperparams, setup = setupcmd(parsed_args)

    # create the model
    hyperparams = get_hyperparams(parsed_args)
    model, u0, p0, lossfn_regularized, train_dataset, test_dataset, vars, labels =
        setup(loc; hyperparams...)

    # test if namedparams is implemented correctly
    pnamed = namedparams(model, p0)
    @assert sum(map(length, pnamed)) == length(p0)

    # test if the loss function works
    prob = ODEProblem(model, u0, train_dataset.tspan)
    predictor = Predictor(prob, vars)
    loss = Loss(lossfn_regularized, predictor, train_dataset)

    dLdθ = Zygote.gradient(loss, p0)
    @assert !isnothing(dLdθ[1]) # gradient is computable
    @assert any(dLdθ[1] .!= 0.0) # not all gradients are 0

    # plot Fb's data
    if hasfield(typeof(model), :movement_range_data)
        fig = plot_movement_range(loc, model.movement_range_data)
        display(fig)
    end
    if hasfield(typeof(model), :social_proximity_data)
        fig = plot_social_proximity(loc, model.social_proximity_data)
        display(fig)
    end

    # test effective reproduction number plot
    R1 = Re(model, u0, p0, train_dataset.tspan, train_dataset.tsteps)
    R2 = Re(model, u0, p0, test_dataset.tspan, test_dataset.tsteps)
    fig = plot_Re([R1; R2], train_dataset.tspan[2])
    display(fig)
    if model isa SEIRDFbMobility4
        αt1 = fatality_rate(model, u0, p0, train_dataset.tspan, train_dataset.tsteps)
        αt2 = fatality_rate(model, u0, p0, test_dataset.tspan, test_dataset.tsteps)
        fig_αt = plot_fatality_rate([αt1; αt2], train_dataset.tspan[2])
        display(fig_αt)
    end

    # test forecasts plot
    fit = predictor(p0, train_dataset.tspan, train_dataset.tsteps)
    pred = predictor(p0, test_dataset.tspan, test_dataset.tsteps)
    eval_conf = EvalConfig([mae, map, rmse], [7, 14, 21, 28], labels)
    fig = plot_forecasts(eval_conf, fit, pred, train_dataset, test_dataset)
    display(fig)
end

function check_model_performance(loc, model; benchmark = false)
    parsed_args =
        parse_commandline(["--locations=$loc", "--", model, "train_growing_trajectory"])
    _, gethyper, setup = setupcmd(parsed_args)
    hyperparams = gethyper(parsed_args)

    model, u0, p0, lossfn, train_dataset, _, vars, _ = setup(loc; hyperparams...)
    du = similar(u0)
    # check if dynamics function is type stable
    @code_warntype model(du, u0, p0, 0.0)

    prob = ODEProblem(model, u0, train_dataset.tspan)
    predictor = Predictor(prob, vars)
    # check if prediction function is type stable
    @code_warntype predictor(p0, train_dataset.tspan, train_dataset.tsteps)

    sol = predictor(p0, train_dataset.tspan, train_dataset.tsteps)
    pred = @view sol[:, :]
    # check if loss metric function with regularization is type stable
    @code_warntype lossfn(pred, train_dataset.data)

    loss = Loss(lossfn, predictor, train_dataset)
    # check if training loss is type stable
    @code_warntype loss(p0)

    if benchmark
        display(@benchmark $model($du, $u0, $p0, 0))
        display(@benchmark $predictor($p0, $train_dataset.tspan, $train_dataset.tsteps))
        display(@benchmark $lossfn($pred, $train_dataset.data))
        display(@benchmark $loss($p0))
        display(@benchmark Zygote.gradient($loss, $p0))
    end
end

function check_models()
    for loc ∈ [
            Covid19ModelVN.LOC_CODE_VIETNAM
            Covid19ModelVN.LOC_CODE_UNITED_STATES
            collect(keys(Covid19ModelVN.LOC_NAMES_VN))
            collect(keys(Covid19ModelVN.LOC_NAMES_US))
        ],
        model ∈ ["baseline", "fbmobility1"]

        check_model_methods(loc, model)
        check_model_performance(loc, model)
    end
    for loc ∈ [
            collect(keys(Covid19ModelVN.LOC_NAMES_VN))
            collect(keys(Covid19ModelVN.LOC_NAMES_US))
        ],
        model ∈ ["fbmobility2", "fbmobility3", "fbmobility4"]

        check_model_methods(loc, model)
        check_model_performance(loc, model)
    end
end
