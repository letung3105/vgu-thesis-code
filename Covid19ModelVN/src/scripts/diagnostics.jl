using BenchmarkTools
using Distributions
using OrdinaryDiffEq.EnsembleAnalysis

include("include/cmd.jl")

function plot_movement_range!(
    pos::GridPosition, loc::AbstractString, data::AbstractMatrix{<:Real}
)
    ax = Axis(pos; title="Movement Range Data for '$loc'")
    @views lines!(ax, data[1, :]; label="rel. change")
    @views lines!(ax, data[2, :]; label="stay put")
    axislegend(ax; position=:lt)
    return ax
end

function plot_social_proximity!(
    pos::GridPosition, loc::AbstractString, data::AbstractMatrix{<:Real}
)
    ax = Axis(pos; title="Social Proximity to Cases for '$loc'")
    @views lines!(ax, data[1, :]; label="social proximity to cases")
    axislegend(ax; position=:lt)
    return ax
end

function visualize_data(
    loc;
    train_range=Day(32),
    forecast_range=Day(28),
    lag_movement_range=Day(0),
    lag_social_proximity=Day(0),
)
    conf, first_date, split_date, last_date = experiment_covid19_data(
        loc, train_range, forecast_range
    )
    train_dataset, test_dataset = train_test_split(conf, first_date, split_date, last_date)

    fig = Figure()
    current_row = 1

    ncompartments = size(train_dataset.data, 1)
    for i in 1:ncompartments
        ax = Axis(fig[i, 1])
        @views data = [train_dataset.data[i, :]; test_dataset.data[i, :]]
        lines!(ax, data)
        current_row += 1
    end

    try
        data = experiment_movement_range(
            loc, first_date, split_date, last_date, lag_movement_range
        )
        plot_movement_range!(fig[current_row, 1], loc, data)
        current_row += 1
    catch e
        @warn e
    end

    try
        data = experiment_social_proximity(
            loc, first_date, split_date, last_date, lag_social_proximity
        )
        plot_social_proximity!(fig[current_row, 1], loc, data)
        current_row += 1
    catch e
        @warn e
    end

    fig.scene.resolution = (600, 300 * (current_row - 1))
    return fig
end

function visualize_data_all_locations()
    for loc in [
        Covid19ModelVN.LOC_CODE_VIETNAM
        Covid19ModelVN.LOC_CODE_UNITED_STATES
        collect(keys(Covid19ModelVN.LOC_NAMES_VN))
        collect(keys(Covid19ModelVN.LOC_NAMES_US))
    ]
        fig = visualize_data(loc)
        display(fig)
    end
end

function check_model_methods(loc, model)
    @info("Testing model", loc, model)

    # setup model at for a location with the default settings
    parsed_args = parse_commandline([
        "--locations=$loc", "--", model, "train_growing_trajectory"
    ])
    loc = parsed_args[:locations][1]
    _, get_hyperparams, setup = setupcmd(parsed_args)

    # create the model
    hyperparams = get_hyperparams(parsed_args)
    model, u0, p0, lossfn, train_dataset, test_dataset, vars, labels = setup(
        loc; hyperparams...
    )

    # test if namedparams is implemented correctly
    pnamed = namedparams(model, p0)
    @assert sum(map(length, pnamed)) == length(p0)

    # test if the loss function works
    prob = ODEProblem(model, u0, train_dataset.tspan)
    predictor = Predictor(prob, vars)
    loss = Loss{true}(lossfn, predictor, train_dataset)

    dLdθ = Zygote.gradient(loss, p0)
    @assert !isnothing(dLdθ[1]) # gradient is computable
    @assert any(dLdθ[1] .!= 0.0) # not all gradients are 0

    # plot Fb's data
    if hasfield(typeof(model), :movement_range_data)
        fig = Figure()
        plot_movement_range!(fig[1, 1], loc, model.movement_range_data)
        display(fig)
    end
    if hasfield(typeof(model), :social_proximity_data)
        fig = Figure()
        plot_social_proximity!(fig[1, 1], loc, model.social_proximity_data)
        display(fig)
    end

    # test effective reproduction number plot
    R1 = Re(model, u0, p0, train_dataset.tspan, train_dataset.tsteps)
    R2 = Re(model, u0, p0, test_dataset.tspan, test_dataset.tsteps)
    fig = plot_Re([R1; R2], train_dataset.tspan[2])
    display(fig)

    αt1 = fatality_rate(model, u0, p0, train_dataset.tspan, train_dataset.tsteps)
    αt2 = fatality_rate(model, u0, p0, test_dataset.tspan, test_dataset.tsteps)
    fig_αt = plot_fatality_rate([αt1; αt2], train_dataset.tspan[2])
    display(fig_αt)

    # test forecasts plot
    fit = predictor(p0, train_dataset.tspan, train_dataset.tsteps)
    pred = predictor(p0, test_dataset.tspan, test_dataset.tsteps)
    eval_conf = EvalConfig([mae, map, rmse], [7, 14, 21, 28], labels)
    fig = plot_forecasts(eval_conf, fit, pred, train_dataset, test_dataset)
    return display(fig)
end

function check_model_performance(loc, model)
    parsed_args = parse_commandline([
        "--locations=$loc", "--", model, "train_growing_trajectory"
    ])
    _, gethyper, setup = setupcmd(parsed_args)
    hyperparams = gethyper(parsed_args)

    model, u0, p0, lossfn, train_dataset, _, vars, _ = setup(loc; hyperparams...)
    du = similar(u0)

    prob = ODEProblem(model, u0, train_dataset.tspan)
    predictor = Predictor(prob, vars)
    loss = Loss{true}(lossfn, predictor, train_dataset)

    sol = predictor(p0, train_dataset.tspan, train_dataset.tsteps)
    pred = @view sol[:, :]

    display(@benchmark $model($du, $u0, $p0, 0))
    display(@benchmark $predictor($p0, $train_dataset.tspan, $train_dataset.tsteps))
    display(@benchmark $lossfn($pred, $train_dataset.data))
    display(@benchmark $loss($p0))
    return display(@benchmark Zygote.gradient($loss, $p0))
end

function check_models(; benchmark=false)
    for loc in [
            Covid19ModelVN.LOC_CODE_VIETNAM
            Covid19ModelVN.LOC_CODE_UNITED_STATES
            collect(keys(Covid19ModelVN.LOC_NAMES_VN))
            collect(keys(Covid19ModelVN.LOC_NAMES_US))
        ],
        model in ["baseline", "fbmobility1"]

        check_model_methods(loc, model)
        benchmark && check_model_performance(loc, model)
    end
    for loc in [
            collect(keys(Covid19ModelVN.LOC_NAMES_VN))
            collect(keys(Covid19ModelVN.LOC_NAMES_US))
        ],
        model in ["fbmobility2"]

        check_model_methods(loc, model)
        benchmark && check_model_performance(loc, model)
    end
end
