# Activate the environment for running the script
if isfile("Project.toml") && isfile("Manifest.toml")
    import Pkg
    Pkg.activate(".")
end

include("experiments.jl")

function setup_fbmobility2(
    loc::AbstractString;
    train_range::Day = Day(32),
    forecast_range::Day = Day(28),
)
    train_dataset, test_dataset, first_date, last_date =
        experiment_covid19_data(loc, train_range, forecast_range)
    @assert size(train_dataset.data, 2) == Dates.value(train_range)
    @assert size(test_dataset.data, 2) == Dates.value(forecast_range)

    movement_range_data = experiment_movement_range(loc, first_date, last_date)
    @assert size(movement_range_data, 2) == Dates.value(train_range + forecast_range)

    social_proximity_lag = Day(14)
    social_proximity_data = experiment_social_proximity(
        loc,
        first_date - social_proximity_lag,
        last_date - social_proximity_lag,
    )
    @assert size(social_proximity_data, 2) == Dates.value(train_range + forecast_range)

    # get the initial states and available observations depending on the model type
    # and the considered location
    u0, vars, labels = experiment_SEIRD_initial_states(loc, train_dataset.data[:, 1])
    model = CovidModelSEIRDFbMobility2(
        u0,
        train_dataset.tspan,
        movement_range_data,
        social_proximity_data,
    )
    return model, train_dataset, test_dataset, vars, labels
end

function experiment_fbmobility2(
    loc::AbstractString;
    savedir::AbstractString,
    name::AbstractString = "fbmobility2",
)
    snapshots_dir = joinpath(savedir, loc)
    uuid = Dates.format(now(), "yyyymmddHHMMSS")
    sessname = "$uuid.$name.$loc"

    model, train_dataset, test_dataset, vars, labels = setup_fbmobility2(loc)
    predictor = Predictor(model.problem, vars)
    loss = experiment_loss(predictor, train_dataset, 0.001)

    p0 = Covid19ModelVN.initial_params(model)
    minimizers = train_model(
        loss,
        p0,
        TrainSession[TrainSession(
            "$sessname.bfgs",
            BFGS(initial_stepnorm = 0.01),
            100,
            100,
        )],
        lower_bounds = [
            1 / 5
            1 / 21
            0.0
            fill(-Inf, DiffEqFlux.paramlength(model.β_ann))
        ],
        upper_bounds = [1 / 2 1 / 7 0.06 fill(Inf, DiffEqFlux.paramlength(model.β_ann))],
        snapshots_dir = snapshots_dir,
    )

    minimizer = first(minimizers)
    eval_config = EvalConfig([mae, mape, rmse], [7, 14, 21, 28], labels)
    # get the effective reproduction number learned by the model
    R_effective_plot =
        plot_effective_reproduction_number(model, minimizer, train_dataset, test_dataset)
    save(joinpath(snapshots_dir, "$sessname.R_effective.png"), R_effective_plot)
    # plot the model's forecasts againts the ground truth, and calculate to model's
    # error on the test data
    forecasts_plot, df_forecasts_errors =
        evaluate_model(eval_config, predictor, minimizer, train_dataset, test_dataset)
    save(joinpath(snapshots_dir, "$sessname.forecasts.png"), forecasts_plot)
    save_dataframe(df_forecasts_errors, joinpath(snapshots_dir, "$sessname.errors.csv"))

    return R_effective_plot, forecasts_plot, df_forecasts_errors
end

for loc ∈
    [collect(keys(Covid19ModelVN.LOC_NAMES_VN)); collect(keys(Covid19ModelVN.LOC_NAMES_US))]

    R_effective_plot, forecasts_plot =
        experiment_fbmobility2(loc, savedir = "snapshots/test")
    display(R_effective_plot)
    display(forecasts_plot)
end
