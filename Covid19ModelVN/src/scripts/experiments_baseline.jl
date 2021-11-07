# Activate the environment for running the script
if isfile("Project.toml") && isfile("Manifest.toml")
    import Pkg
    Pkg.activate(".")
end

include("experiments.jl")

function setup_baseline(
    loc::AbstractString;
    train_range::Day = Day(32),
    forecast_range::Day = Day(28),
)
    train_dataset, test_dataset = experiment_covid19_data(loc, train_range, forecast_range)
    @assert size(train_dataset.data, 2) == Dates.value(train_range)
    @assert size(test_dataset.data, 2) == Dates.value(forecast_range)

    # get the initial states and available observations depending on the model type
    # and the considered location
    u0, vars, labels = experiment_SEIRD_initial_states(loc, train_dataset.data[:, 1])
    model = CovidModelSEIRDBaseline(u0, train_dataset.tspan)
    return model, train_dataset, test_dataset, vars, labels
end

function experiment_baseline(
    loc::AbstractString;
    savedir::AbstractString,
    name::AbstractString = "baseline",
)
    snapshots_dir = joinpath(savedir, loc)
    uuid = Dates.format(now(), "yyyymmddHHMMSS")
    sessname = "$uuid.$name.$loc"

    model, train_dataset, test_dataset, vars, labels = setup_fbmobility1(loc)
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


experiment_baseline("hcm", savedir = "snapshots/test")