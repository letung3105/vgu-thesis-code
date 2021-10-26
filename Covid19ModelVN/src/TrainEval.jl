module TrainEval

export train_model,
    evaluate_model, calculate_forecasts_errors, plot_forecasts, lookup_saved_params

using Serialization, Plots, DataFrames, CSV, DiffEqFlux

"""
Get default losses figure file path

# Arguments

* `fdir`: the root directory of the file
* `uuid`: the file unique identifier
"""
get_losses_plot_fpath(fdir, uuid) = joinpath(fdir, "$uuid.losses.png")

"""
Get default file path for saved parameters

# Arguments

* `fdir`: the root directory of the file
* `uuid`: the file unique identifier
"""
get_params_save_fpath(fdir, uuid) = joinpath(fdir, "$uuid.params.jls")

function train_model(train_loss_fn, test_loss_fn, p0, sessions, exp_dir)
    params = p0
    for sess ∈ sessions
        losses_plot_fpath = get_losses_plot_fpath(exp_dir, sess.name)
        params_save_fpath = get_params_save_fpath(exp_dir, sess.name)
        cb = TrainCallback(
            sess.maxiters,
            TrainCallbackConfig(
                train_loss_fn,
                losses_plot_fpath,
                div(sess.maxiters, 20),
                params_save_fpath,
                div(sess.maxiters, 20),
            ),
        )

        @info "Running $(sess.name)"
        try
            DiffEqFlux.sciml_train(
                test_loss_fn,
                params,
                sess.optimizer,
                maxiters = sess.maxiters,
                cb = cb,
            )
        catch e
            @error e
            if isa(e, InterruptException)
                rethrow(e)
            end
        end

        params = cb.state.minimizer
        Serialization.serialize(params_save_fpath, params)
    end
    return nothing
end

"""
Get the file paths and uuids of all the saved parameters of an experiment

# Arguments

* `dir`: the directory that contains the saved parameters
"""
function lookup_saved_params(dir)
    params_files = filter(x -> endswith(x, ".jls"), readdir(dir))
    fpaths = map(f -> joinpath(dir, f), params_files)
    uuids = map(f -> first(rsplit(f, ".", limit = 3)), params_files)
    return fpaths, uuids
end

function calculate_forecasts_errors(
    metric_fn,
    pred,
    test_dataset,
    forecast_ranges,
    vars,
    labels,
)
    errors = [
        metric_fn(pred[var, 1:days], test_dataset.data[col, 1:days]) for
        days ∈ forecast_ranges, (col, var) ∈ enumerate(vars)
    ]
    return DataFrame(errors, vec(labels))
    CSV.write(csv_fpath, df)
end

"""
Plot the forecasted values produced by `predict_fn` against the ground truth data and calculated the error for each
forecasted value using `metric_fn`.

# Arguments

* `predict_fn`: a function that takes the model parameters and computes the ODE solver output
* `metric_fn`: a function that computes the error between two data arrays
* `train_dataset`: the data that was used to train the model
* `test_dataset`: ground truth data for the forecasted period
* `minimizer`: the model's paramerters that will be used to produce the forecast
* `forecast_ranges`: a range of day horizons that will be forecasted
* `vars`: the model's states that will be compared
* `cols`: the ground truth values that will be compared
* `labels`: plot labels of each of the ground truth values
"""
function plot_forecasts(
    fit,
    pred,
    train_dataset,
    test_dataset,
    forecast_ranges,
    vars,
    labels,
)
    plts = []
    for days ∈ forecast_ranges, (col, (var, label)) ∈ enumerate(zip(vars, labels))
        plt = plot(title = "$days-day forecast", legend = :outertop, xrotation = 45)
        scatter!(
            [train_dataset.data[col, :]; test_dataset.data[col, 1:days]],
            label = label,
            fillcolor = nothing,
        )
        plot!([fit[var, :]; pred[var, 1:days]], label = "forecast $label", lw = 2)
        vline!([train_dataset.tspan[2]], color = :black, ls = :dash, label = nothing)
        push!(plts, plt)
    end

    nforecasts = length(forecast_ranges)
    nvariables = length(vars)
    return plot(
        plts...,
        layout = (nforecasts, nvariables),
        size = (300 * nvariables, 300 * nforecasts),
    )
end

function evaluate_model(
    predict_fn,
    train_dataset,
    test_dataset,
    exp_dir,
    eval_forecast_ranges,
    eval_vars,
    eval_labels,
)
    fpaths_params, uuids = lookup_saved_params(exp_dir)
    for (fpath_params, uuid) ∈ zip(fpaths_params, uuids)
        fit, pred = try
            minimizer = Serialization.deserialize(fpath_params)
            fit = predict_fn(minimizer, train_dataset.tspan, train_dataset.tsteps)
            pred = predict_fn(minimizer, test_dataset.tspan, test_dataset.tsteps)
            (fit, pred)
        catch e
            if isa(e, InterruptException)
                rethrow(e)
            end
            @warn e
            continue
        end

        for metric_fn ∈ [rmse, mae, mape]
            csv_fpath = joinpath(exp_dir, "$uuid.evaluate.$metric_fn.csv")
            if !isfile(csv_fpath)
                df = calculate_forecasts_errors(
                    metric_fn,
                    pred,
                    test_dataset,
                    eval_forecast_ranges,
                    eval_vars,
                    eval_labels,
                )
                CSV.write(csv_fpath, df)
            end
        end

        fig_fpath = joinpath(exp_dir, "$uuid.evaluate.png")
        if !isfile(fig_fpath)
            plt = plot_forecasts(
                fit,
                pred,
                train_dataset,
                test_dataset,
                eval_forecast_ranges,
                eval_vars,
                eval_labels,
            )
            savefig(plt, fig_fpath)
        end
    end
end

end
