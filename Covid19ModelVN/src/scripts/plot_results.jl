using CSV: CSV
using CairoMakie

function plot_outputs_combined(
    dir, uuids, locs, labels, plot_type; train_range=Day(48), forecast_range=Day(28)
)
    fig = Figure(; resolution=(400 * length(locs), 400 * length(labels)))
    for (i, label) in enumerate(labels)
        Label(
            fig[i, 1:length(locs), Top()],
            label;
            valign=:bottom,
            padding=(0, 0, 26, 0),
            textsize=26,
        )
        for (j, loc) in enumerate(locs)
            dataconf, first_date, split_date, last_date = experiment_covid19_data(
                loc, train_range, forecast_range
            )
            train_dataset, test_dataset = train_test_split(
                dataconf, first_date, split_date, last_date
            )

            ground_truth = if plot_type == :fit
                train_dataset.data[i, :]
            elseif plot_type == :pred
                test_dataset.data[i, :]
            end

            ax = Axis(fig[i, j]; title="$loc")
            @views scatterlines!(
                ax,
                ground_truth,
                label="ground truth",
                linewidth=3,
                color=(:black, 1.0),
                markercolor=(:black, 1.0),
                markersize=6,
            )

            for (t, model) in enumerate(uuids)
                select_output = if plot_type == :fit
                    (x -> endswith(x, ".fit.csv"))
                elseif plot_type == :pred
                    (x -> endswith(x, ".predictions.csv"))
                end
                fname_pred = first(
                    filter(select_output, readdir(joinpath(dir, model, loc)))
                )
                fpath_pred = joinpath(dir, model, loc, fname_pred)
                df_pred = CSV.read(fpath_pred, DataFrame)
                scatterlines!(
                    ax,
                    df_pred[!, label];
                    label=model,
                    linewidth=3,
                    color=(Makie.ColorSchemes.tab10[t], 0.7),
                    markercolor=(Makie.ColorSchemes.tab10[t], 0.7),
                    markersize=6,
                )
            end
            axislegend(ax; position=:lt, bgcolor=(:white, 0.7))
        end
    end

    return fig
end

runcmd(
    string.([
        "--beta_bounds",
        0.2 / 4,
        6.68 / 4,
        "--train_days=48",
        "--locations",
        Covid19ModelVN.LOC_CODE_VIETNAM,
        Covid19ModelVN.LOC_CODE_UNITED_STATES,
        keys(Covid19ModelVN.LOC_NAMES_VN)...,
        keys(Covid19ModelVN.LOC_NAMES_US)...,
        "--savedir=testsnapshots/thesis-results/tobeincluded",
        "--show_progress",
        "baseline",
        "eval",
        "--uuid=baseline",
    ]),
)

runcmd(
    string.([
        "--beta_bounds",
        0.2 / 4,
        6.68 / 4,
        "--train_days=48",
        "--locations",
        Covid19ModelVN.LOC_CODE_VIETNAM,
        Covid19ModelVN.LOC_CODE_UNITED_STATES,
        keys(Covid19ModelVN.LOC_NAMES_VN)...,
        keys(Covid19ModelVN.LOC_NAMES_US)...,
        "--savedir=testsnapshots/thesis-results/tobeincluded",
        "--show_progress",
        "fbmobility1",
        "eval",
        "--uuid=fb1",
    ]),
)

runcmd(
    string.([
        "--beta_bounds",
        0.2 / 4,
        6.68 / 4,
        "--train_days=48",
        "--locations",
        keys(Covid19ModelVN.LOC_NAMES_VN)...,
        keys(Covid19ModelVN.LOC_NAMES_US)...,
        "--savedir=testsnapshots/thesis-results/tobeincluded",
        "--show_progress",
        "fbmobility2",
        "eval",
        "--uuid=fb2",
    ]),
)

let
    fig = plot_outputs_combined(
        "testsnapshots/thesis-results/tobeincluded",
        ["baseline", "fb1"],
        [Covid19ModelVN.LOC_CODE_VIETNAM, Covid19ModelVN.LOC_CODE_UNITED_STATES],
        ["deaths", "new cases", "total cases"],
        :fit,
    )
    save("testsnapshots/thesis-results/tobeincluded/country_level_fit.png", fig)
end

let
    fig = plot_outputs_combined(
        "testsnapshots/thesis-results/tobeincluded",
        ["baseline", "fb1"],
        [Covid19ModelVN.LOC_CODE_VIETNAM, Covid19ModelVN.LOC_CODE_UNITED_STATES],
        ["deaths", "new cases", "total cases"],
        :pred,
    )
    save("testsnapshots/thesis-results/tobeincluded/country_level_pred.png", fig)
end

let
    fig = plot_outputs_combined(
        "testsnapshots/thesis-results/tobeincluded",
        ["baseline", "fb1", "fb2"],
        keys(Covid19ModelVN.LOC_NAMES_VN),
        ["deaths", "new cases", "total cases"],
        :fit,
    )
    save("testsnapshots/thesis-results/tobeincluded/vn_provinces_fit.png", fig)
end

let
    fig = plot_outputs_combined(
        "testsnapshots/thesis-results/tobeincluded",
        ["baseline", "fb1", "fb2"],
        keys(Covid19ModelVN.LOC_NAMES_VN),
        ["deaths", "new cases", "total cases"],
        :pred,
    )
    save("testsnapshots/thesis-results/tobeincluded/vn_provinces_pred.png", fig)
end

let
    fig = plot_outputs_combined(
        "testsnapshots/thesis-results/tobeincluded",
        ["baseline", "fb1", "fb2"],
        keys(Covid19ModelVN.LOC_NAMES_US),
        ["deaths", "new cases", "total cases"],
        :fit,
    )
    save("testsnapshots/thesis-results/tobeincluded/us_counties_fit.png", fig)
end

let
    fig = plot_outputs_combined(
        "testsnapshots/thesis-results/tobeincluded",
        ["baseline", "fb1", "fb2"],
        keys(Covid19ModelVN.LOC_NAMES_US),
        ["deaths", "new cases", "total cases"],
        :pred,
    )
    save("testsnapshots/thesis-results/tobeincluded/us_counties_pred.png", fig)
end

let
    dir = "testsnapshots/thesis-results/tobeincluded"
    model0 = "baseline"
    model1 = "fb1"
    model2 = "fb2"
    loc = "hcm"

    fname0 = first(
        filter(
            x -> endswith(x, ".time_steps_errors.csv"), readdir(joinpath(dir, model0, loc))
        ),
    )
    fpath0 = joinpath(dir, model0, loc, fname0)
    fname1 = first(
        filter(
            x -> endswith(x, ".time_steps_errors.csv"), readdir(joinpath(dir, model1, loc))
        ),
    )
    fpath1 = joinpath(dir, model1, loc, fname1)
    fname2 = first(
        filter(
            x -> endswith(x, ".time_steps_errors.csv"), readdir(joinpath(dir, model2, loc))
        ),
    )
    fpath2 = joinpath(dir, model2, loc, fname2)

    df0 = CSV.read(fpath0, DataFrame)
    df1 = CSV.read(fpath1, DataFrame)
    df2 = CSV.read(fpath2, DataFrame)
    group = [fill(1, nrow(df0)); fill(2, nrow(df1)); fill(3, nrow(df2))]
    fig = Figure()
    ax = Axis(fig[1, 1])
    barplot!(
        ax,
        [collect(1:28); collect(1:28); collect(1:28)],
        [df0[!, "new cases"]; df1[!, "new cases"]; df2[!, "new cases"]];
        dodge=group,
        color=Makie.ColorSchemes.tab10[group],
    )
    fig
end
