include("include/cmd.jl")

using CSV: CSV
using CairoMakie
import Covid19ModelVN

function plot_outputs_combined(
    dir, uuids, locs, labels, plot_type; train_range=Day(48), forecast_range=Day(28)
)
    MARKERS = [:rect, :utriangle, :dtriangle]
    fig = Figure(; resolution=(400 * length(locs), 400 * length(labels)))
    for (i, label) in enumerate(labels)
        Label(
            fig[i, 1:length(locs), Bottom()],
            label;
            valign=:bottom,
            padding=(0, 0, 0, 32),
            textsize=24,
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
                linewidth=2,
                color=(:black, 1.0),
                markercolor=(:black, 1.0),
                markerspace=8,
                markersize=8,
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
                    linewidth=2,
                    color=(Makie.ColorSchemes.tab10[t], 0.7),
                    marker=MARKERS[t],
                    markercolor=(Makie.ColorSchemes.tab10[t], 0.7),
                    markerspace=12,
                    markersize=12,
                )
            end
            axislegend(ax; position=:lt, bgcolor=(:white, 0.7))
        end
    end

    return fig
end

function plot_Re_and_fatality_combined(dir, uuids, loc) where {R<:Real}
    MARKERS = [:rect, :utriangle, :dtriangle]
    fig = Figure(; resolution=(400 * 2, 400))
    ax1 = Axis(
        fig[1, 1];
        xlabel="Days since the 500th confirmed case",
        ylabel="Effective reproduction number",
    )
    ax2 = Axis(
        fig[1, 2]; xlabel="Days since the 500th confirmed case", ylabel="Fatality rate (%)"
    )

    hlines!(ax1, [1]; color=:green, linestyle=:dash, linewidth=3, label="threshold")

    for (t, model) in enumerate(uuids)
        fname_Re = first(
            filter(x -> endswith(x, ".R_effective.csv"), readdir(joinpath(dir, model, loc)))
        )
        fpath_Re = joinpath(dir, model, loc, fname_Re)
        df_Re = CSV.read(fpath_Re, DataFrame)

        scatterlines!(
            ax1,
            df_Re[!, :Re];
            label=model,
            linewidth=2,
            color=(Makie.ColorSchemes.tab10[t], 0.7),
            marker=MARKERS[t],
            markercolor=(Makie.ColorSchemes.tab10[t], 0.7),
            markerspace=12,
            markersize=12,
        )

        fname_αt = first(
            filter(
                x -> endswith(x, ".fatality_rate.csv"), readdir(joinpath(dir, model, loc))
            ),
        )
        fpath_αt = joinpath(dir, model, loc, fname_αt)
        df_αt = CSV.read(fpath_αt, DataFrame)

        scatterlines!(
            ax2,
            df_αt[!, :αt];
            label=model,
            linewidth=2,
            color=(Makie.ColorSchemes.tab10[t], 0.7),
            marker=MARKERS[t],
            markercolor=(Makie.ColorSchemes.tab10[t], 0.7),
            markerspace=12,
            markersize=12,
        )
    end

    axislegend(ax1; position=:rt, bgcolor=(:white, 0.7))
    axislegend(ax2; position=:rt, bgcolor=(:white, 0.7))

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
    save("testsnapshots/thesis-results/tobeincluded/fit_country_level.pdf", fig)
end

let
    fig = plot_outputs_combined(
        "testsnapshots/thesis-results/tobeincluded",
        ["baseline", "fb1"],
        [Covid19ModelVN.LOC_CODE_VIETNAM, Covid19ModelVN.LOC_CODE_UNITED_STATES],
        ["deaths", "new cases", "total cases"],
        :pred,
    )
    save("testsnapshots/thesis-results/tobeincluded/pred_country_level.pdf", fig)
end

let
    locations = collect(keys(Covid19ModelVN.LOC_NAMES_VN))
    fig = plot_outputs_combined(
        "testsnapshots/thesis-results/tobeincluded",
        ["baseline", "fb1", "fb2"],
        locations[1:2],
        ["deaths", "new cases", "total cases"],
        :fit,
    )
    save("testsnapshots/thesis-results/tobeincluded/fit_vn_provinces1.pdf", fig)
    fig = plot_outputs_combined(
        "testsnapshots/thesis-results/tobeincluded",
        ["baseline", "fb1", "fb2"],
        locations[3:4],
        ["deaths", "new cases", "total cases"],
        :fit,
    )
    save("testsnapshots/thesis-results/tobeincluded/fit_vn_provinces2.pdf", fig)
end

let
    locations = collect(keys(Covid19ModelVN.LOC_NAMES_VN))
    fig = plot_outputs_combined(
        "testsnapshots/thesis-results/tobeincluded",
        ["baseline", "fb1", "fb2"],
        locations[1:2],
        ["deaths", "new cases", "total cases"],
        :pred,
    )
    save("testsnapshots/thesis-results/tobeincluded/pred_vn_provinces1.pdf", fig)
    fig = plot_outputs_combined(
        "testsnapshots/thesis-results/tobeincluded",
        ["baseline", "fb1", "fb2"],
        locations[3:4],
        ["deaths", "new cases", "total cases"],
        :pred,
    )
    save("testsnapshots/thesis-results/tobeincluded/pred_vn_provinces2.pdf", fig)
end

let
    locations = collect(keys(Covid19ModelVN.LOC_NAMES_US))
    fig = plot_outputs_combined(
        "testsnapshots/thesis-results/tobeincluded",
        ["baseline", "fb1", "fb2"],
        locations[1:2],
        ["deaths", "new cases", "total cases"],
        :fit,
    )
    save("testsnapshots/thesis-results/tobeincluded/fit_us_counties1.pdf", fig)
    fig = plot_outputs_combined(
        "testsnapshots/thesis-results/tobeincluded",
        ["baseline", "fb1", "fb2"],
        locations[3:4],
        ["deaths", "new cases", "total cases"],
        :fit,
    )
    save("testsnapshots/thesis-results/tobeincluded/fit_us_counties2.pdf", fig)
end

let
    locations = collect(keys(Covid19ModelVN.LOC_NAMES_US))
    fig = plot_outputs_combined(
        "testsnapshots/thesis-results/tobeincluded",
        ["baseline", "fb1", "fb2"],
        locations[1:2],
        ["deaths", "new cases", "total cases"],
        :pred,
    )
    save("testsnapshots/thesis-results/tobeincluded/pred_us_counties1.pdf", fig)
    fig = plot_outputs_combined(
        "testsnapshots/thesis-results/tobeincluded",
        ["baseline", "fb1", "fb2"],
        locations[3:4],
        ["deaths", "new cases", "total cases"],
        :pred,
    )
    save("testsnapshots/thesis-results/tobeincluded/pred_us_counties2.pdf", fig)
end

let
    for loc in [keys(Covid19ModelVN.LOC_NAMES_VN)..., keys(Covid19ModelVN.LOC_NAMES_US)...]
        fig = plot_Re_and_fatality_combined(
            "testsnapshots/thesis-results/tobeincluded", ["baseline", "fb1", "fb2"], loc
        )
        save("testsnapshots/thesis-results/tobeincluded/Re_and_fatality_$loc.pdf", fig)
    end
end

let
    for loc in [Covid19ModelVN.LOC_CODE_VIETNAM, Covid19ModelVN.LOC_CODE_UNITED_STATES]
        fig = plot_Re_and_fatality_combined(
            "testsnapshots/thesis-results/tobeincluded", ["baseline", "fb1"], loc
        )
        save("testsnapshots/thesis-results/tobeincluded/Re_and_fatality_$loc.pdf", fig)
    end
end