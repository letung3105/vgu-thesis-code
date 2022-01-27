include("include/cmd.jl")

using CSV: CSV
using CairoMakie
using Covid19ModelVN: Covid19ModelVN

function plot_combined_errors(dir, locs, models, metric, unit)
    fig = Figure(; resolution=(800, 400 * ((length(locs) - 1) ÷ 2 + 1)))
    for (i, loc) in enumerate(locs)
        deaths = Float64[]
        new_cases = Float64[]
        total_cases = Float64[]

        for model in models
            fdir = joinpath(dir, model, loc)
            files = filter!(x -> endswith(x, ".errors.csv"), readdir(fdir))
            if isempty(files)
                continue
            end

            fpath = joinpath(fdir, first(files))
            df = CSV.read(fpath, DataFrame)
            filter!(x -> x.horizon == 28 && x.metric == metric, df)

            push!(deaths, first(df[!, "deaths"]))
            push!(new_cases, first(df[!, "new cases"]))
            push!(total_cases, first(df[!, "total cases"]))
        end

        @assert length(deaths) == length(new_cases)
        @assert length(total_cases) == length(new_cases)

        dodge_group = repeat(1:length(deaths), 3)
        x = vcat([fill(i, length(deaths)) for i in 1:3]...)
        y = [deaths; new_cases; total_cases]
        ax = Axis(
            fig[(i - 1) ÷ 2 + 1, (i - 1) % 2 + 1];
            title=loc,
            xticks=(1:3, ["deaths", "new cases", "total cases"]),
            ylabel="$metric ($unit)",
        )

        barplot!(
            ax,
            x,
            y;
            color=map(x -> Makie.ColorSchemes.tab10[x], dodge_group),
            dodge=dodge_group,
        )

        Legend(
            fig[(i - 1) ÷ 2 + 1, (i - 1) % 2 + 1],
            [
                PolyElement(; polycolor=Makie.ColorSchemes.tab10[i]) for
                i in 1:length(models)
            ],
            models;
            tellheight=false,
            tellwidth=false,
            orientation=:horizontal,
            halign=:left,
            valign=:top,
            margin=(10, 10, 10, 10),
            bgcolor=(:white, 0.8),
        )
    end

    return fig
end

let
    dir = "testsnapshots/thesis-results/tobeincluded"
    metric = "mape"
    unit = "%"

    fig_country = plot_combined_errors(
        dir, ["vietnam", "unitedstates"], ["baseline", "fb1"], metric, unit
    )
    fig_us = plot_combined_errors(
        dir, keys(Covid19ModelVN.LOC_NAMES_US), ["baseline", "fb1", "fb2"], metric, unit
    )
    fig_vn = plot_combined_errors(
        dir, keys(Covid19ModelVN.LOC_NAMES_VN), ["baseline", "fb1", "fb2"], metric, unit
    )
    save(joinpath(dir, "errors_country.pdf"), fig_country)
    save(joinpath(dir, "errors_us.pdf"), fig_us)
    save(joinpath(dir, "errors_vn.pdf"), fig_vn)
end
