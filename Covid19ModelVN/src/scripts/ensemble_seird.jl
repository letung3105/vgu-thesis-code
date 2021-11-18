using BenchmarkTools
using Distributions
using OrdinaryDiffEq.EnsembleAnalysis

include("include/experiments.jl")

let
    loc = "cook_il"
    train_dataset, test_dataset, first_date, last_date =
        experiment_covid19_data(loc, Day(32), Day(28), true)
    u0, vars, labels = experiment_SEIRD_initial_states(loc, train_dataset.data[:, 1])

    βdist = Uniform(0.0, 0.37)
    γdist = truncated(Normal(1 / 3), 1 / 5, 1 / 2)
    λdist = truncated(Normal(1 / 14), 1 / 21, 1 / 7)
    αdist = truncated(Normal(0.025), 0.0, 0.05)
    function prob_func(prob, i, repeat)
        p = [rand(βdist), rand(γdist), rand(λdist), rand(αdist)]
        remake(prob, p = p)
    end

    prob = ODEProblem(Covid19ModelVN.SEIRD!, u0, test_dataset.tspan)
    ensemble_prob = EnsembleProblem(prob, prob_func = prob_func)
    sim = solve(
        ensemble_prob,
        Tsit5(),
        EnsembleThreads(),
        trajectories = 10000,
        saveat = 0:59,
        save_idxs = [5, 6],
    )

    summary = EnsembleSummary(sim)
    @show summary.num_monte

    sz = size(summary.u)
    fig = Figure(resolution = (600, 400 * sz[1]))
    for state_id = 1:sz[1]
        ax = Axis(fig[state_id, 1])
        bnd = band!(
            ax,
            summary.t,
            summary.qlow[state_id, :],
            summary.qhigh[state_id, :],
            color = (Makie.ColorSchemes.tab10[2], 0.5),
        )
        ln1 = lines!(
            ax,
            summary.t,
            summary.u[state_id, :],
            linewidth = 3,
            color = Makie.ColorSchemes.tab10[2],
        )
        ln2 = lines!(
            ax,
            [train_dataset.data[state_id, :]; test_dataset.data[state_id, :]],
            linewidth = 3,
            color = Makie.ColorSchemes.tab10[1],
        )
        Legend(
            fig[state_id, 1],
            [bnd, ln1, ln2],
            ["model 95% quantile", "model mean", "ground truth"],
            margin = (5, 5, 5, 5),
            tellwidth = false,
            tellheight = false,
            halign = :left,
            valign = :top,
        )
    end
    display(fig)
end