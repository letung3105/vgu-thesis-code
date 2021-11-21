include("include/cmd.jl")
include("include/trainalgs.jl")

let
    batch_timestamp = Dates.format(now(), "yyyymmddHHMMSS")
    model = "fbmobility4"
    locations = collect(union(keys(Covid19ModelVN.LOC_NAMES_US), keys(Covid19ModelVN.LOC_NAMES_VN)))

    for loc in locations
        timestamp = Dates.format(now(), "yyyymmddHHMMSS")
        uuid = "$timestamp.$model.$loc"

        parsed_args = parse_commandline([
            "--locations=$loc",
            "--alpha_bounds",
            "0.003",
            "0.05",
            "--",
            model
        ])
        _, gethyper, model_setup = setupcmd(parsed_args)
        hyperparams = gethyper(parsed_args)
        setup = () -> model_setup(loc, hyperparams)
        forecast_horizons = parsed_args[:forecast_horizons]

        snapshots_dir = joinpath("snapshots/growingtspan/$batch_timestamp", loc)
        !isdir(snapshots_dir) && mkpath(snapshots_dir)

        try
            growing_fit(
                uuid,
                setup;
                snapshots_dir,
                forecast_horizons,
                η = 1e-1,
                η_decay_rate = 0.5,
                η_decay_step = 100,
                η_limit = 1e-4,
                λ_weight_decay = 0,
                maxiters_initial = 200,
                maxiters_growth = 200,
                batch_initial = 8,
                batch_growth = 8,
                showprogress = true
            )
            experiment_eval(uuid, setup, forecast_horizons, snapshots_dir)
        catch e
            e isa InterruptException && rethrow(e)
            @warn e
        end
    end
end
