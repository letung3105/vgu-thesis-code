using ArgParse

include("experiments.jl")

function runcmd(args)
    parsed_args = parse_commandline(args)
    model_name, get_hyperparams, model_setup = setupcmd(parsed_args)
    if (parsed_args[:_COMMAND_] == :eval)
        uuid = parsed_args[:eval][:uuid]
        hyperparams = get_hyperparams(parsed_args)
        for loc in parsed_args[:locations]
            snapshots_dir = joinpath(parsed_args[:savedir], uuid, loc)
            setup = () -> model_setup(loc; hyperparams...)
            experiment_eval(setup, parsed_args[:forecast_horizons], snapshots_dir)
        end
    else
        experiment_run(
            model_name,
            model_setup,
            parsed_args[:locations],
            get_hyperparams(parsed_args),
            parsed_args[:_COMMAND_],
            parsed_args[parsed_args[:_COMMAND_]];
            forecast_horizons=parsed_args[:forecast_horizons],
            savedir=parsed_args[:savedir],
            show_progress=parsed_args[:show_progress],
            make_animation=parsed_args[:make_animation],
            multithreading=parsed_args[:multithreading],
        )
    end
end

function setupcmd(parsed_args)
    cmdmappings = Dict(
        :baseline => ("baseline", get_baseline_hyperparams, setup_baseline),
        :fbmobility1 => ("fbmobility1", get_fbmobility1_hyperparams, setup_fbmobility1),
        :fbmobility2 => ("fbmobility2", get_fbmobility2_hyperparams, setup_fbmobility2),
    )
    return cmdmappings[parsed_args[:model_name]]
end

function get_baseline_hyperparams(parsed_args)
    return (
        γ0=parsed_args[:gamma0],
        λ0=parsed_args[:lambda0],
        β_bounds=(parsed_args[:beta_bounds][1], parsed_args[:beta_bounds][2]),
        γ_bounds=(parsed_args[:gamma_bounds][1], parsed_args[:gamma_bounds][2]),
        λ_bounds=(parsed_args[:lambda_bounds][1], parsed_args[:lambda_bounds][2]),
        α_bounds=(parsed_args[:alpha_bounds][1], parsed_args[:alpha_bounds][2]),
        train_range=Day(parsed_args[:train_days]),
        forecast_range=Day(parsed_args[:test_days]),
        loss_type=parsed_args[:loss_type],
        loss_regularization=parsed_args[:loss_regularization],
        loss_time_weighting=parsed_args[:loss_time_weighting],
    )
end

function get_fbmobility1_hyperparams(parsed_args)
    return (
        γ0=parsed_args[:gamma0],
        λ0=parsed_args[:lambda0],
        β_bounds=(parsed_args[:beta_bounds][1], parsed_args[:beta_bounds][2]),
        γ_bounds=(parsed_args[:gamma_bounds][1], parsed_args[:gamma_bounds][2]),
        λ_bounds=(parsed_args[:lambda_bounds][1], parsed_args[:lambda_bounds][2]),
        α_bounds=(parsed_args[:alpha_bounds][1], parsed_args[:alpha_bounds][2]),
        train_range=Day(parsed_args[:train_days]),
        forecast_range=Day(parsed_args[:test_days]),
        movement_range_lag=Day(parsed_args[:movement_range_lag_days]),
        loss_type=parsed_args[:loss_type],
        loss_regularization=parsed_args[:loss_regularization],
        loss_time_weighting=parsed_args[:loss_time_weighting],
    )
end

function get_fbmobility2_hyperparams(parsed_args)
    return (
        γ0=parsed_args[:gamma0],
        λ0=parsed_args[:lambda0],
        β_bounds=(parsed_args[:beta_bounds][1], parsed_args[:beta_bounds][2]),
        γ_bounds=(parsed_args[:gamma_bounds][1], parsed_args[:gamma_bounds][2]),
        λ_bounds=(parsed_args[:lambda_bounds][1], parsed_args[:lambda_bounds][2]),
        α_bounds=(parsed_args[:alpha_bounds][1], parsed_args[:alpha_bounds][2]),
        train_range=Day(parsed_args[:train_days]),
        forecast_range=Day(parsed_args[:test_days]),
        movement_range_lag=Day(parsed_args[:movement_range_lag_days]),
        social_proximity_lag=Day(parsed_args[:social_proximity_lag_days]),
        loss_type=parsed_args[:loss_type],
        loss_regularization=parsed_args[:loss_regularization],
        loss_time_weighting=parsed_args[:loss_time_weighting],
    )
end

function parse_commandline(args)
    s = ArgParseSettings()

    isvalidmodel(name) = name == :baseline || name == :fbmobility1 || name == :fbmobility2

    isvalidloss(name) = name == :polar || name == :sse

    @add_arg_table s begin
        "model_name"
        help = "name of the model that will be used"
        arg_type = Symbol
        range_tester = isvalidmodel
        required = true

        "eval"
        help = "draw plot and make tables for evaluating the model"
        action = :command

        "train_growing_trajectory"
        help = "train the model by iteratively growing time span"
        action = :command

        "train_growing_trajectory_two_stages"
        help = "train the model by iteratively growing time span, then use LBFGS"
        action = :command

        "train_whole_trajectory"
        help = "train the model on the whole time span"
        action = :command

        "train_whole_trajectory_two_stages"
        help = "train the model on the whole time span, then use LBFGS"
        action = :command

        "--locations"
        help = "the code of the locations whose data will be used to train the model"
        action = :store_arg
        nargs = '*'
        arg_type = String
        default = String[]

        "--forecast_horizons"
        help = "the numbers of days that will be forecasted"
        action = :store_arg
        nargs = '*'
        arg_type = Int32
        default = Int32[7, 14, 21, 28]

        "--savedir"
        help = "path to the directory where the model outputs are saved"
        arg_type = String
        default = "./snapshots"

        "--multithreading"
        help = "use multiple threads to train the model at multiple locations at once"
        action = :store_true

        "--show_progress"
        help = "show a progress meter that keeps track of the training sessions"
        action = :store_true

        "--make_animation"
        help = "show a progress meter that keeps track of the training sessions"
        action = :store_true

        "--train_days"
        help = "number of days used for training"
        arg_type = Int
        default = 32

        "--test_days"
        help = "number of days used for testing"
        arg_type = Int
        default = 28

        "--movement_range_lag_days"
        help = "number of lag days that is used when reading the Movement Range Maps dataset"
        arg_type = Int
        default = 0 # no lags

        "--social_proximity_lag_days"
        help = "number of lag days that is used when reading the Social Proximity to Cases index"
        arg_type = Int
        default = 0 # no lags

        "--gamma0"
        help = "inverse of the mean incubation period"
        arg_type = Float64
        default = 1.0 / 4.0 # 4 days incubation period

        "--lambda0"
        help = "inverse of the mean infectious period"
        arg_type = Float64
        default = 1.0 / 14.0 # 14 days infectious period

        "--beta_bounds"
        help = "lower and upper bounds contraints for the average contact rate"
        nargs = 2
        arg_type = Float64
        default = [0.0, 6.68 / 4] # Re ∈ [0.0; 6.68]

        "--gamma_bounds"
        help = "lower and upper bounds contraints for the inverse of the mean incubation period"
        nargs = 2
        arg_type = Float64
        default = [1.0 / 4.0, 1.0 / 4.0] # keep this constant

        "--lambda_bounds"
        help = "lower and upper bounds contraints for the inverse of the mean infectious period"
        nargs = 2
        arg_type = Float64
        default = [1.0 / 14.0, 1.0 / 14.0] # keep this constant

        "--alpha_bounds"
        help = "lower and upper bounds contraints for the inverse of the mean infectious period"
        nargs = 2
        arg_type = Float64
        default = [5e-3, 5e-2] # fatality 0.5% - 5%

        "--loss_type"
        help = "choose the loss function that is used for training"
        arg_type = Symbol
        range_tester = isvalidloss
        default = :sse

        "--loss_regularization"
        help = "scaling factor for the weight decay term"
        arg_type = Float64
        default = 0.0 # no regularization

        "--loss_time_weighting"
        help = "scaling factor for the time scaling"
        arg_type = Float64
        default = 0.0 # time time weights
    end

    @add_arg_table s["eval"] begin
        "--uuid"
        help = "the unique identifer to the model training session"
        arg_type = String
    end

    @add_arg_table s["train_growing_trajectory"] begin
        "--lr"
        help = "learning rate to be given to ADAM"
        arg_type = Float64

        "--lr_decay_rate"
        help = "learning rate exponential decay rate"
        arg_type = Float64

        "--lr_decay_step"
        help = "number of iterations taken before decaying the learning rate"
        arg_type = Int

        "--lr_limit"
        help = "the minimum value at which learning rate decay is stopped"
        arg_type = Float64

        "--maxiters_initial"
        help = "the max number of iterations used for fiting the first time span"
        arg_type = Int

        "--maxiters_growth"
        help = "increase the max number of iterations by a fixed amount when growing the time span"
        arg_type = Int

        "--tspan_size_initial"
        help = "number of data points in the initial time span"
        arg_type = Int

        "--tspan_size_growth"
        help = "number of new data points taken when growing the time span"
        arg_type = Int
    end

    @add_arg_table s["train_growing_trajectory_two_stages"] begin
        "--lr"
        help = "learning rate to be given to ADAM"
        arg_type = Float64

        "--lr_decay_rate"
        help = "learning rate exponential decay rate"
        arg_type = Float64

        "--lr_decay_step"
        help = "number of iterations taken before decaying the learning rate"
        arg_type = Int

        "--lr_limit"
        help = "the minimum value at which learning rate decay is stopped"
        arg_type = Float64

        "--maxiters_initial"
        help = "the max number of iterations used for fiting the first time span"
        arg_type = Int

        "--maxiters_growth"
        help = "increase the max number of iterations by a fixed amount when growing the time span"
        arg_type = Int

        "--maxiters_second"
        help = "max number of interations used for the second stage"
        arg_type = Int

        "--tspan_size_initial"
        help = "number of data points in the initial time span"
        arg_type = Int

        "--tspan_size_growth"
        help = "number of new data points taken when growing the time span"
        arg_type = Int
    end

    @add_arg_table s["train_whole_trajectory"] begin
        "--lr"
        help = "learning rate to be given to ADAM"
        arg_type = Float64

        "--lr_decay_rate"
        help = "learning rate exponential decay rate"
        arg_type = Float64

        "--lr_decay_step"
        help = "number of iterations taken before decaying the learning rate"
        arg_type = Int

        "--lr_limit"
        help = "the minimum value at which learning rate decay is stopped"
        arg_type = Float64

        "--maxiters"
        help = "the max number of iterations used"
        arg_type = Int

        "--minibatching"
        help = "size of the minibatch used when training, 0 means no minibatching"
        arg_type = Int
    end

    @add_arg_table s["train_whole_trajectory_two_stages"] begin
        "--lr"
        help = "learning rate to be given to ADAM"
        arg_type = Float64

        "--lr_decay_rate"
        help = "learning rate exponential decay rate"
        arg_type = Float64

        "--lr_decay_step"
        help = "number of iterations taken before decaying the learning rate"
        arg_type = Int

        "--lr_limit"
        help = "the minimum value at which learning rate decay is stopped"
        arg_type = Float64

        "--maxiters_first"
        help = "the max number of iterations used in the first stage"
        arg_type = Int

        "--maxiters_second"
        help = "the max number of iterations used in the second stage"
        arg_type = Int

        "--minibatching"
        help = "size of the minibatch used when training in the first stage, 0 means no minibatching"
        arg_type = Int
    end

    return parse_args(args, s; as_symbols=true)
end
