using ArgParse

include("experiments.jl")

function runcmd(args = ARGS)
    parsed_args = parse_commandline(args)
    if parsed_args[:_COMMAND_] == :baseline
        cmd_baseline(parsed_args)
    elseif parsed_args[:_COMMAND_] == :fbmobility1
        cmd_fbmobility1(parsed_args)
    elseif parsed_args[:_COMMAND_] == :fbmobility2
        cmd_fbmobility2(parsed_args)
    elseif parsed_args[:_COMMAND_] == :fbmobility3
        cmd_fbmobility3(parsed_args)
    elseif parsed_args[:_COMMAND_] == :fbmobility4
        cmd_fbmobility4(parsed_args)
    else
        @error "Unsupported command '$(parse_args[:_COMMAND_])'"
    end
end

function cmd_baseline(parsed_args)
    experiment_run(
        parsed_args[:name],
        setup_baseline,
        parsed_args[:locations],
        (
            ζ = parsed_args[:zeta],
            γ0 = parsed_args[:gamma0],
            λ0 = parsed_args[:lambda0],
            α0 = parsed_args[:alpha0],
            γ_bounds = (parsed_args[:gamma_bounds][1], parsed_args[:gamma_bounds][2]),
            λ_bounds = (parsed_args[:lambda_bounds][1], parsed_args[:lambda_bounds][2]),
            α_bounds = (parsed_args[:alpha_bounds][1], parsed_args[:alpha_bounds][2]),
            train_range = Day(parsed_args[:train_days]),
            forecast_range = Day(parsed_args[:test_days]),
        ),
        get_train_configs(parsed_args),
        savedir = parsed_args[:savedir],
    )
end

function cmd_fbmobility1(parsed_args)
    experiment_run(
        parsed_args[:name],
        setup_fbmobility1,
        parsed_args[:locations],
        (
            ζ = parsed_args[:zeta],
            γ0 = parsed_args[:gamma0],
            λ0 = parsed_args[:lambda0],
            α0 = parsed_args[:alpha0],
            γ_bounds = (parsed_args[:gamma_bounds][1], parsed_args[:gamma_bounds][2]),
            λ_bounds = (parsed_args[:lambda_bounds][1], parsed_args[:lambda_bounds][2]),
            α_bounds = (parsed_args[:alpha_bounds][1], parsed_args[:alpha_bounds][2]),
            train_range = Day(parsed_args[:train_days]),
            forecast_range = Day(parsed_args[:test_days]),
        ),
        get_train_configs(parsed_args),
        savedir = parsed_args[:savedir],
    )
end

function cmd_fbmobility2(parsed_args)
    experiment_run(
        parsed_args[:name],
        setup_fbmobility2,
        parsed_args[:locations],
        (
            ζ = parsed_args[:zeta],
            γ0 = parsed_args[:gamma0],
            λ0 = parsed_args[:lambda0],
            α0 = parsed_args[:alpha0],
            γ_bounds = (parsed_args[:gamma_bounds][1], parsed_args[:gamma_bounds][2]),
            λ_bounds = (parsed_args[:lambda_bounds][1], parsed_args[:lambda_bounds][2]),
            α_bounds = (parsed_args[:alpha_bounds][1], parsed_args[:alpha_bounds][2]),
            train_range = Day(parsed_args[:train_days]),
            forecast_range = Day(parsed_args[:test_days]),
            social_proximity_lag = Day(parsed_args[:spc_lag_days]),
        ),
        get_train_configs(parsed_args),
        savedir = parsed_args[:savedir],
    )
end

function cmd_fbmobility3(parsed_args)
    experiment_run(
        parsed_args[:name],
        setup_fbmobility3,
        parsed_args[:locations],
        (
            ζ = parsed_args[:zeta],
            γ0 = parsed_args[:gamma0],
            λ0 = parsed_args[:lambda0],
            γ_bounds = (parsed_args[:gamma_bounds][1], parsed_args[:gamma_bounds][2]),
            λ_bounds = (parsed_args[:lambda_bounds][1], parsed_args[:lambda_bounds][2]),
            α_bounds = (parsed_args[:alpha_bounds][1], parsed_args[:alpha_bounds][2]),
            train_range = Day(parsed_args[:train_days]),
            forecast_range = Day(parsed_args[:test_days]),
            social_proximity_lag = Day(parsed_args[:spc_lag_days]),
        ),
        get_train_configs(parsed_args),
        savedir = parsed_args[:savedir],
    )
end

function cmd_fbmobility4(parsed_args)
    experiment_run(
        parsed_args[:name],
        setup_fbmobility4,
        parsed_args[:locations],
        (
            ζ = parsed_args[:zeta],
            γ0 = parsed_args[:gamma0],
            λ0 = parsed_args[:lambda0],
            γ_bounds = (parsed_args[:gamma_bounds][1], parsed_args[:gamma_bounds][2]),
            λ_bounds = (parsed_args[:lambda_bounds][1], parsed_args[:lambda_bounds][2]),
            α_bounds = (parsed_args[:alpha_bounds][1], parsed_args[:alpha_bounds][2]),
            train_range = Day(parsed_args[:train_days]),
            forecast_range = Day(parsed_args[:test_days]),
            social_proximity_lag = Day(parsed_args[:spc_lag_days]),
        ),
        get_train_configs(parsed_args),
        savedir = parsed_args[:savedir],
    )
end

function get_train_configs(parsed_args)
    train_configs = TrainConfig[]
    if parsed_args[:adam_maxiters] > 0
        push!(
            train_configs,
            TrainConfig("ADAM", ADAM(parsed_args[:adam_lr]), parsed_args[:adam_maxiters]),
        )
    end
    if parsed_args[:bfgs_maxiters] > 0
        push!(
            train_configs,
            TrainConfig(
                "BFGS",
                BFGS(initial_stepnorm = parsed_args[:bfgs_initial_stepnorm]),
                parsed_args[:bfgs_maxiters],
            ),
        )
    end
    return train_configs
end

function parse_commandline(args)
    s = ArgParseSettings()

    @add_arg_table s begin
        "name"
        help = "name of the training and evalution session"
        arg_type = String
        required = true

        "baseline"
        help = "train and inference with the baseline model"
        action = :command

        "fbmobility1"
        help = "train and inference with the fbmobility1 model"
        action = :command

        "fbmobility2"
        help = "train and inference with the fbmobility2 model"
        action = :command

        "fbmobility3"
        help = "train and inference with the fbmobility3 model"
        action = :command

        "fbmobility4"
        help = "train and inference with the fbmobility4 model"
        action = :command

        "--locations"
        help = "the code of the locations whose data will be used to train the model"
        action = :store_arg
        nargs = '*'
        arg_type = String
        default = String[]

        "--savedir"
        help = "path to the directory where the model outputs are saved"
        arg_type = String
        default = "./snapshots"

        "--adam_maxiters"
        help = "max number of iterations used to run the ADAM optimizer"
        arg_type = Int
        default = 500

        "--adam_lr"
        help = "the learning rate given to the ADAM optimizer"
        arg_type = Float64
        default = 0.01

        "--bfgs_maxiters"
        help = "max number of iterations used to run the BFGS optimizer"
        arg_type = Int
        default = 500

        "--bfgs_initial_stepnorm"
        help = "the initial_stepnorm given to the BFGS optimizer"
        arg_type = Float64
        default = 0.01

        "--train_days"
        help = "number of days used for training"
        arg_type = Int
        default = 32

        "--test_days"
        help = "number of days used for testing"
        arg_type = Int
        default = 28

        "--spc_lag_days"
        help = "number of lag days that is used when reading the Social Proximity to Cases index"
        arg_type = Int
        default = 14

        "--zeta"
        help = "loss function time weights"
        arg_type = Float64
        default = 0.01

        "--gamma0"
        help = "inverse of the mean incubation period"
        arg_type = Float64
        default = 1 / 3

        "--lambda0"
        help = "inverse of the mean infectious period"
        arg_type = Float64
        default = 1 / 14

        "--alpha0"
        help = "the fatality rate"
        arg_type = Float64
        default = 0.025

        "--gamma_bounds"
        help = "lower and upper bounds contraints for the inverse of the mean incubation period"
        nargs = 2
        arg_type = Float64
        default = [1 / 5, 1 / 2]

        "--lambda_bounds"
        help = "lower and upper bounds contraints for the inverse of the mean infectious period"
        nargs = 2
        arg_type = Float64
        default = [1 / 21, 1 / 7]

        "--alpha_bounds"
        help = "lower and upper bounds contraints for the inverse of the mean infectious period"
        nargs = 2
        arg_type = Float64
        default = [0.01, 0.06]
    end

    return parse_args(args, s, as_symbols = true)
end
