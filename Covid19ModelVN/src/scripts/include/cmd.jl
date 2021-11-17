using ArgParse

include("experiments.jl")

function runcmd(args)
    parsed_args = parse_commandline(args)
    name, get_hyperparams, setup = setupcmd(parsed_args)
    experiment_run(
        name,
        setup,
        parsed_args[:locations],
        get_hyperparams(parsed_args),
        get_train_configs(parsed_args),
        multithreading = parsed_args[:multithreading],
        savedir = parsed_args[:savedir],
        show_progress = parsed_args[:show_progress],
    )
end

function setupcmd(parsed_args)
    cmdmappings = Dict(
        [
            :baseline => ("baseline", get_baseline_hyperparams, setup_baseline)
            :fbmobility1 =>
                ("fbmobility1", get_fbmobility1_hyperparams, setup_fbmobility1)
            :fbmobility2 =>
                ("fbmobility2", get_fbmobility2_hyperparams, setup_fbmobility2)
            :fbmobility3 =>
                ("fbmobility3", get_fbmobility3_hyperparams, setup_fbmobility3)
            :fbmobility4 =>
                ("fbmobility4", get_fbmobility4_hyperparams, setup_fbmobility4)
        ],
    )
    if !haskey(cmdmappings, parsed_args[:_COMMAND_])
        error("Unsupported command '$(parse_args[:_COMMAND_])'")
    end
    return cmdmappings[parsed_args[:_COMMAND_]]
end

get_baseline_hyperparams(parsed_args) = (
    L2_λ = parsed_args[:L2_lambda],
    ζ = parsed_args[:zeta],
    γ0 = parsed_args[:gamma0],
    λ0 = parsed_args[:lambda0],
    α0 = parsed_args[:alpha0],
    γ_bounds = (parsed_args[:gamma_bounds][1], parsed_args[:gamma_bounds][2]),
    λ_bounds = (parsed_args[:lambda_bounds][1], parsed_args[:lambda_bounds][2]),
    α_bounds = (parsed_args[:alpha_bounds][1], parsed_args[:alpha_bounds][2]),
    train_range = Day(parsed_args[:train_days]),
    forecast_range = Day(parsed_args[:test_days]),
    ma7 = !parsed_args[:ma7_disable],
)

get_fbmobility1_hyperparams(parsed_args) = (
    L2_λ = parsed_args[:L2_lambda],
    ζ = parsed_args[:zeta],
    γ0 = parsed_args[:gamma0],
    λ0 = parsed_args[:lambda0],
    α0 = parsed_args[:alpha0],
    γ_bounds = (parsed_args[:gamma_bounds][1], parsed_args[:gamma_bounds][2]),
    λ_bounds = (parsed_args[:lambda_bounds][1], parsed_args[:lambda_bounds][2]),
    α_bounds = (parsed_args[:alpha_bounds][1], parsed_args[:alpha_bounds][2]),
    train_range = Day(parsed_args[:train_days]),
    forecast_range = Day(parsed_args[:test_days]),
    ma7 = !parsed_args[:ma7_disable],
)

get_fbmobility2_hyperparams(parsed_args) = (
    L2_λ = parsed_args[:L2_lambda],
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
    ma7 = !parsed_args[:ma7_disable],
)

get_fbmobility3_hyperparams(parsed_args) = (
    L2_λ = parsed_args[:L2_lambda],
    ζ = parsed_args[:zeta],
    γ0 = parsed_args[:gamma0],
    λ0 = parsed_args[:lambda0],
    α0 = parsed_args[:alpha0],
    β_bounds = (parsed_args[:beta_bounds][1], parsed_args[:beta_bounds][2]),
    γ_bounds = (parsed_args[:gamma_bounds][1], parsed_args[:gamma_bounds][2]),
    λ_bounds = (parsed_args[:lambda_bounds][1], parsed_args[:lambda_bounds][2]),
    α_bounds = (parsed_args[:alpha_bounds][1], parsed_args[:alpha_bounds][2]),
    train_range = Day(parsed_args[:train_days]),
    forecast_range = Day(parsed_args[:test_days]),
    social_proximity_lag = Day(parsed_args[:spc_lag_days]),
    ma7 = !parsed_args[:ma7_disable],
)

get_fbmobility4_hyperparams(parsed_args) = (
    L2_λ = parsed_args[:L2_lambda],
    ζ = parsed_args[:zeta],
    γ0 = parsed_args[:gamma0],
    λ0 = parsed_args[:lambda0],
    β_bounds = (parsed_args[:beta_bounds][1], parsed_args[:beta_bounds][2]),
    γ_bounds = (parsed_args[:gamma_bounds][1], parsed_args[:gamma_bounds][2]),
    λ_bounds = (parsed_args[:lambda_bounds][1], parsed_args[:lambda_bounds][2]),
    α_bounds = (parsed_args[:alpha_bounds][1], parsed_args[:alpha_bounds][2]),
    train_range = Day(parsed_args[:train_days]),
    forecast_range = Day(parsed_args[:test_days]),
    social_proximity_lag = Day(parsed_args[:spc_lag_days]),
    ma7 = !parsed_args[:ma7_disable],
)

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

        "--multithreading"
        help = "use multiple threads to train the model at multiple locations at once"
        action = :store_true

        "--savedir"
        help = "path to the directory where the model outputs are saved"
        arg_type = String
        default = "./snapshots"

        "--show_progress"
        help = "show a progress meter that keeps track of the training sessions"
        action = :store_true

        "--adam_maxiters"
        help = "max number of iterations used to run the ADAM optimizer"
        arg_type = Int
        default = 500

        "--adam_lr"
        help = "the learning rate given to the ADAM optimizer"
        arg_type = Float32
        default = 1f-2

        "--bfgs_maxiters"
        help = "max number of iterations used to run the BFGS optimizer"
        arg_type = Int
        default = 500

        "--bfgs_initial_stepnorm"
        help = "the initial_stepnorm given to the BFGS optimizer"
        arg_type = Float32
        default = 1f-2

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

        "--ma7_disable"
        help = "do not apply a 7-day moving average to all the time series datasets"
        action = :store_false

        "--L2_lambda"
        help = "L2-regularization term weight"
        arg_type = Float32
        default = 1f-4

        "--zeta"
        help = "loss function time weights"
        arg_type = Float32
        default = -5f-2

        "--gamma0"
        help = "inverse of the mean incubation period"
        arg_type = Float32
        default = 1.0f0 / 3.0f0

        "--lambda0"
        help = "inverse of the mean infectious period"
        arg_type = Float32
        default = 1.0f0 / 14.0f0

        "--alpha0"
        help = "the fatality rate"
        arg_type = Float32
        default = 2.5f-2

        "--beta_bounds"
        help = "lower and upper bounds contraints for the average contact rate"
        nargs = 2
        arg_type = Float32
        default = [0.0f0, 1.336f0] # ℜe ∈ [0.0; 6.68]

        "--gamma_bounds"
        help = "lower and upper bounds contraints for the inverse of the mean incubation period"
        nargs = 2
        arg_type = Float32
        default = [1.0f0 / 5.0f0, 1.0f0 / 2.0f0]

        "--lambda_bounds"
        help = "lower and upper bounds contraints for the inverse of the mean infectious period"
        nargs = 2
        arg_type = Float32
        default = [1.0f0 / 21.0f0, 1.0f0 / 7.0f0]

        "--alpha_bounds"
        help = "lower and upper bounds contraints for the inverse of the mean infectious period"
        nargs = 2
        arg_type = Float32
        default = [0.01f0, 0.06f0]
    end

    return parse_args(args, s, as_symbols = true)
end
