using ArgParse

include("experiments.jl")

function runcmd(args)
    parsed_args = parse_commandline(args)
    model_name, get_hyperparams, model_setup = setupcmd(parsed_args)
    experiment_run(
        model_name,
        model_setup,
        parsed_args[:locations],
        get_hyperparams(parsed_args),
        parsed_args[:_COMMAND_],
        parsed_args[parsed_args[:_COMMAND_]],
        forecast_horizons = parsed_args[:forecast_horizons],
        savedir = parsed_args[:savedir],
        show_progress = parsed_args[:show_progress],
        make_animation = parsed_args[:make_animation],
        multithreading = parsed_args[:multithreading],
    )
end

function setupcmd(parsed_args)
    cmdmappings = Dict(
        "baseline" => ("baseline", get_baseline_hyperparams, setup_baseline),
        "fbmobility1" =>
            ("fbmobility1", get_fbmobility1_hyperparams, setup_fbmobility1),
        "fbmobility2" =>
            ("fbmobility2", get_fbmobility2_hyperparams, setup_fbmobility2),
        "fbmobility3" =>
            ("fbmobility3", get_fbmobility3_hyperparams, setup_fbmobility3),
        "fbmobility4" =>
            ("fbmobility4", get_fbmobility4_hyperparams, setup_fbmobility4),
    )
    return cmdmappings[parsed_args[:model_name]]
end

get_baseline_hyperparams(parsed_args) = (
    γ0 = parsed_args[:gamma0],
    λ0 = parsed_args[:lambda0],
    α0 = parsed_args[:alpha0],
    γ_bounds = (parsed_args[:gamma_bounds][1], parsed_args[:gamma_bounds][2]),
    λ_bounds = (parsed_args[:lambda_bounds][1], parsed_args[:lambda_bounds][2]),
    α_bounds = (parsed_args[:alpha_bounds][1], parsed_args[:alpha_bounds][2]),
    train_range = Day(parsed_args[:train_days]),
    forecast_range = Day(parsed_args[:test_days]),
)

get_fbmobility1_hyperparams(parsed_args) = (
    γ0 = parsed_args[:gamma0],
    λ0 = parsed_args[:lambda0],
    α0 = parsed_args[:alpha0],
    γ_bounds = (parsed_args[:gamma_bounds][1], parsed_args[:gamma_bounds][2]),
    λ_bounds = (parsed_args[:lambda_bounds][1], parsed_args[:lambda_bounds][2]),
    α_bounds = (parsed_args[:alpha_bounds][1], parsed_args[:alpha_bounds][2]),
    train_range = Day(parsed_args[:train_days]),
    forecast_range = Day(parsed_args[:test_days]),
)

get_fbmobility2_hyperparams(parsed_args) = (
    γ0 = parsed_args[:gamma0],
    λ0 = parsed_args[:lambda0],
    α0 = parsed_args[:alpha0],
    γ_bounds = (parsed_args[:gamma_bounds][1], parsed_args[:gamma_bounds][2]),
    λ_bounds = (parsed_args[:lambda_bounds][1], parsed_args[:lambda_bounds][2]),
    α_bounds = (parsed_args[:alpha_bounds][1], parsed_args[:alpha_bounds][2]),
    train_range = Day(parsed_args[:train_days]),
    forecast_range = Day(parsed_args[:test_days]),
    social_proximity_lag = Day(parsed_args[:spc_lag_days]),
)

get_fbmobility3_hyperparams(parsed_args) = (
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
)

get_fbmobility4_hyperparams(parsed_args) = (
    γ0 = parsed_args[:gamma0],
    λ0 = parsed_args[:lambda0],
    β_bounds = (parsed_args[:beta_bounds][1], parsed_args[:beta_bounds][2]),
    γ_bounds = (parsed_args[:gamma_bounds][1], parsed_args[:gamma_bounds][2]),
    λ_bounds = (parsed_args[:lambda_bounds][1], parsed_args[:lambda_bounds][2]),
    α_bounds = (parsed_args[:alpha_bounds][1], parsed_args[:alpha_bounds][2]),
    train_range = Day(parsed_args[:train_days]),
    forecast_range = Day(parsed_args[:test_days]),
    social_proximity_lag = Day(parsed_args[:spc_lag_days]),
)

function parse_commandline(args)
    s = ArgParseSettings()

    isvalidmodel(name) =
        name == "baseline" ||
        name == "fbmobility1" ||
        name == "fbmobility2" ||
        name == "fbmobility3" ||
        name == "fbmobility4"

    @add_arg_table s begin
        "model_name"
        help = "name of the model that will be used"
        arg_type = String
        range_tester = isvalidmodel
        required = true

        "train_growing_trajectory"
        help = "train the model by iteratively growing time span"
        action = :command

        "train_whole_trajectory"
        help = "train the model on the whole time span"
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

        "--spc_lag_days"
        help = "number of lag days that is used when reading the Social Proximity to Cases index"
        arg_type = Int
        default = 14

        "--gamma0"
        help = "inverse of the mean incubation period"
        arg_type = Float64
        default = 1.0 / 3.0

        "--lambda0"
        help = "inverse of the mean infectious period"
        arg_type = Float64
        default = 1.0 / 14.0

        "--alpha0"
        help = "the fatality rate"
        arg_type = Float64
        default = 2.5e-2

        "--beta_bounds"
        help = "lower and upper bounds contraints for the average contact rate"
        nargs = 2
        arg_type = Float64
        default = [0.0, 1.336] # ℜe ∈ [0.0; 6.68]

        "--gamma_bounds"
        help = "lower and upper bounds contraints for the inverse of the mean incubation period"
        nargs = 2
        arg_type = Float64
        default = [1.0 / 5.0, 1.0 / 2.0]

        "--lambda_bounds"
        help = "lower and upper bounds contraints for the inverse of the mean infectious period"
        nargs = 2
        arg_type = Float64
        default = [1.0 / 21.0, 1.0 / 7.0]

        "--alpha_bounds"
        help = "lower and upper bounds contraints for the inverse of the mean infectious period"
        nargs = 2
        arg_type = Float64
        default = [0.01, 0.06]
    end

    @add_arg_table s["train_growing_trajectory"] begin
        "--lr"
        help = "learning rate to be given to ADAM"
        arg_type = Float64
        default = 1e-2

        "--lr_decay_rate"
        help = "learning rate exponential decay rate"
        arg_type = Float64
        default = 0.5

        "--lr_decay_step"
        help = "number of iterations taken before decaying the learning rate"
        arg_type = Int
        default = 100

        "--lr_limit"
        help = "the minimum value at which learning rate decay is stopped"
        arg_type = Float64
        default = 1e-4

        "--weight_decay"
        help = "scaling factor for the weight decay term"
        arg_type = Float64
        default = 1e-4

        "--maxiters_initial"
        help = "the max number of iterations used for fiting the first time span"
        arg_type = Int
        default = 200

        "--maxiters_growth"
        help = "increase the max number of iterations by a fixed amount when growing the time span"
        arg_type = Int
        default = 200

        "--tspan_size_initial"
        help = "number of data points in the initial time span"
        arg_type = Int
        default = 10

        "--tspan_size_growth"
        help = "number of new data points taken when growing the time span"
        arg_type = Int
        default = 10
    end

    @add_arg_table s["train_whole_trajectory"] begin
        "--lr"
        help = "learning rate to be given to ADAM"
        arg_type = Float64
        default = 1e-2

        "--lr_decay_rate"
        help = "learning rate exponential decay rate"
        arg_type = Float64
        default = 0.5

        "--lr_decay_step"
        help = "number of iterations taken before decaying the learning rate"
        arg_type = Int
        default = 100

        "--lr_limit"
        help = "the minimum value at which learning rate decay is stopped"
        arg_type = Float64
        default = 1e-4

        "--weight_decay"
        help = "scaling factor for the weight decay term"
        arg_type = Float64
        default = 1e-4

        "--maxiters"
        help = "the max number of iterations used"
        arg_type = Int
        default = 200

        "--minibatching"
        help = "size of the minibatch used when training, 0 means no minibatching"
        arg_type = Int
        default = 0
    end

    return parse_args(args, s, as_symbols = true)
end
