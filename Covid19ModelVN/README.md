# A Covid19 model for Vietnam

This repository is a supplementary material for my thesis at the university.

## Getting started

The interpreter for the Julia language is required for running the provided scripts.
To install the Julia interpreter, visit the [download page](https://julialang.org/downloads/) and follow the provided instructions.

Once the interpreter has been installed, change directory to the `Covid19ModelVN` folder and run the following script in a terminal to install the required third-party dependencies.

```bash
$ julia --project --eval "import Pkg; Pkg.instantiate()"
```

## Project structure

+ `src/helpers.jl` contains helpers methods the dealing with time series datasets and I/O operations.
+ `src/models.jl` contains the models implementations.
+ `src/train_eval.jl` contains helpers data structures and methods for training and evaluating the models.

+ `src/datasets.jl` contains methods for aggregating different datasets into easy to handle formats
+ `src/FacebookData.jl` contains methods for extracting mobility data and friend connections data from Facebook's datasets
+ `src/JHUCSSEData.jl` contains methods for extracting Covid-19 data from John Hopkins University's datasets
+ `src/PopulationData.jl`contains methods for combining GSO's population data with GADM v2.8 identifiers
+ `src/VnCdcData.jl` contains methods for parsing Covid-19 data from VnCDC
+ `src/VnExpressData.jl` contains methods for getting Covid-19 data from VnExpress.net

+ `src/scripts/include/cmd.jl` contains methods for setting up and running the command-line application
+ `src/scripts/include/experiments.jl` contains methods for data preprocessing, the loss functions, and methods for setting up and running the experiments.
+ `src/scripts/include/trainalgs.jl` contains our implementations for the model training procedures

+ `src/scripts/48days.jl` contains the experiments that had been run to obtained the results presented in our thesis
+ `src/scripts/app.jl` contains the entrypoint for the command-line application used for training and evaluating the model
+ `src/scripts/benchmark.jl` contains methods for checking the model performance
+ `src/scripts/diagnostics.jl` contains methods for checking whether the methods for training and evaluating the model can be run correctly
+ `src/scripts/ensemble_seir.jl` contains the implementations of an ensemble stochastic SEIR model to visualize the effects of different model parameters
+ `src/scripts/mkdatasets.jl` contains the entrypoint for the command-line application for creating all the intermediate datasets
+ `src/scripts/plot_results.jl` contains methods for plotting the figures shown in the thesis
+ `src/scripts/train_simulated.jl` contains a the code for training the baseline model against simulated data to see if our model is implemented correctly

## Script usage

As an end user, you will be running the script located at `src/scripts/app.jl` to train and evaluate the model with data for different locations.
Currently, the model can only be trained with data for predefined locations that were considered in the thesis.
Running the script required the settings of various parameters for the model and for the training procedure.
To get a list of available options, run the following script in a terminal:

```bash
$ julia --project src/scripts/app.jl -- --help
```

which will printed

```bash
usage: app.jl [--locations [LOCATIONS...]]
              [--forecast_horizons [FORECAST_HORIZONS...]]
              [--savedir SAVEDIR] [--multithreading] [--show_progress]
              [--make_animation] [--train_days TRAIN_DAYS]
              [--test_days TEST_DAYS]
              [--movement_range_lag_days MOVEMENT_RANGE_LAG_DAYS]
              [--social_proximity_lag_days SOCIAL_PROXIMITY_LAG_DAYS]
              [--gamma0 GAMMA0] [--lambda0 LAMBDA0]
              [--beta_bounds BETA_BOUNDS BETA_BOUNDS]
              [--gamma_bounds GAMMA_BOUNDS GAMMA_BOUNDS]
              [--lambda_bounds LAMBDA_BOUNDS LAMBDA_BOUNDS]
              [--alpha_bounds ALPHA_BOUNDS ALPHA_BOUNDS]
              [--loss_type LOSS_TYPE]
              [--loss_regularization LOSS_REGULARIZATION]
              [--loss_time_weighting LOSS_TIME_WEIGHTING] [-h]
              model_name
              {eval|train_growing_trajectory|train_growing_trajectory_two_stages|train_whole_trajectory|train_whole_trajectory_two_stages}

commands:
  eval                  draw plot and make tables for evaluating the
                        model
  train_growing_trajectory
                        train the model by iteratively growing time
                        span
  train_growing_trajectory_two_stages
                        train the model by iteratively growing time
                        span, then use LBFGS
  train_whole_trajectory
                        train the model on the whole time span
  train_whole_trajectory_two_stages
                        train the model on the whole time span, then
                        use LBFGS

positional arguments:
  model_name            name of the model that will be used (type:
                        Symbol)

optional arguments:
  --locations [LOCATIONS...]
                        the code of the locations whose data will be
                        used to train the model
  --forecast_horizons [FORECAST_HORIZONS...]
                        the numbers of days that will be forecasted
                        (type: Int32, default: Int32[7, 14, 21, 28])
  --savedir SAVEDIR     path to the directory where the model outputs
                        are saved (default: "./snapshots")
  --multithreading      use multiple threads to train the model at
                        multiple locations at once
  --show_progress       show a progress meter that keeps track of the
                        training sessions
  --make_animation      show a progress meter that keeps track of the
                        training sessions
  --train_days TRAIN_DAYS
                        number of days used for training (type: Int64,
                        default: 32)
  --test_days TEST_DAYS
                        number of days used for testing (type: Int64,
                        default: 28)
  --movement_range_lag_days MOVEMENT_RANGE_LAG_DAYS
                        number of lag days that is used when reading
                        the Movement Range Maps dataset (type: Int64,
                        default: 0)
  --social_proximity_lag_days SOCIAL_PROXIMITY_LAG_DAYS
                        number of lag days that is used when reading
                        the Social Proximity to Cases index (type:
                        Int64, default: 0)
  --gamma0 GAMMA0       inverse of the mean incubation period (type:
                        Float64, default: 0.25)
  --lambda0 LAMBDA0     inverse of the mean infectious period (type:
                        Float64, default: 0.0714286)
  --beta_bounds BETA_BOUNDS BETA_BOUNDS
                        lower and upper bounds contraints for the
                        average contact rate (type: Float64, default:
                        [0.0, 1.67])
  --gamma_bounds GAMMA_BOUNDS GAMMA_BOUNDS
                        lower and upper bounds contraints for the
                        inverse of the mean incubation period (type:
                        Float64, default: [0.25, 0.25])
  --lambda_bounds LAMBDA_BOUNDS LAMBDA_BOUNDS
                        lower and upper bounds contraints for the
                        inverse of the mean infectious period (type:
                        Float64, default: [0.0714286, 0.0714286])
  --alpha_bounds ALPHA_BOUNDS ALPHA_BOUNDS
                        lower and upper bounds contraints for the
                        inverse of the mean infectious period (type:
                        Float64, default: [0.005, 0.05])
  --loss_type LOSS_TYPE
                        choose the loss function that is used for
                        training (type: Symbol, default: :sse)
  --loss_regularization LOSS_REGULARIZATION
                        scaling factor for the weight decay term
                        (type: Float64, default: 0.0)
  --loss_time_weighting LOSS_TIME_WEIGHTING
                        scaling factor for the time scaling (type:
                        Float64, default: 0.0)
  -h, --help            show this help message and exit
```

Command-line arguments given to the script can be specified after the double dashes `--`, for example, this command runs an experiment with the settings that we presented in the thesis:

```bash
$ julia --project src/scripts/app.jl --
  --beta_bounds 0.05 1.67
  --train_days 48
  --loss_regularization 0.00001
  --loss_time_weighting -0.001
  --locations dongnai
  --savedir testsnapshots/batch48days
  --multithreading fbmobility2
  train_growing_trajectory_two_stages
  --lr 0.05
  --lr_limit 0.00001
  --lr_decay_rate 0.5
  --lr_decay_step 1000
  --maxiters_initial 1000
  --maxiters_growth 0
  --maxiters_second 1000
  --tspan_size_initial 4
  --tspan_size_growth 4
```

To see the helper message without having to run the script, visit `src/scripts/include/cmd.jl`.
Because Julia is JIT-compiled, running the script from a terminal may take a lot of time for the initial boot up.
Thus we recommended running the package in Julia interactive mode when experimenting with the provided models.
The command-line arguments can be given programmatically in similar formats as shown in the helper message.
For more example of how to use the options programmatically, see `src/scripts/48days.jl`.

You will be asked to download several datasets hosted at [https://github.com/letung3105/coviddata](https://github.com/letung3105/coviddata) on the first time that you run the model training.
These are the datasets that had been constructed specifically for the experiments.
When you accept the download prompt, these datasets will be downloaded to your local machine and will be typically stored at `$HOME/.julia/datadeps`.
See [https://oxinabox.github.io/DataDeps.jl/stable](https://oxinabox.github.io/DataDeps.jl/stable) if you want to have more control over where these datasets will be stored.

## Compatibility

The project has been tested with Julia version 1.7.0 on a Linux operating system.
