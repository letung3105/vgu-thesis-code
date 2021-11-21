module Covid19ModelVN

using Dates, Statistics, Serialization
using CairoMakie, DataDeps, DataFrames, CSV
using OrdinaryDiffEq, DiffEqFlux
using StaticArrays
using ProgressMeter

export AbstractCovidModel,
    SEIRDBaseline,
    SEIRDFbMobility1,
    SEIRDFbMobility2,
    SEIRDFbMobility3,
    SEIRDFbMobility4,
    SEIRDFbMobility5,
    ℜe,
    fatality_rate,
    initparams,
    namedparams,
    EvalConfig,
    Predictor,
    Loss,
    TrainCallback,
    TrainCallbackConfig,
    TrainCallbackState,
    make_video_stream_callback,
    evaluate_model,
    calculate_forecasts_errors,
    plot_losses,
    plot_forecasts,
    plot_effective_reproduction_number,
    plot_ℜe,
    logit,
    hswish,
    boxconst,
    boxconst_inv,
    mae,
    mape,
    rmse,
    rmsle,
    get_prebuilt_covid_timeseries,
    get_prebuilt_population,
    get_prebuilt_movement_range,
    get_prebuilt_social_proximity,
    make_covid_timeseries,
    make_population,
    make_movement_range,
    make_social_proximity,
    TimeseriesDataset,
    TimeseriesConfig,
    timeseries_dataloader,
    train_test_split,
    load_timeseries,
    save_dataframe,
    lookup_saved_params,
    get_losses_save_fpath,
    get_params_save_fpath,
    bound,
    bound!,
    moving_average,
    moving_average!

include("FacebookData.jl")
include("JHUCSSEData.jl")
include("PopulationData.jl")
include("VnExpressData.jl")
include("VnCdcData.jl")

include("helpers.jl")
include("datasets.jl")
include("models.jl")
include("train_eval.jl")

end
