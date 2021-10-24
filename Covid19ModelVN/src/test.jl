# Activate the environment for running the script
if isfile("Project.toml") && isfile("Manifest.toml")
    import Pkg
    Pkg.activate(".")
end

using Dates, Plots, DataFrames, Covid19ModelVN.Datasets

const DEFAULT_DATASETS_DIR = "datasets"

function main()
    DEFAULT_VIETNAM_GADM1_POPULATION_DATASET(DEFAULT_DATASETS_DIR)
    DEFAULT_VIETNAM_COVID_DATA_TIMESERIES(DEFAULT_DATASETS_DIR)
    DEFAULT_VIETNAM_PROVINCES_CONFIRMED_TIMESERIES(DEFAULT_DATASETS_DIR)
    DEFAULT_VIETNAM_PROVINCES_TOTAL_CONFIRMED_TIMESERIES(DEFAULT_DATASETS_DIR)

    DEFAULT_VIETNAM_PROVINCE_CONFIRMED_AND_DEATHS_TIMESERIES(
        DEFAULT_DATASETS_DIR,
        "HoChiMinh",
    )
    DEFAULT_VIETNAM_PROVINCE_CONFIRMED_AND_DEATHS_TIMESERIES(
        DEFAULT_DATASETS_DIR,
        "BinhDuong",
    )
    DEFAULT_VIETNAM_PROVINCE_CONFIRMED_AND_DEATHS_TIMESERIES(
        DEFAULT_DATASETS_DIR,
        "DongNai",
    )
    DEFAULT_VIETNAM_PROVINCE_CONFIRMED_AND_DEATHS_TIMESERIES(DEFAULT_DATASETS_DIR, "LongAn")

    DEFAULT_VIETNAM_AVERAGE_MOVEMENT_RANGE(DEFAULT_DATASETS_DIR)
    DEFAULT_VIETNAM_PROVINCE_AVERAGE_MOVEMENT_RANGE(DEFAULT_DATASETS_DIR, 26)
    DEFAULT_VIETNAM_INTRA_CONNECTEDNESS_INDEX(DEFAULT_DATASETS_DIR)
    DEFAULT_VIETNAM_SOCIAL_PROXIMITY_TO_CASES_INDEX(DEFAULT_DATASETS_DIR)

    return nothing
end

let
    df_covid = DEFAULT_VIETNAM_PROVINCE_CONFIRMED_AND_DEATHS_TIMESERIES(
        DEFAULT_DATASETS_DIR,
        "HoChiMinh",
    )

    train_first_date = first(filter!(x -> x.confirmed_total >= 500, df_covid).date)
    train_range = Day(30)
    forecast_range = Day(30)

    train_dataset, test_dataset = load_covid_cases_datasets(
        df_covid,
        [:dead_total, :confirmed_total],
        train_first_date,
        train_range,
        forecast_range,
        7,
    )
    plt = plot(title = "covid cases", legend = :outertop)
    plot!(
        [train_dataset.data test_dataset.data]',
        labels = ["dead total" "confirmed total"],
    )
    display(plt)

    df_population = DEFAULT_VIETNAM_GADM1_POPULATION_DATASET(DEFAULT_DATASETS_DIR)
    province_id =
        first(filter(x -> x.gadm1_name == "Hồ Chí Minh city", df_population).gadm1_id)

    df_movement =
        DEFAULT_VIETNAM_PROVINCE_AVERAGE_MOVEMENT_RANGE(DEFAULT_DATASETS_DIR, province_id)
    movement_data = load_fb_movement_range(
        df_movement,
        train_first_date,
        train_range,
        forecast_range,
        Day(2),
        7,
    )
    plt = plot(title = "Movement Range Index", legend = :outertop)
    plot!(movement_data, labels = ["relative movement change" "relative stay put ratio"])
    display(plt)

    df_spc = DEFAULT_VIETNAM_SOCIAL_PROXIMITY_TO_CASES_INDEX(DEFAULT_DATASETS_DIR)
    spc_data = load_social_proximity_to_cases_index(
        df_spc,
        "Hồ Chí Minh city",
        train_first_date,
        train_range,
        forecast_range,
        Day(2),
        7,
    )
    plt = plot(title = "Social Proximity to Cases Index", legend = :outertop)
    plot!(spc_data)
    display(plt)

    nothing
end
