# Activate the environment for running the script
if isfile("Project.toml") && isfile("Manifest.toml")
    import Pkg
    Pkg.activate(".")
end

using Dates, Plots, DataDeps, DataFrames, CSV, Covid19ModelVN.Datasets, Covid19ModelVN.Helpers

import Covid19ModelVN.JHUCSSEData, Covid19ModelVN.FacebookData

const DEFAULT_DATASETS_DIR = "datasets"

function test_load_datasets()
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

function test_visualizations()
    train_range = Day(31)
    forecast_range = Day(28)

    df_covid = DEFAULT_VIETNAM_PROVINCE_CONFIRMED_AND_DEATHS_TIMESERIES(
        DEFAULT_DATASETS_DIR,
        "HoChiMinh",
    )
    first_date = first(filter!(x -> x.confirmed_total >= 500, df_covid).date)
    split_date = first_date + train_range
    last_date = split_date + forecast_range

    covid_cols = [:dead_total, :confirmed_total]
    moving_average!(df_covid, covid_cols, 7)

    train_dataset, test_dataset =
        train_test_split(df_covid, covid_cols, :date, first_date, split_date, last_date)
    plt = plot(title = "covid cases", legend = :outertop)
    plot!(
        [train_dataset.data test_dataset.data]',
        labels = ["dead total" "confirmed total"],
    )
    display(plt)

    df_population = DEFAULT_VIETNAM_GADM1_POPULATION_DATASET(DEFAULT_DATASETS_DIR)
    province_id =
        first(filter(x -> x.gadm1_name == "Hồ Chí Minh city", df_population).gadm1_id)

    # load facebook movement range
    df_movement_range = DEFAULT_VIETNAM_AVERAGE_MOVEMENT_RANGE(DEFAULT_DATASETS_DIR)
    movement_range_cols =
        [:all_day_bing_tiles_visited_relative_change, :all_day_ratio_single_tile_users]
    moving_average!(df_movement_range, movement_range_cols, 7)
    # load timeseries data with the chosen temporal lag
    load_movement_range(lag) = load_timeseries(
        df_movement_range,
        movement_range_cols,
        :ds,
        first_date - lag,
        last_date - lag,
    )
    movement_data = load_movement_range(Day(2))
    plt = plot(title = "Movement Range Index", legend = :outertop)
    plot!(movement_data, labels = ["relative movement change" "relative stay put ratio"])
    display(plt)

    df_vn_spc = DEFAULT_VIETNAM_SOCIAL_PROXIMITY_TO_CASES_INDEX(DEFAULT_DATASETS_DIR)
    moving_average!(df_vn_spc, "Hồ Chí Minh city", 7)
    load_spc(lag) = load_timeseries(
        df_vn_spc,
        "Hồ Chí Minh city",
        :date,
        first_date - lag,
        first_date - lag + train_range + forecast_range,
    )
    vn_spc_data = load_spc(Day(2))
    plt = plot(title = "Social Proximity to Cases Index", legend = :outertop)
    plot!(vn_spc_data)
    display(plt)

    nothing
end

let
    df_confirmed = CSV.read(
        datadep"jhu-csse-covid19/time_series_covid19_confirmed_global.csv",
        DataFrame,
    )
    df_recovered = CSV.read(
        datadep"jhu-csse-covid19/time_series_covid19_recovered_global.csv",
        DataFrame,
    )
    df_deaths =
        CSV.read(datadep"jhu-csse-covid19/time_series_covid19_deaths_global.csv", DataFrame)

    @time df = JHUCSSEData.combine_country_level_timeseries(
        df_confirmed,
        df_recovered,
        df_deaths,
        "Vietnam",
    )
    save_dataframe(
        df,
        "datasets/jhu-csse-covid19",
        "time_series_covid19_combined_vietnam",
    )
end

let
    df_confirmed =
        CSV.read(datadep"jhu-csse-covid19/time_series_covid19_confirmed_US.csv", DataFrame)
    df_deaths =
        CSV.read(datadep"jhu-csse-covid19/time_series_covid19_deaths_US.csv", DataFrame)

    @time population = JHUCSSEData.get_us_county_population(df_deaths, "Alabama", "Autauga")
    @time df = JHUCSSEData.combine_us_county_level_timeseries(df_confirmed, df_deaths, "Alabama", "Autauga")
    @show population
    save_dataframe(
        df,
        "datasets/jhu-csse-covid19",
        "time_series_covid19_combined_us_autauga",
    )
end

df_movement_range = FacebookData.read_movement_range(
    "datasets/facebook/movement-range-data-2021-10-09/movement-range-2021-10-09.txt"
)
df_movement_range_vn = FacebookData.region_average_movement_range(df_movement_range, "VNM")
save_dataframe(df_movement_range_vn, "datasets/facebook", "facebook-movement-range-average-vnm")

df_social_connectedness = FacebookData.read_social_connectedness(
    "datasets/facebook/social-connectedness-index/gadm1_nuts2_gadm1_nuts2_aug2020.tsv"
)
df_social_connectedness_vn = FacebookData.inter_province_social_connectedness(df_social_connectedness, "VNM")
save_dataframe(df_social_connectedness_vn, "datasets/facebook", "facebook-social-connectedness-vnm")