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

    DEFAULT_VIETNAM_PROVINCE_CONFIRMED_AND_DEATHS_TIMESERIES(DEFAULT_DATASETS_DIR, "HoChiMinh")
    DEFAULT_VIETNAM_PROVINCE_CONFIRMED_AND_DEATHS_TIMESERIES(DEFAULT_DATASETS_DIR, "BinhDuong")
    DEFAULT_VIETNAM_PROVINCE_CONFIRMED_AND_DEATHS_TIMESERIES(DEFAULT_DATASETS_DIR, "DongNai")
    DEFAULT_VIETNAM_PROVINCE_CONFIRMED_AND_DEATHS_TIMESERIES(DEFAULT_DATASETS_DIR, "LongAn")

    DEFAULT_VIETNAM_AVERAGE_MOVEMENT_RANGE(DEFAULT_DATASETS_DIR)
    DEFAULT_VIETNAM_PROVINCE_AVERAGE_MOVEMENT_RANGE(DEFAULT_DATASETS_DIR, 26)
    DEFAULT_VIETNAM_INTRA_CONNECTEDNESS_INDEX(DEFAULT_DATASETS_DIR)
    DEFAULT_VIETNAM_SOCIAL_PROXIMITY_TO_CASES_INDEX(DEFAULT_DATASETS_DIR)

    return nothing
end

let
    df_covid1 = DEFAULT_VIETNAM_PROVINCE_CONFIRMED_AND_DEATHS_TIMESERIES(DEFAULT_DATASETS_DIR, "HoChiMinh")
    df_covid2 = DEFAULT_VIETNAM_PROVINCES_TOTAL_CONFIRMED_TIMESERIES(DEFAULT_DATASETS_DIR)

    filter!(x -> Date(2021, 5, 14) <= x.date <= Date(2021, 7, 14), df_covid1)
    filter!(x -> Date(2021, 5, 14) <= x.date <= Date(2021, 7, 14), df_covid2)

    @show df_covid1.confirmed_total
    @show df_covid2[!, "Hồ Chí Minh city"]

    plt = plot()
    plot!(df_covid1.date, df_covid1.confirmed_total)
    plot!(df_covid2.date, df_covid2[!, "Hồ Chí Minh city"])
    display(plt)

    df_spc = DEFAULT_VIETNAM_SOCIAL_PROXIMITY_TO_CASES_INDEX(DEFAULT_DATASETS_DIR)

    plt = plot(df_spc.date, Array(df_spc[!, ["Hồ Chí Minh city", "Bình Dương"]]))
    display(plt)

    df_population = DEFAULT_VIETNAM_GADM1_POPULATION_DATASET(DEFAULT_DATASETS_DIR)
    id_hcm = first(filter(x -> x.gadm1_name == "Hồ Chí Minh city", df_population).gadm1_id)
    id_bd = first(filter(x -> x.gadm1_name == "Bình Dương", df_population).gadm1_id)

    @show id_hcm
    @show id_bd

    df_movement_hcm = DEFAULT_VIETNAM_PROVINCE_AVERAGE_MOVEMENT_RANGE(DEFAULT_DATASETS_DIR, id_hcm)
    df_movement_bd = DEFAULT_VIETNAM_PROVINCE_AVERAGE_MOVEMENT_RANGE(DEFAULT_DATASETS_DIR, id_bd)

    plt = plot()
    plot!(df_movement_hcm.ds, Array(df_movement_hcm[!, Not(:ds)]))
    plot!(df_movement_bd.ds, Array(df_movement_bd[!, Not(:ds)]))
    display(plt)

    nothing
end