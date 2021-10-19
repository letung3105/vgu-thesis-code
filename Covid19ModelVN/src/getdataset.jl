# Activate the environment for running the script
if isfile("Project.toml") && isfile("Manifest.toml")
    import Pkg
    Pkg.activate(".")
end

using Dates
import Covid19ModelVN.PopulationData,
    Covid19ModelVN.FacebookData, Covid19ModelVN.VnExpressData

const DEFAULT_DATASETS_DIR = "datasets"

df_vn_gadm1_population = PopulationData.save_vietnam_gadm1_population(
    joinpath(DEFAULT_DATASETS_DIR, "gadm", "VNM_adm.gpkg"),
    joinpath(DEFAULT_DATASETS_DIR, "gso", "VNM-2020-population-all-regions.csv"),
    DEFAULT_DATASETS_DIR,
    "VNM-gadm1-population",
    recreate = true,
)

df_cases_timeseries = VnExpressData.save_cases_timeseries(
    DEFAULT_DATASETS_DIR,
    "vietnam-cases-timeseries",
    Date(2021, 4, 27),
    Date(2021, 10, 13),
    recreate = true,
)

df_provinces_confirmed_cases = VnExpressData.save_provinces_confirmed_cases_timeseries(
    DEFAULT_DATASETS_DIR,
    "vietnam-provinces-confirmed-timeseries",
    Date(2021, 4, 27),
    Date(2021, 10, 13),
    recreate = true,
)

df_provinces_total_confirmed_cases = VnExpressData.save_provinces_total_confirmed_cases_timeseries(
    DEFAULT_DATASETS_DIR,
    "vietnam-provinces-total-confirmed-timeseries",
    Date(2021, 4, 27),
    Date(2021, 10, 13),
    recreate = true,
)

df_fb_vn_movement_range = FacebookData.save_country_average_movement_range(
    joinpath(
        DEFAULT_DATASETS_DIR,
        "facebook",
        "movement-range-data-2021-10-09",
        "movement-range-2021-10-09.txt",
    ),
    DEFAULT_DATASETS_DIR,
    "facebook-average-movement-range",
    "VNM",
    recreate = true,
)

df_fb_vn_connected_index = FacebookData.save_intra_country_gadm1_nuts2_connectedness_index(
    joinpath(
        DEFAULT_DATASETS_DIR,
        "facebook",
        "social-connectedness-index",
        "gadm1_nuts2_gadm1_nuts2_aug2020.tsv",
    ),
    DEFAULT_DATASETS_DIR,
    "facebook-intra-connectedness-index",
    "VNM",
    recreate = true,
)