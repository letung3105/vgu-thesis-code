# Activate the environment for running the script
if isfile("Project.toml") && isfile("Manifest.toml")
    import Pkg
    Pkg.activate(".")
end

using Dates
import Covid19ModelVN.PopulationData,
    Covid19ModelVN.FacebookData, Covid19ModelVN.VnExpressData

FacebookData.save_intra_country_gadm1_nuts2_connectedness_index(
    joinpath(
        "datasets",
        "facebook",
        "social-connectedness-index",
        "gadm1_nuts2_gadm1_nuts2_aug2020.tsv",
    ),
    "datasets",
    "facebook-gadm1-nuts2-connectedness-index",
    "VNM",
    recreate = true,
)

PopulationData.save_vietnam_gadm1_population(
    joinpath("datasets", "gadm", "VNM_adm.gpkg"),
    joinpath("datasets", "gso", "VNM-2020-population-all-regions.csv"),
    "datasets",
    "VNM-gadm1-population",
    recreate = true,
)

df_provinces_confirmed_cases = VnExpressData.save_provinces_confirmed_cases_timeseries(
    "datasets",
    "vnexpress-provinces-confirmed-cases",
    Date(2021, 4, 27),
    Date(2021, 10, 13),
    recreate = true,
)

df_provinces_total_confirmed_cases = VnExpressData.save_provinces_total_confirmed_cases_timeseries(
    "datasets",
    "vnexpress-provinces-total-confirmed-cases",
    Date(2021, 4, 27),
    Date(2021, 10, 13),
    recreate = true,
)

setdiff(df_cases_timeseries[!, 2:end], df_)