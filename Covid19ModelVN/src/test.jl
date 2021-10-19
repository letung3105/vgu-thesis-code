# Activate the environment for running the script
if isfile("Project.toml") && isfile("Manifest.toml")
    import Pkg
    Pkg.activate(".")
end

using Dates, DataFrames
import Covid19ModelVN.PopulationData,
    Covid19ModelVN.FacebookData, Covid19ModelVN.VnExpressData, HTTP, CSV

df_vn_fb_connedtedness = FacebookData.save_intra_country_gadm1_nuts2_connectedness_index(
    joinpath(
        "datasets",
        "facebook",
        "social-connectedness-index",
        "gadm1_nuts2_gadm1_nuts2_aug2020.tsv",
    ),
    "datasets",
    "facebook-gadm1-nuts2-connectedness-index",
    "VNM",
)

df_population_data = PopulationData.save_vietnam_gadm1_population(
    joinpath("datasets", "gadm", "VNM_adm.gpkg"),
    joinpath("datasets", "gso", "VNM-2020-population-all-regions.csv"),
    "datasets",
    "VNM-gadm1-population",
)

df_provinces_confirmed_cases = VnExpressData.save_provinces_confirmed_cases_timeseries(
    "datasets",
    "vietnam-provinces-confirmed-cases",
    Date(2021, 4, 27),
    Date(2021, 10, 13),
    recreate = true
)

df_provinces_total_confirmed_cases = VnExpressData.save_provinces_total_confirmed_cases_timeseries(
    "datasets",
    "vietnam-provinces-total-confirmed-cases",
    Date(2021, 4, 27),
    Date(2021, 10, 13),
    recreate = true
)

df_provinces_confirmed_cases
let
    province_city_names = names(df_provinces_confirmed_cases)[2:end]
    @show setdiff(province_city_names, df_population_data.gadm1_name)
    @show setdiff(df_population_data.gadm1_name, province_city_names)
    nothing
end

let
    province_city_names = names(df_provinces_total_confirmed_cases)[2:end]
    @show setdiff(province_city_names, df_population_data.gadm1_name)
    @show setdiff(df_population_data.gadm1_name, province_city_names)
    nothing
end

let
    url = "https://vnexpress.net/microservice/sheet/type/covid19_2021_by_location"
    res = HTTP.get(url)
    df = CSV.read(res.body, DataFrame)
    VnExpressData.clean_provinces_confirmed_cases_timeseries!(df)
    df
end