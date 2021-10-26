# Activate the environment for running the script
if isfile("Project.toml") && isfile("Manifest.toml")
    import Pkg
    Pkg.activate(".")
end

import Covid19ModelVN.JHUCSSEData,
    Covid19ModelVN.FacebookData,
    Covid19ModelVN.PopulationData,
    Covid19ModelVN.VnExpressData,
    Covid19ModelVN.VnCdcData

const CACHE_DIR = ".cacheddata"

const FPATH_VIETNAM_PROVINCES_GADM_AND_GSO_POPULATION =
    joinpath(CACHE_DIR, "average-population-by-province-vietnam.csv")

const FPATH_VIETNAM_COVID_TIMESERIES = joinpath(CACHE_DIR, "timeseries-covid19-vietnam.csv")

const FPATH_VIETNAM_INTER_PROVINCE_SOCIAL_CONNECTEDNESS =
    joinpath(CACHE_DIR, "social-connectedness-vietnam.csv")

const FPATH_VIETNAM_AVERAGE_MOVEMENT_RANGE =
    joinpath(CACHE_DIR, "movement-range-vietnam.csv")

const FPATH_HCM_CITY_AVERAGE_MOVEMENT_RANGE =
    joinpath(CACHE_DIR, "movement-range-hcm-city.csv")

const FPATH_BINH_DUONG_AVERAGE_MOVEMENT_RANGE =
    joinpath(CACHE_DIR, "movement-range-binh-duong.csv")

const FPATH_DONG_NAI_AVERAGE_MOVEMENT_RANGE =
    joinpath(CACHE_DIR, "movement-range-dong-nai.csv")

const FPATH_LONG_AN_AVERAGE_MOVEMENT_RANGE =
    joinpath(CACHE_DIR, "movement-range-long-an.csv")

function cache_default_data(; recreate = false)
    PopulationData.save_vietnam_province_level_gadm_and_gso_population(
        FPATH_VIETNAM_PROVINCES_GADM_AND_GSO_POPULATION,
        recreate = recreate,
    )
    JHUCSSEData.save_country_level_timeseries(
        [FPATH_VIETNAM_COVID_TIMESERIES],
        ["Vietnam"],
        recreate = recreate,
    )
    FacebookData.save_inter_province_social_connectedness(
        [FPATH_VIETNAM_INTER_PROVINCE_SOCIAL_CONNECTEDNESS],
        ["VNM"],
        recreate = recreate,
    )
    FacebookData.save_region_average_movement_range(
        [
            FPATH_VIETNAM_AVERAGE_MOVEMENT_RANGE,
            FPATH_HCM_CITY_AVERAGE_MOVEMENT_RANGE,
            FPATH_BINH_DUONG_AVERAGE_MOVEMENT_RANGE,
            FPATH_DONG_NAI_AVERAGE_MOVEMENT_RANGE,
            FPATH_LONG_AN_AVERAGE_MOVEMENT_RANGE,
        ],
        ["VNM", "VNM", "VNM", "VNM", "VNM"],
        [nothing, 26, 10, 2, 39],
        recreate = recreate,
    )
end