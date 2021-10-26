# Activate the environment for running the script
if isfile("Project.toml") && isfile("Manifest.toml")
    import Pkg
    Pkg.activate(".")
end

using Dates, Plots, DataDeps, DataFrames, CSV, Covid19ModelVN.Helpers

import GeoDataFrames,
    Covid19ModelVN.JHUCSSEData,
    Covid19ModelVN.FacebookData,
    Covid19ModelVN.PopulationData,
    Covid19ModelVN.VnExpressData,
    Covid19ModelVN.VnCdcData

const CACHE_DIR = ".cacheddata"

const FPATH_PROVINCES_POPULATION_WITH_GADM_ID_VIETNAM =
    joinpath(CACHE_DIR, "average-population-by-province-vietnam.csv")

const FPATH_COVID_TIMESERIES_VIETNAM = joinpath(CACHE_DIR, "timeseries-covid19-vietnam.csv")

const FPATH_FB_SOCIAL_CONNECTEDNESS_VIETNAM =
    joinpath(CACHE_DIR, "social-connectedness-vietnam.csv")

const FPATH_FB_MOVEMENT_RANGE_AVERAGE_VIETNAM =
    joinpath(CACHE_DIR, "movement-range-vietnam.csv")

const FPATH_FB_MOVEMENT_RANGE_AVERAGE_HCM_CITY =
    joinpath(CACHE_DIR, "movement-range-hcm-city.csv")

const FPATH_FB_MOVEMENT_RANGE_AVERAGE_BINH_DUONG =
    joinpath(CACHE_DIR, "movement-range-binh-duong.csv")

const FPATH_FB_MOVEMENT_RANGE_AVERAGE_DONG_NAI =
    joinpath(CACHE_DIR, "movement-range-dong-nai.csv")

const FPATH_FB_MOVEMENT_RANGE_AVERAGE_LONG_AN =
    joinpath(CACHE_DIR, "movement-range-long-an.csv")

function cache_vietnam_province_level_gadm_and_gso_population(fpath; recreate = false)
    if isfile(fpath) && !recreate
        return fpath
    end

    df_gadm = GeoDataFrames.read(datadep"gadm2.8/VNM_adm.gpkg", 1)
    df_population =
        CSV.read(datadep"gso/vietnam-2020-average-population-by-province.csv", DataFrame)

    df_combined = PopulationData.combine_vietnam_province_level_gadm_and_gso_population(
        df_gadm,
        df_population,
    )
    return save_dataframe(df_combined, fpath)
end

function cache_jhu_csse_covid_timeseries_country_level(
    fpaths,
    country_names;
    recreate = false,
)
    df_confirmed =
        CSV.read(datadep"jhu-csse/time_series_covid19_confirmed_global.csv", DataFrame)
    df_recovered =
        CSV.read(datadep"jhu-csse/time_series_covid19_recovered_global.csv", DataFrame)
    df_deaths = CSV.read(datadep"jhu-csse/time_series_covid19_deaths_global.csv", DataFrame)

    for (fpath, country_name) ∈ zip(fpaths, country_names)
        if isfile(fpath) && !recreate
            continue
        end

        df_combined = JHUCSSEData.combine_country_level_timeseries(
            df_confirmed,
            df_recovered,
            df_deaths,
            country_name,
        )
        save_dataframe(df_combined, fpath)
    end

    return nothing
end

function cache_jhu_csse_covid_timeseries_us_county_level(
    fpaths,
    state_names,
    county_names;
    recreate = false,
)
    df_confirmed =
        CSV.read(datadep"jhu-csse/time_series_covid19_confirmed_US.csv", DataFrame)
    df_deaths = CSV.read(datadep"jhu-csse/time_series_covid19_deaths_US.csv", DataFrame)

    for (fpath, state_name, county_name) ∈ zip(fpaths, state_names, county_names)
        if isfile(fpath) && !recreate
            continue
        end

        df_combined = JHUCSSEData.combine_us_county_level_timeseries(
            df_confirmed,
            df_deaths,
            state_name,
            county_name,
        )
        save_dataframe(df_combined, fpath)
    end

    return nothing
end

function cache_fb_movement_range_timeseries_region_average(
    fpaths,
    country_codes,
    subdivision_ids;
    recreate = false,
)
    df_movement_range =
        FacebookData.read_movement_range(datadep"facebook/movement-range-2021-10-09.txt")

    for (fpath, country_code, subdivision_id) ∈ zip(fpaths, country_codes, subdivision_ids)
        if isfile(fpath) && !recreate
            return fpath
        end

        df_region_movement_range = FacebookData.region_average_movement_range(
            df_movement_range,
            country_code,
            subdivision_id,
        )
        save_dataframe(df_region_movement_range, fpath)
    end

    return nothing
end

function cache_fb_social_connectedness_inter_province(
    fpaths,
    country_codes;
    recreate = false,
)
    df_social_connectedness = FacebookData.read_social_connectedness(
        datadep"facebook/gadm1_nuts2_gadm1_nuts2.tsv",
    )

    for (fpath, country_code) ∈ zip(fpaths, country_codes)
        if isfile(fpath) && !recreate
            return fpath
        end

        df_country_social_connectedness = FacebookData.inter_province_social_connectedness(
            df_social_connectedness,
            country_code,
        )
        save_dataframe(df_country_social_connectedness, fpath)
    end

    return nothing
end

function cache_default_data(; recreate = false)
    cache_vietnam_province_level_gadm_and_gso_population(
        FPATH_PROVINCES_POPULATION_WITH_GADM_ID_VIETNAM,
        recreate = recreate,
    )
    cache_jhu_csse_covid_timeseries_country_level(
        [FPATH_COVID_TIMESERIES_VIETNAM],
        ["Vietnam"],
        recreate = recreate,
    )
    cache_fb_social_connectedness_inter_province(
        [FPATH_FB_SOCIAL_CONNECTEDNESS_VIETNAM],
        ["VNM"],
        recreate = recreate,
    )
    cache_fb_movement_range_timeseries_region_average(
        [
            FPATH_FB_MOVEMENT_RANGE_AVERAGE_VIETNAM,
            FPATH_FB_MOVEMENT_RANGE_AVERAGE_HCM_CITY,
            FPATH_FB_MOVEMENT_RANGE_AVERAGE_BINH_DUONG,
            FPATH_FB_MOVEMENT_RANGE_AVERAGE_DONG_NAI,
            FPATH_FB_MOVEMENT_RANGE_AVERAGE_LONG_AN,
        ],
        ["VNM", "VNM", "VNM", "VNM", "VNM"],
        [nothing, 26, 10, 2, 39],
        recreate = recreate,
    )
end

cache_default_data()
