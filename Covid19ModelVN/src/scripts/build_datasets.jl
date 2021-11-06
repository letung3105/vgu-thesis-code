# Activate the environment for running the script
if isfile("Project.toml") && isfile("Manifest.toml")
    import Pkg
    Pkg.activate(".")
end

using Covid19ModelVN, DataDeps, CSV, DataFrames

import Covid19ModelVN.JHUCSSEData,
    Covid19ModelVN.FacebookData,
    Covid19ModelVN.PopulationData,
    Covid19ModelVN.VnExpressData,
    Covid19ModelVN.VnCdcData

function build_datasets_main(dir::AbstractString = "")
    # POPULATION DATA
    PopulationData.save_vietnam_province_level_gadm_and_gso_population(
        joinpath(dir, "average-population-vn-provinces.csv"),
    )
    JHUCSSEData.save_us_counties_population(
        joinpath(dir, "average-population-us-counties.csv"),
    )

    # COVID CASES
    JHUCSSEData.save_country_level_timeseries([
        JHUCSSEData.CountryCovidTimeseriesFile(
            joinpath(dir, "timeseries-covid19-combined-vietnam.csv"),
            "Vietnam",
        ),
        JHUCSSEData.CountryCovidTimeseriesFile(
            joinpath(dir, "timeseries-covid19-combined-united-states.csv"),
            "US",
        ),
    ])
    JHUCSSEData.save_us_county_level_timeseries([
        JHUCSSEData.CountyCovidTimeseriesFile(
            joinpath(dir, "timeseries-covid19-combined-los-angeles-CA.csv"),
            "California",
            "Los Angeles",
        ),
        JHUCSSEData.CountyCovidTimeseriesFile(
            joinpath(dir, "timeseries-covid19-combined-cook-county-IL.csv"),
            "Illinois",
            "Cook",
        ),
        JHUCSSEData.CountyCovidTimeseriesFile(
            joinpath(dir, "timeseries-covid19-combined-harris-county-TX.csv"),
            "Texas",
            "Harris",
        ),
        JHUCSSEData.CountyCovidTimeseriesFile(
            joinpath(dir, "timeseries-covid19-combined-maricopa-county-AZ.csv"),
            "Arizona",
            "Maricopa",
        ),
    ])

    # MOVEMENT RANGE MAPS
    FacebookData.save_region_average_movement_range([
        FacebookData.RegionMovementRangeFile(
            joinpath(dir, "movement-range-vietnam.csv"),
            "VNM",
            nothing,
        ),
        FacebookData.RegionMovementRangeFile(
            joinpath(dir, "movement-range-hcm-city.csv"),
            "VNM",
            26,
        ),
        FacebookData.RegionMovementRangeFile(
            joinpath(dir, "movement-range-binh-duong.csv"),
            "VNM",
            10,
        ),
        FacebookData.RegionMovementRangeFile(
            joinpath(dir, "movement-range-dong-nai.csv"),
            "VNM",
            2,
        ),
        FacebookData.RegionMovementRangeFile(
            joinpath(dir, "movement-range-long-an.csv"),
            "VNM",
            39,
        ),
        FacebookData.RegionMovementRangeFile(
            joinpath(dir, "movement-range-united-states.csv"),
            "USA",
            nothing,
        ),
        FacebookData.RegionMovementRangeFile(
            joinpath(dir, "movement-range-los-angeles-CA.csv"),
            "USA",
            6037,
        ),
        FacebookData.RegionMovementRangeFile(
            joinpath(dir, "movement-range-cook-county-IL.csv"),
            "USA",
            17031,
        ),
        FacebookData.RegionMovementRangeFile(
            joinpath(dir, "movement-range-harris-county-TX.csv"),
            "USA",
            13145,
        ),
        FacebookData.RegionMovementRangeFile(
            joinpath(dir, "movement-range-maricopa-county-AZ.csv"),
            "USA",
            4013,
        ),
    ])
    FacebookData.save_region_average_movement_range(
        [
            FacebookData.RegionMovementRangeFile(
                joinpath(dir, "movement-range-united-states-2020.csv"),
                "USA",
                nothing,
            ),
            FacebookData.RegionMovementRangeFile(
                joinpath(dir, "movement-range-los-angeles-CA-2020.csv"),
                "USA",
                6037,
            ),
            FacebookData.RegionMovementRangeFile(
                joinpath(dir, "movement-range-cook-county-IL-2020.csv"),
                "USA",
                17031,
            ),
            FacebookData.RegionMovementRangeFile(
                joinpath(dir, "movement-range-harris-county-TX-2020.csv"),
                "USA",
                13145,
            ),
            FacebookData.RegionMovementRangeFile(
                joinpath(dir, "movement-range-maricopa-county-AZ-2020.csv"),
                "USA",
                4013,
            ),
        ],
        fpath_movement_range = datadep"facebook/movement-range-data-2020-03-01--2020-12-31.txt",
    )

    # SOCIAL PROXIMITY TO CASES INDEX
    let
        @info "Reading social connectedness index dataset"
        df_sci = FacebookData.read_social_connectedness(
            datadep"facebook/gadm1_nuts2_gadm1_nuts2.tsv",
        )

        @info "Get SCI between provinces"
        df_sci_vn = FacebookData.inter_province_social_connectedness(df_sci, "VNM")

        @info "Reading population data"
        df_population =
            CSV.read(joinpath(dir, "average-population-vn-provinces.csv"), DataFrame)

        @info "Reading Covid-19 timeseries"
        df_covid = CSV.read(
            datadep"vnexpress/timeseries-vietnam-provinces-confirmed.csv",
            DataFrame,
        )

        @info "Calculating social proximity to cases index"
        df_scp_vn = FacebookData.calculate_social_proximity_to_cases(
            df_population,
            df_covid,
            df_sci_vn,
        )
        save_dataframe(
            df_scp_vn,
            joinpath(dir, "social-proximity-to-cases-vn-provinces.csv"),
        )
    end

    let
        @info "Reading social connectedness index dataset"
        df_sci_counties =
            FacebookData.read_social_connectedness(datadep"facebook/county_county.tsv")

        @info "Reading population data"
        df_population =
            CSV.read(joinpath(dir, "average-population-us-counties.csv"), DataFrame)

        @info "Reading Covid-19 timeseries"
        df_covid = JHUCSSEData.get_us_counties_timeseries_confirmed(
            CSV.read(datadep"jhu-csse/time_series_covid19_confirmed_US.csv", DataFrame),
        )

        @info "Calculating social proximity to cases index"
        df_scp_us = FacebookData.calculate_social_proximity_to_cases(
            df_population,
            df_covid,
            df_sci_counties,
        )
        save_dataframe(
            df_scp_us,
            joinpath(dir, "social-proximity-to-cases-us-counties.csv"),
        )
    end
end

build_datasets_main()
