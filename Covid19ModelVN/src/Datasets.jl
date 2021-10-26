module Datasets

export DEFAULT_VIETNAM_GADM1_POPULATION_DATASET,
    DEFAULT_VIETNAM_COVID_DATA_TIMESERIES,
    DEFAULT_VIETNAM_AVERAGE_MOVEMENT_RANGE,
    DEFAULT_VIETNAM_INTRA_CONNECTEDNESS_INDEX,
    DEFAULT_VIETNAM_PROVINCE_CONFIRMED_AND_DEATHS_TIMESERIES,
    DEFAULT_VIETNAM_PROVINCE_AVERAGE_MOVEMENT_RANGE,
    DEFAULT_VIETNAM_SOCIAL_PROXIMITY_TO_CASES_INDEX

using Dates
import Covid19ModelVN.FacebookData,
    Covid19ModelVN.VnExpressData, Covid19ModelVN.PopulationData, Covid19ModelVN.VnCdcData

DEFAULT_VIETNAM_SOCIAL_PROXIMITY_TO_CASES_INDEX(datasets_dir; recreate = false) =
    FacebookData.save_social_proximity_to_cases_index(
        joinpath(datasets_dir, "VNM-gadm1-population.csv"),
        joinpath(
            datasets_dir,
            "20210427-20211013-vietnam-provinces-confirmed-timeseries.csv",
        ),
        joinpath(datasets_dir, "VNM-facebook-intra-connectedness-index.csv"),
        datasets_dir,
        "VNM-social-proximity-to-cases",
        recreate = recreate,
    )

DEFAULT_VIETNAM_PROVINCE_CONFIRMED_AND_DEATHS_TIMESERIES(datasets_dir, name) =
    VnCdcData.parse_json_cases_and_deaths(joinpath(datasets_dir, "vncdc", "$name.json"))

DEFAULT_VIETNAM_PROVINCE_AVERAGE_MOVEMENT_RANGE(
    datasets_dir,
    province_id;
    recreate = false,
) = FacebookData.save_country_average_movement_range(
    joinpath(
        datasets_dir,
        "facebook",
        "movement-range-data-2021-10-09",
        "movement-range-2021-10-09.txt",
    ),
    datasets_dir,
    "facebook-average-movement-range",
    "VNM",
    province_id,
    recreate = recreate,
)

DEFAULT_VIETNAM_GADM1_POPULATION_DATASET(datasets_dir; recreate = false) =
    PopulationData.save_vietnam_gadm1_population(
        joinpath(datasets_dir, "gadm", "VNM_adm.gpkg"),
        joinpath(datasets_dir, "gso", "VNM-2020-population-all-regions.csv"),
        datasets_dir,
        "VNM-gadm1-population",
        recreate = recreate,
    )

DEFAULT_VIETNAM_COVID_DATA_TIMESERIES(datasets_dir; recreate = false) =
    VnExpressData.save_cases_timeseries(
        datasets_dir,
        "vietnam-covid-data-timeseries",
        Date(2021, 4, 27),
        Date(2021, 10, 13),
        recreate = recreate,
    )

DEFAULT_VIETNAM_AVERAGE_MOVEMENT_RANGE(datasets_dir; recreate = false) =
    FacebookData.save_country_average_movement_range(
        joinpath(
            datasets_dir,
            "facebook",
            "movement-range-data-2021-10-09",
            "movement-range-2021-10-09.txt",
        ),
        datasets_dir,
        "facebook-average-movement-range",
        "VNM",
        recreate = recreate,
    )

DEFAULT_VIETNAM_INTRA_CONNECTEDNESS_INDEX(datasets_dir; recreate = false) =
    FacebookData.save_intra_country_gadm1_nuts2_connectedness_index(
        joinpath(
            datasets_dir,
            "facebook",
            "social-connectedness-index",
            "gadm1_nuts2_gadm1_nuts2_aug2020.tsv",
        ),
        datasets_dir,
        "facebook-intra-connectedness-index",
        "VNM",
        recreate = recreate,
    )

end
