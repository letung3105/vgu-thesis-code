const LOC_CODE_VIETNAM = "vietnam"
const LOC_CODE_HCM_CITY = "hcm"
const LOC_CODE_BINH_DUONG = "binhduong"
const LOC_CODE_DONG_NAI = "dongnai"
const LOC_CODE_LONG_AN = "longan"

const LOC_NAMES_VN = Dict([
    LOC_CODE_HCM_CITY => "Hồ Chí Minh city",
    LOC_CODE_BINH_DUONG => "Bình Dương",
    LOC_CODE_DONG_NAI => "Đồng Nai",
    LOC_CODE_LONG_AN => "Long An",
])

const LOC_CODE_UNITED_STATES = "unitedstates"
const LOC_CODE_LOS_ANGELES_CA = "losangeles_ca"
const LOC_CODE_COOK_IL = "cook_il"
const LOC_CODE_HARRIS_TX = "harris_tx"
const LOC_CODE_MARICOPA_AZ = "maricopa_az"

const LOC_NAMES_US = Dict([
    LOC_CODE_LOS_ANGELES_CA => "Los Angeles, California, US",
    LOC_CODE_COOK_IL => "Cook, Illinois, US",
    LOC_CODE_HARRIS_TX => "Harris, Texas, US",
    LOC_CODE_MARICOPA_AZ => "Maricopa, Arizona, US",
])

const DATADEP_NAME = "covid19model"
const FNAME_AVERAGE_POPULATION_VN_PROVINCES = "average-population-vn-provinces.csv"
const FNAME_AVERAGE_POPULATION_US_COUNTIES = "average-population-us-counties.csv"
const FNAME_SOCIAL_PROXIMITY_TO_CASES_VN_PROVINCES = "social-proximity-to-cases-vn-provinces.csv"
const FNAME_SOCIAL_PROXIMITY_TO_CASES_US_COUNTIES = "social-proximity-to-cases-us-counties.csv"
const FNAME_MOVEMENT_RANGE_VIETNAM = "movement-range-vietnam.csv"
const FNAME_MOVEMENT_RANGE_HCM_CITY = "movement-range-hcm-city.csv"
const FNAME_MOVEMENT_RANGE_BINH_DUONG = "movement-range-binh-duong.csv"
const FNAME_MOVEMENT_RANGE_DONG_NAI = "movement-range-dong-nai.csv"
const FNAME_MOVEMENT_RANGE_LONG_AN = "movement-range-long-an.csv"
const FNAME_MOVEMENT_RANGE_UNITED_STATES = "movement-range-united-states.csv"
const FNAME_MOVEMENT_RANGE_UNITED_STATES_2020 = "movement-range-united-states-2020.csv"
const FNAME_MOVEMENT_RANGE_LOS_ANGELES_CA = "movement-range-los-angeles-CA.csv"
const FNAME_MOVEMENT_RANGE_LOS_ANGELES_CA_2020 = "movement-range-los-angeles-CA-2020.csv"
const FNAME_MOVEMENT_RANGE_COOK_COUNTY_IL = "movement-range-cook-county-IL.csv"
const FNAME_MOVEMENT_RANGE_COOK_COUNTY_IL_2020 = "movement-range-cook-county-IL-2020.csv"
const FNAME_MOVEMENT_RANGE_HARRIS_COUNTY_TX = "movement-range-harris-county-TX.csv"
const FNAME_MOVEMENT_RANGE_HARRIS_COUNTY_TX_2020 = "movement-range-harris-county-TX-2020.csv"
const FNAME_MOVEMENT_RANGE_MARICOPA_COUNTY_AZ = "movement-range-maricopa-county-AZ.csv"
const FNAME_MOVEMENT_RANGE_MARICOPA_COUNTY_AZ_2020 = "movement-range-maricopa-county-AZ-2020.csv"
const FNAME_COVID19_TIMESERIES_VIETNAM = "timeseries-covid19-combined-vietnam.csv"
const FNAME_COVID19_TIMESERIES_UNITED_STATES = "timeseries-covid19-combined-united-states.csv"
const FNAME_COVID19_TIMESERIES_LOS_ANGELES_CA = "timeseries-covid19-combined-los-angeles-CA.csv"
const FNAME_COVID19_TIMESERIES_COOK_COUNTY_IL = "timeseries-covid19-combined-cook-county-IL.csv"
const FNAME_COVID19_TIMESERIES_HARRIS_COUNTY_TX = "timeseries-covid19-combined-harris-county-TX.csv"
const FNAME_COVID19_TIMESERIES_MARICOPA_COUNTY_AZ = "timeseries-covid19-combined-maricopa-county-AZ.csv"

const DATAROOT_COVID19MODEL = "https://github.com/letung3105/coviddata/raw/master/covid19model/"
const COVID19MODEL_FILES = String[
    joinpath(DATAROOT_COVID19MODEL, FNAME_AVERAGE_POPULATION_VN_PROVINCES),
    joinpath(DATAROOT_COVID19MODEL, FNAME_AVERAGE_POPULATION_US_COUNTIES),
    joinpath(DATAROOT_COVID19MODEL, FNAME_SOCIAL_PROXIMITY_TO_CASES_VN_PROVINCES),
    joinpath(DATAROOT_COVID19MODEL, FNAME_SOCIAL_PROXIMITY_TO_CASES_US_COUNTIES),
    joinpath(DATAROOT_COVID19MODEL, FNAME_MOVEMENT_RANGE_VIETNAM),
    joinpath(DATAROOT_COVID19MODEL, FNAME_MOVEMENT_RANGE_HCM_CITY),
    joinpath(DATAROOT_COVID19MODEL, FNAME_MOVEMENT_RANGE_BINH_DUONG),
    joinpath(DATAROOT_COVID19MODEL, FNAME_MOVEMENT_RANGE_DONG_NAI),
    joinpath(DATAROOT_COVID19MODEL, FNAME_MOVEMENT_RANGE_LONG_AN),
    joinpath(DATAROOT_COVID19MODEL, FNAME_MOVEMENT_RANGE_UNITED_STATES),
    joinpath(DATAROOT_COVID19MODEL, FNAME_MOVEMENT_RANGE_UNITED_STATES_2020),
    joinpath(DATAROOT_COVID19MODEL, FNAME_MOVEMENT_RANGE_LOS_ANGELES_CA),
    joinpath(DATAROOT_COVID19MODEL, FNAME_MOVEMENT_RANGE_LOS_ANGELES_CA_2020),
    joinpath(DATAROOT_COVID19MODEL, FNAME_MOVEMENT_RANGE_COOK_COUNTY_IL),
    joinpath(DATAROOT_COVID19MODEL, FNAME_MOVEMENT_RANGE_COOK_COUNTY_IL_2020),
    joinpath(DATAROOT_COVID19MODEL, FNAME_MOVEMENT_RANGE_HARRIS_COUNTY_TX),
    joinpath(DATAROOT_COVID19MODEL, FNAME_MOVEMENT_RANGE_HARRIS_COUNTY_TX_2020),
    joinpath(DATAROOT_COVID19MODEL, FNAME_MOVEMENT_RANGE_MARICOPA_COUNTY_AZ),
    joinpath(DATAROOT_COVID19MODEL, FNAME_MOVEMENT_RANGE_MARICOPA_COUNTY_AZ_2020),
    joinpath(DATAROOT_COVID19MODEL, FNAME_COVID19_TIMESERIES_VIETNAM),
    joinpath(DATAROOT_COVID19MODEL, FNAME_COVID19_TIMESERIES_UNITED_STATES),
    joinpath(DATAROOT_COVID19MODEL, FNAME_COVID19_TIMESERIES_LOS_ANGELES_CA),
    joinpath(DATAROOT_COVID19MODEL, FNAME_COVID19_TIMESERIES_COOK_COUNTY_IL),
    joinpath(DATAROOT_COVID19MODEL, FNAME_COVID19_TIMESERIES_HARRIS_COUNTY_TX),
    joinpath(DATAROOT_COVID19MODEL, FNAME_COVID19_TIMESERIES_MARICOPA_COUNTY_AZ),
]

function __init__()
    register(
        DataDep(DATADEP_NAME, "Dataset: Datsets for the experiments", COVID19MODEL_FILES)
    )
    return nothing
end

"""
    get_prebuilt_covid_timeseries(location_code::AbstractString)

Get the Covid-19 time series, that is created for the model, for a location

# Arguments

* `location_code`: short code that uniquely identify the location
"""
function get_prebuilt_covid_timeseries(location_code::AbstractString)
    vncdc_data = Dict([
        LOC_CODE_HCM_CITY => datadep"vncdc/HoChiMinh.json",
        LOC_CODE_BINH_DUONG => datadep"vncdc/BinhDuong.json",
        LOC_CODE_DONG_NAI => datadep"vncdc/DongNai.json",
        LOC_CODE_LONG_AN => datadep"vncdc/LongAn.json",
    ])
    jhu_data = Dict([
        LOC_CODE_VIETNAM => @datadep_str("$DATADEP_NAME/$FNAME_COVID19_TIMESERIES_VIETNAM"),
        LOC_CODE_UNITED_STATES =>
            @datadep_str("$DATADEP_NAME/$FNAME_COVID19_TIMESERIES_UNITED_STATES"),
        LOC_CODE_LOS_ANGELES_CA =>
            @datadep_str("$DATADEP_NAME/$FNAME_COVID19_TIMESERIES_LOS_ANGELES_CA"),
        LOC_CODE_COOK_IL =>
            @datadep_str("$DATADEP_NAME/$FNAME_COVID19_TIMESERIES_COOK_COUNTY_IL"),
        LOC_CODE_HARRIS_TX =>
            @datadep_str("$DATADEP_NAME/$FNAME_COVID19_TIMESERIES_HARRIS_COUNTY_TX"),
        LOC_CODE_MARICOPA_AZ =>
            @datadep_str("$DATADEP_NAME/$FNAME_COVID19_TIMESERIES_MARICOPA_COUNTY_AZ"),
    ])

    df = if location_code ∈ keys(vncdc_data)
        VnCdcData.read_timeseries_confirmed_and_deaths(vncdc_data[location_code])
    elseif location_code ∈ keys(jhu_data)
        CSV.read(jhu_data[location_code], DataFrame)
    else
        throw("Unsupported location code '$location_code'!")
    end
    df[!, :date] .= Date.(df[!, :date])

    return df
end

"""
    make_covid_timeseries(dir::AbstractString; recreate::Bool = false)

Create all the combined Covid-19 datasets that are used by the experiments

# Arguments

* `dir`: the path to the directory where the datasets are saved
* `recreate`: choose whether to overwrite a dataset that already exists in the directory
"""
function make_covid_timeseries(dir::AbstractString; recreate::Bool=false)
    JHUCSSEData.save_country_level_timeseries(
        [
            JHUCSSEData.CountryCovidTimeseriesFile(
                joinpath(dir, FNAME_COVID19_TIMESERIES_VIETNAM), "Vietnam"
            ),
            JHUCSSEData.CountryCovidTimeseriesFile(
                joinpath(dir, FNAME_COVID19_TIMESERIES_UNITED_STATES), "US"
            ),
        ];
        recreate,
    )
    JHUCSSEData.save_us_county_level_timeseries(
        [
            JHUCSSEData.CountyCovidTimeseriesFile(
                joinpath(dir, FNAME_COVID19_TIMESERIES_LOS_ANGELES_CA),
                "California",
                "Los Angeles",
            ),
            JHUCSSEData.CountyCovidTimeseriesFile(
                joinpath(dir, FNAME_COVID19_TIMESERIES_COOK_COUNTY_IL), "Illinois", "Cook"
            ),
            JHUCSSEData.CountyCovidTimeseriesFile(
                joinpath(dir, FNAME_COVID19_TIMESERIES_HARRIS_COUNTY_TX), "Texas", "Harris"
            ),
            JHUCSSEData.CountyCovidTimeseriesFile(
                joinpath(dir, FNAME_COVID19_TIMESERIES_MARICOPA_COUNTY_AZ),
                "Arizona",
                "Maricopa",
            ),
        ];
        recreate,
    )
    return nothing
end

"""
    get_prebuilt_population(location_code::AbstractString)

Get the population data along with the standardized location identifier,
that is created for the model, for a location

# Arguments

* `location_code`: short code that uniquely identify the location
"""
function get_prebuilt_population(location_code::AbstractString)
    fpath, locname = if location_code == LOC_CODE_VIETNAM
        return 97_582_700
    elseif location_code == LOC_CODE_UNITED_STATES
        return 332_889_844
    elseif location_code ∈ keys(LOC_NAMES_VN)
        @datadep_str("$DATADEP_NAME/$FNAME_AVERAGE_POPULATION_VN_PROVINCES"),
        LOC_NAMES_VN[location_code]
    elseif location_code ∈ keys(LOC_NAMES_US)
        @datadep_str("$DATADEP_NAME/$FNAME_AVERAGE_POPULATION_US_COUNTIES"),
        LOC_NAMES_US[location_code]
    else
        throw("Unsupported location code '$location_code'!")
    end

    df_population = CSV.read(fpath, DataFrame)
    population = first(filter(x -> x.NAME_1 == locname, df_population).AVGPOPULATION)

    return population
end

"""
    make_population(dir::AbstractString; recreate::Bool = false)

Create all the population datasets that are used by the experiments

# Arguments

* `dir`: the path to the directory where the datasets are saved
* `recreate`: choose whether to overwrite a dataset that already exists in the directory
"""
function make_population(dir::AbstractString; recreate::Bool=false)
    PopulationData.save_vietnam_province_level_gadm_and_gso_population(
        joinpath(dir, FNAME_AVERAGE_POPULATION_VN_PROVINCES); recreate
    )
    JHUCSSEData.save_us_counties_population(
        joinpath(dir, FNAME_AVERAGE_POPULATION_US_COUNTIES); recreate
    )
    return nothing
end

"""
    get_prebuilt_movement_range(location_code::AbstractString)

Get the average movement range maps dataset, that is created for the model, for a location

# Arguments

* `location_code`: short code that uniquely identify the location
"""
function get_prebuilt_movement_range(location_code::AbstractString)
    data = Dict([
        LOC_CODE_VIETNAM => @datadep_str("$DATADEP_NAME/$FNAME_MOVEMENT_RANGE_VIETNAM"),
        LOC_CODE_HCM_CITY => @datadep_str("$DATADEP_NAME/$FNAME_MOVEMENT_RANGE_HCM_CITY"),
        LOC_CODE_BINH_DUONG =>
            @datadep_str("$DATADEP_NAME/$FNAME_MOVEMENT_RANGE_BINH_DUONG"),
        LOC_CODE_DONG_NAI => @datadep_str("$DATADEP_NAME/$FNAME_MOVEMENT_RANGE_DONG_NAI"),
        LOC_CODE_LONG_AN => @datadep_str("$DATADEP_NAME/$FNAME_MOVEMENT_RANGE_LONG_AN"),
        LOC_CODE_UNITED_STATES =>
            @datadep_str("$DATADEP_NAME/$FNAME_MOVEMENT_RANGE_UNITED_STATES"),
        LOC_CODE_LOS_ANGELES_CA =>
            @datadep_str("$DATADEP_NAME/$FNAME_MOVEMENT_RANGE_LOS_ANGELES_CA"),
        LOC_CODE_COOK_IL =>
            @datadep_str("$DATADEP_NAME/$FNAME_MOVEMENT_RANGE_COOK_COUNTY_IL"),
        LOC_CODE_HARRIS_TX =>
            @datadep_str("$DATADEP_NAME/$FNAME_MOVEMENT_RANGE_HARRIS_COUNTY_TX"),
        LOC_CODE_MARICOPA_AZ =>
            @datadep_str("$DATADEP_NAME/$FNAME_MOVEMENT_RANGE_MARICOPA_COUNTY_AZ"),
    ])

    if location_code ∉ keys(data)
        throw("Unsupported location code '$location_code'!")
    end

    df = CSV.read(data[location_code], DataFrame)
    df[!, :ds] .= Date.(df[!, :ds])

    return df
end

"""
    make_movement_range(dir::AbstractString; recreate::Bool = false)

Create all the average indices from the movement range maps dataset

# Arguments

* `dir`: the path to the directory where the datasets are saved
* `recreate`: choose whether to overwrite a dataset that already exists in the directory
"""
function make_movement_range(dir::AbstractString; recreate::Bool=false)
    FacebookData.save_region_average_movement_range(
        [
            FacebookData.RegionMovementRangeFile(
                joinpath(dir, FNAME_MOVEMENT_RANGE_VIETNAM), "VNM", nothing
            ),
            FacebookData.RegionMovementRangeFile(
                joinpath(dir, FNAME_MOVEMENT_RANGE_HCM_CITY), "VNM", 26
            ),
            FacebookData.RegionMovementRangeFile(
                joinpath(dir, FNAME_MOVEMENT_RANGE_BINH_DUONG), "VNM", 10
            ),
            FacebookData.RegionMovementRangeFile(
                joinpath(dir, FNAME_MOVEMENT_RANGE_DONG_NAI), "VNM", 2
            ),
            FacebookData.RegionMovementRangeFile(
                joinpath(dir, FNAME_MOVEMENT_RANGE_LONG_AN), "VNM", 39
            ),
            FacebookData.RegionMovementRangeFile(
                joinpath(dir, FNAME_MOVEMENT_RANGE_UNITED_STATES), "USA", nothing
            ),
            FacebookData.RegionMovementRangeFile(
                joinpath(dir, FNAME_MOVEMENT_RANGE_LOS_ANGELES_CA), "USA", 6037
            ),
            FacebookData.RegionMovementRangeFile(
                joinpath(dir, FNAME_MOVEMENT_RANGE_COOK_COUNTY_IL), "USA", 17031
            ),
            FacebookData.RegionMovementRangeFile(
                joinpath(dir, FNAME_MOVEMENT_RANGE_HARRIS_COUNTY_TX), "USA", 13145
            ),
            FacebookData.RegionMovementRangeFile(
                joinpath(dir, FNAME_MOVEMENT_RANGE_MARICOPA_COUNTY_AZ), "USA", 4013
            ),
        ];
        recreate,
    )
    FacebookData.save_region_average_movement_range(
        [
            FacebookData.RegionMovementRangeFile(
                joinpath(dir, FNAME_MOVEMENT_RANGE_UNITED_STATES_2020), "USA", nothing
            ),
            FacebookData.RegionMovementRangeFile(
                joinpath(dir, FNAME_MOVEMENT_RANGE_LOS_ANGELES_CA_2020), "USA", 6037
            ),
            FacebookData.RegionMovementRangeFile(
                joinpath(dir, FNAME_MOVEMENT_RANGE_COOK_COUNTY_IL_2020), "USA", 17031
            ),
            FacebookData.RegionMovementRangeFile(
                joinpath(dir, FNAME_MOVEMENT_RANGE_HARRIS_COUNTY_TX_2020), "USA", 13145
            ),
            FacebookData.RegionMovementRangeFile(
                joinpath(dir, FNAME_MOVEMENT_RANGE_MARICOPA_COUNTY_AZ_2020), "USA", 4013
            ),
        ];
        fpath_movement_range=datadep"facebook/movement-range-data-2020-03-01--2020-12-31.txt",
        recreate,
    )
    return nothing
end

"""
    get_prebuilt_social_proximity(location_code::AbstractString)

Get the social proximity to cases index datasets, that is created for the model, for a location

# Arguments

* `location_code`: short code that uniquely identify the location
"""
function get_prebuilt_social_proximity(location_code::AbstractString)
    fpath, locname = if location_code ∈ keys(LOC_NAMES_VN)
        @datadep_str("$DATADEP_NAME/$FNAME_SOCIAL_PROXIMITY_TO_CASES_VN_PROVINCES"),
        LOC_NAMES_VN[location_code]
    elseif location_code ∈ keys(LOC_NAMES_US)
        @datadep_str("$DATADEP_NAME/$FNAME_SOCIAL_PROXIMITY_TO_CASES_US_COUNTIES"),
        LOC_NAMES_US[location_code]
    else
        throw("Unsupported location code '$location_code'!")
    end

    df = CSV.read(fpath, DataFrame)
    df = df[!, ["date", locname]]
    df[!, :date] .= Date.(df[!, :date])

    return df, locname
end

"""
    make_social_proximity(
        dir::AbstractString;
        recreate::Bool = false,
        fpath_population_vn = joinpath(dir, FNAME_AVERAGE_POPULATION_VN_PROVINCES),
        fpath_population_us = joinpath(dir, FNAME_AVERAGE_POPULATION_US_COUNTIES),
    )

Create all the social proximity to cases datasets that are used by the experiments

# Arguments

* `dir`: the path to the directory where the datasets are saved
* `recreate`: choose whether to overwrite a dataset that already exists in the directory
"""
function make_social_proximity(
    dir::AbstractString;
    recreate::Bool=false,
    fpath_population_vn=joinpath(dir, FNAME_AVERAGE_POPULATION_VN_PROVINCES),
    fpath_population_us=joinpath(dir, FNAME_AVERAGE_POPULATION_US_COUNTIES),
)
    fpath_spc_vn = joinpath(dir, FNAME_SOCIAL_PROXIMITY_TO_CASES_VN_PROVINCES)
    if !isfile(fpath_spc_vn) || recreate
        if !isdir(dir)
            mkpath(dir)
        end

        @info "Reading social connectedness index dataset"
        df_sci = FacebookData.read_social_connectedness(
            datadep"facebook/gadm1_nuts2_gadm1_nuts2.tsv"
        )
        df_sci_vn = FacebookData.inter_province_social_connectedness(df_sci, "VNM")
        @info "Reading population data"
        df_population = CSV.read(fpath_population_vn, DataFrame)
        @info "Reading Covid-19 timeseries"
        df_covid = CSV.read(
            datadep"vnexpress/timeseries-vietnam-provinces-confirmed.csv", DataFrame
        )

        @info "Calculating social proximity to cases index"
        df_scp_vn = FacebookData.calculate_social_proximity_to_cases(
            df_population, df_covid, df_sci_vn
        )
        save_dataframe(df_scp_vn, fpath_spc_vn)
    end

    fpath_spc_us = joinpath(dir, FNAME_SOCIAL_PROXIMITY_TO_CASES_US_COUNTIES)
    if !isfile(fpath_spc_us) || recreate
        if !isdir(dir)
            mkpath(dir)
        end

        @info "Reading social connectedness index dataset"
        df_sci_counties = FacebookData.read_social_connectedness(
            datadep"facebook/county_county.tsv"
        )
        @info "Reading population data"
        df_population = CSV.read(fpath_population_us, DataFrame)
        @info "Reading Covid-19 timeseries"
        df_covid = JHUCSSEData.get_us_counties_timeseries_confirmed(
            CSV.read(datadep"jhu-csse/time_series_covid19_confirmed_US.csv", DataFrame)
        )

        @info "Calculating social proximity to cases index"
        df_scp_us = FacebookData.calculate_social_proximity_to_cases(
            df_population, df_covid, df_sci_counties
        )
        save_dataframe(df_scp_us, fpath_spc_us)
    end

    return nothing
end
