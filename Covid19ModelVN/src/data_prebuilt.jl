using DataDeps, DataFrames, CSV

import Covid19ModelVN.VnCdcData

export get_prebuilt_covid_timeseries,
    get_prebuilt_population, get_prebuilt_movement_range, get_prebuilt_social_proximity

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

function __init__()
    register(
        DataDep(
            "covid19model",
            """
            Dataset: Datsets for the experiments
            """,
            [
                "https://github.com/letung3105/coviddata/raw/master/covid19model/average-population-vn-provinces.csv",
                "https://github.com/letung3105/coviddata/raw/master/covid19model/average-population-us-counties.csv",
                "https://github.com/letung3105/coviddata/raw/master/covid19model/social-proximity-to-cases-vn-provinces.csv",
                "https://github.com/letung3105/coviddata/raw/master/covid19model/social-proximity-to-cases-us-counties.csv",
                "https://github.com/letung3105/coviddata/raw/master/covid19model/movement-range-vietnam.csv",
                "https://github.com/letung3105/coviddata/raw/master/covid19model/movement-range-hcm-city.csv",
                "https://github.com/letung3105/coviddata/raw/master/covid19model/movement-range-binh-duong.csv",
                "https://github.com/letung3105/coviddata/raw/master/covid19model/movement-range-dong-nai.csv",
                "https://github.com/letung3105/coviddata/raw/master/covid19model/movement-range-long-an.csv",
                "https://github.com/letung3105/coviddata/raw/master/covid19model/movement-range-united-states.csv",
                "https://github.com/letung3105/coviddata/raw/master/covid19model/movement-range-united-states-2020.csv",
                "https://github.com/letung3105/coviddata/raw/master/covid19model/movement-range-los-angeles-CA.csv",
                "https://github.com/letung3105/coviddata/raw/master/covid19model/movement-range-los-angeles-CA-2020.csv",
                "https://github.com/letung3105/coviddata/raw/master/covid19model/movement-range-cook-county-IL.csv",
                "https://github.com/letung3105/coviddata/raw/master/covid19model/movement-range-cook-county-IL-2020.csv",
                "https://github.com/letung3105/coviddata/raw/master/covid19model/movement-range-harris-county-TX.csv",
                "https://github.com/letung3105/coviddata/raw/master/covid19model/movement-range-harris-county-TX-2020.csv",
                "https://github.com/letung3105/coviddata/raw/master/covid19model/movement-range-maricopa-county-AZ.csv",
                "https://github.com/letung3105/coviddata/raw/master/covid19model/movement-range-maricopa-county-AZ-2020.csv",
                "https://github.com/letung3105/coviddata/raw/master/covid19model/timeseries-covid19-combined-vietnam.csv",
                "https://github.com/letung3105/coviddata/raw/master/covid19model/timeseries-covid19-combined-united-states.csv",
                "https://github.com/letung3105/coviddata/raw/master/covid19model/timeseries-covid19-combined-los-angeles-CA.csv",
                "https://github.com/letung3105/coviddata/raw/master/covid19model/timeseries-covid19-combined-cook-county-IL.csv",
                "https://github.com/letung3105/coviddata/raw/master/covid19model/timeseries-covid19-combined-harris-county-TX.csv",
                "https://github.com/letung3105/coviddata/raw/master/covid19model/timeseries-covid19-combined-maricopa-county-AZ.csv",
            ],
        ),
    )
    return nothing
end

function get_prebuilt_covid_timeseries(location_code::AbstractString)
    vncdc_data = Dict([
        LOC_CODE_HCM_CITY => datadep"vncdc/HoChiMinh.json",
        LOC_CODE_BINH_DUONG => datadep"vncdc/BinhDuong.json",
        LOC_CODE_DONG_NAI => datadep"vncdc/DongNai.json",
        LOC_CODE_LONG_AN => datadep"vncdc/LongAn.json",
    ])
    jhu_data = Dict([
        LOC_CODE_VIETNAM =>
            datadep"covid19model/timeseries-covid19-combined-vietnam.csv",
        LOC_CODE_UNITED_STATES =>
            datadep"covid19model/timeseries-covid19-combined-united-states.csv",
        LOC_CODE_LOS_ANGELES_CA =>
            datadep"covid19model/timeseries-covid19-combined-los-angeles-CA.csv",
        LOC_CODE_COOK_IL =>
            datadep"covid19model/timeseries-covid19-combined-cook-county-IL.csv",
        LOC_CODE_HARRIS_TX =>
            datadep"covid19model/timeseries-covid19-combined-harris-county-TX.csv",
        LOC_CODE_MARICOPA_AZ =>
            datadep"covid19model/timeseries-covid19-combined-maricopa-county-AZ.csv",
    ])

    df = if location_code ∈ keys(vncdc_data)
        VnCdcData.read_timeseries_confirmed_and_deaths(vncdc_data[location_code])
    elseif location_code ∈ keys(jhu_data)
        CSV.read(jhu_data[location_code], DataFrame)
    else
        throw("Unsupported location code '$location_code'!")
    end
    return df
end

function get_prebuilt_population(location_code::AbstractString)
    fpath, locname = if location_code == LOC_CODE_VIETNAM
        return 97_582_700
    elseif location_code == LOC_CODE_UNITED_STATES
        return 332_889_844
    elseif location_code ∈ keys(LOC_NAMES_VN)
        datadep"covid19model/average-population-vn-provinces.csv", LOC_NAMES_VN[location_code]
    elseif location_code ∈ keys(LOC_NAMES_US)
        datadep"covid19model/average-population-us-counties.csv", LOC_NAMES_US[location_code]
    else
        throw("Unsupported location code '$location_code'!")
    end

    df_population = CSV.read(fpath, DataFrame)
    population = first(filter(x -> x.NAME_1 == locname, df_population).AVGPOPULATION)
    return population
end

function get_prebuilt_movement_range(location_code::AbstractString)
    data = Dict([
        LOC_CODE_VIETNAM => datadep"covid19model/movement-range-vietnam.csv",
        LOC_CODE_HCM_CITY => datadep"covid19model/movement-range-hcm-city.csv",
        LOC_CODE_BINH_DUONG => datadep"covid19model/movement-range-binh-duong.csv",
        LOC_CODE_DONG_NAI => datadep"covid19model/movement-range-dong-nai.csv",
        LOC_CODE_LONG_AN => datadep"covid19model/movement-range-long-an.csv",
        LOC_CODE_UNITED_STATES =>
            datadep"covid19model/movement-range-united-states-2020.csv",
        LOC_CODE_LOS_ANGELES_CA =>
            datadep"covid19model/movement-range-los-angeles-CA-2020.csv",
        LOC_CODE_COOK_IL =>
            datadep"covid19model/movement-range-cook-county-IL-2020.csv",
        LOC_CODE_HARRIS_TX =>
            datadep"covid19model/movement-range-harris-county-TX-2020.csv",
        LOC_CODE_MARICOPA_AZ =>
            datadep"covid19model/movement-range-maricopa-county-AZ-2020.csv",
    ])

    if location_code ∉ keys(data)
        throw("Unsupported location code '$location_code'!")
    end
    return CSV.read(data[location_code], DataFrame)
end

function get_prebuilt_social_proximity(location_code::AbstractString)
    fpath, locname = if location_code ∈ keys(LOC_NAMES_VN)
        datadep"covid19model/social-proximity-to-cases-vn-provinces.csv",
        LOC_NAMES_VN[location_code]
    elseif location_code ∈ keys(LOC_NAMES_US)
        datadep"covid19model/social-proximity-to-cases-us-counties.csv",
        LOC_NAMES_US[location_code]
    else
        throw("Unsupported location code '$location_code'!")
    end

    df = CSV.read(fpath, DataFrame)
    df = df[!, ["date", locname]]
    return df, locname
end
