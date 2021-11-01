module Covid19ModelVN

using DataDeps

function __init__()
    register(
        DataDep(
            "covid19model",
            """
            Dataset: Population and Social Proximty to Cases
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
                "https://github.com/letung3105/coviddata/raw/master/covid19model/timeseries-covid19-combined-vietnam.csv",
            ],
        ),
    )
    return nothing
end

include("helpers.jl")
include("models.jl")
include("train_eval.jl")

module FacebookData
include("data_facebook.jl")
end

module JHUCSSEData
include("data_jhu_csse.jl")
end

module PopulationData
include("data_population.jl")
end

module VnExpressData
include("data_vnexpress.jl")
end

module VnCdcData
include("data_vncdc.jl")
end

end
