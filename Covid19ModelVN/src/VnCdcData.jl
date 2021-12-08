module VnCdcData

using DataDeps, DataFrames, Dates
using JSON: JSON

function __init__()
    register(
        DataDep(
            "vncdc",
            """
            Dataset: VnCDC Covid-19 Dashboard API Data
            Website: https://ncov.vncdc.gov.vn
            """,
            [
                "https://github.com/letung3105/coviddata/raw/master/vncdc/HoChiMinh.json"
                "https://github.com/letung3105/coviddata/raw/master/vncdc/BinhDuong.json"
                "https://github.com/letung3105/coviddata/raw/master/vncdc/DongNai.json"
                "https://github.com/letung3105/coviddata/raw/master/vncdc/LongAn.json"
            ],
        ),
    )
    return nothing
end

"""
Parse an array of pairs of a date and a value in JSON format

# Arguments

+ `data`: an arbitrary JSON data
"""
function parse_json_date_value_pairs(data::Any)
    dates = Vector{Date}()
    values = Vector{Int}()
    for data_point in data
        push!(dates, Date(Dates.unix2datetime(data_point[1]//1000)))
        push!(values, data_point[2])
    end
    return dates, values
end

"""
Read and parse the JSON for confirmed cases and deaths into a dataframe

# Arguments

+ `fpath`: file path to the JSON data file
"""
function read_timeseries_confirmed_and_deaths(fpath::AbstractString)
    data = JSON.parsefile(fpath)
    column_names = ["confirmed_community", "confirmed_quarantined", "deaths"]
    dfs = Vector{DataFrame}()
    for (i, colname) in enumerate(column_names)
        dates, cases = parse_json_date_value_pairs(data["report"][i]["data"])
        df = DataFrame(["date" => dates, colname => cases])
        push!(dfs, df)
    end

    df = innerjoin(dfs...; on=:date)
    transform!(
        df,
        [:confirmed_community, :confirmed_quarantined] => ((x, y) -> x .+ y) => :confirmed,
    )
    transform!(
        df, :deaths => cumsum => :deaths_total, :confirmed => cumsum => :confirmed_total
    )

    return df
end

end
