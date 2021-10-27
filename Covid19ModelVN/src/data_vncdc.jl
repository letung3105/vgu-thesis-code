using DataDeps, DataFrames, Dates
import JSON

function __init__()
    register(
        DataDep(
            "vncdc",
            """
            Dataset: VnCDC Covid-19 Dashboard API Data
            Website: https://ncov.vncdc.gov.vn
            """,
            [
                "https://github.com/letung3105/vgu-thesis-datasets/raw/master/vncdc/HoChiMinh.json"
                "https://github.com/letung3105/vgu-thesis-datasets/raw/master/vncdc/BinhDuong.json"
                "https://github.com/letung3105/vgu-thesis-datasets/raw/master/vncdc/DongNai.json"
                "https://github.com/letung3105/vgu-thesis-datasets/raw/master/vncdc/LongAn.json"
            ],
        ),
    )
    return nothing
end

function parse_json_date_value_pairs(data)
    dates = Vector{Date}()
    values = Vector{Int}()
    for data_point ∈ data
        push!(dates, Date(Dates.unix2datetime(data_point[1] // 1000)))
        push!(values, data_point[2])
    end
    return dates, values
end

function read_timeseries_confirmed_and_deaths(fpath)
    data = JSON.parsefile(fpath)
    column_names = ["confirmed_community", "confirmed_quarantined", "deaths"]
    dfs = Vector{DataFrame}()
    for (i, colname) ∈ enumerate(column_names)
        dates, cases = parse_json_date_value_pairs(data["report"][i]["data"])
        df = DataFrame(["date" => dates, colname => cases])
        push!(dfs, df)
    end

    df = innerjoin(dfs..., on = :date)
    transform!(
        df,
        [:confirmed_community, :confirmed_quarantined] => ((x, y) -> x .+ y) => :confirmed,
    )
    transform!(
        df,
        :deaths => cumsum => :deaths_total,
        :confirmed => cumsum => :confirmed_total,
    )

    return df
end