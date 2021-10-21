module VnCdcData

using DataFrames, Dates
import JSON

function parse_json_date_value_pairs(data)
    dates = Vector{Date}()
    values = Vector{Int}()
    for data_point in data
        push!(dates, Date(Dates.unix2datetime(data_point[1] // 1000)))
        push!(values, data_point[2])
    end
    return dates, values
end

function parse_json_cases_and_deaths(fpath)
    data = JSON.parsefile(fpath)
    data = data["report"]

    column_names = ["confirmed_community", "confirmed_quarantined", "dead"]
    dfs = Vector{DataFrame}()
    for (i, colname) in enumerate(column_names)
        dates, cases = parse_json_date_value_pairs(data[i]["data"])
        df = DataFrame(["date" => dates, colname => cases])
        push!(dfs, df)
    end

    df = innerjoin(dfs..., on = :date)
    transform!(
        df,
        [:confirmed_community, :confirmed_quarantined] => ((x, y) -> x .+ y) => :confirmed,
    )
    transform!(df, :dead => cumsum => :dead_total, :confirmed => cumsum => :confirmed_total)

    return df
end

end
