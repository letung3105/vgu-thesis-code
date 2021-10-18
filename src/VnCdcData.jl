module VnCdcData

export get_cases_timeseries

using CSV
using Dates
using DataFrames
using JSON
using HTTP

const HOST = "ncov.vncdc.gov.vn"
const PATHS = ["/v2/vietnam/report-epi", "/v2/vietnam/type-injection", "/v2/vietnam/provinces"]
const REQUEST_DATE_FORMAT = dateformat"yyyy-mm-dd"

function request_json(data...; kwargs...)
    res = HTTP.get(data..., kwargs...)
    JSON.parse(String(res.body))
end

function get_province_data(path, province_id, start_time, end_time)
    query = Dict(
        :start_time => Dates.format(start_time, REQUEST_DATE_FORMAT),
        :end_time => Dates.format(end_time, REQUEST_DATE_FORMAT),
        :province_ids => province_id
    )
    uri = HTTP.URI(scheme="https", host=HOST, path=path, query=query)
    request_json(uri)
end

function parse_date_value_pairs(data)
    dates = Vector{Date}()
    values = Vector{Int}()
    for data_point in data
        push!(dates, Date(Dates.unix2datetime(data_point[1] // 1000)))
        push!(values, data_point[2])
    end
    dates, values
end

function parse_cases_and_deaths(data)
    column_names = ["confirmed", "deaths"]
    data = data["report"]
    dfs = Vector{DataFrame}()
    for i in 1:2
        dates, cases = parse_date_value_pairs(data[i]["data"])
        df = DataFrame(["date" => dates, column_names[i] => cases])
        push!(dfs, df)
    end
    innerjoin(dfs..., on=:date)
end

function parse_cases_composition(data)
    data = data[1]
    dfs = Vector{DataFrame}()
    for series in data["series"]
        dates, cases = parse_date_value_pairs(series["data"])
        df = DataFrame(["date" => dates, series["name"] => cases])
        push!(dfs, df)
    end

    df = innerjoin(dfs..., on=:date)
    rename!(df, [
        "Khu phong tỏa" => :confirmed_blockade,
        "Khu cách ly" => :confirmed_quarantined,
        "Cộng đồng" => :confirmed_unquarantined,
        "Sàng lọc tại CSYT" => :confirmed_screening,
        "Không rõ" => :confirmed_unknown,
    ])
end

function get_cases_timeseries(
    fpath::AbstractString, startdate::Date, enddate::Date;
    redownload=false
)
    # file exists and don't need to be updated
    if isfile(fpath) && !redownload
        return CSV.read(fpath, DataFrame)
    end

    # create containing folder if not exists
    fpath_components = splitpath(fpath)
    if length(fpath_components) > 1
        dir = joinpath(fpath_components[1:end - 1]...)
        if !isdir(dir)
            mkpath(dir)
        end
    end

    # send request for provinces that have confirmed cases
    provinces = Dict([d["value"] => d["label"] for d in request_json(HTTP.URI(scheme="https", host=HOST, path=PATHS[3]))])
    province_ids = sort(collect(keys(provinces)))

    get_cases_and_deaths = @task asyncmap(
        id -> parse_cases_and_deaths(get_province_data(PATHS[1], id, startdate, enddate)),
        province_ids
    )
    get_cases_compositions = @task asyncmap(
        id -> parse_cases_composition(get_province_data(PATHS[2], id, startdate, enddate)),
        province_ids
    )

    schedule(get_cases_and_deaths)
    schedule(get_cases_compositions)

    dfs_cases_and_deaths = fetch(get_cases_and_deaths)
    dfs_cases_composition = fetch(get_cases_compositions)
    df_dates = DataFrame(date=startdate:Day(1):enddate)

    # combines dataframes of with different information
    dfs = Vector{DataFrame}()
    for (id, df1, df2) in zip(province_ids, dfs_cases_and_deaths, dfs_cases_composition)
        df = outerjoin(df_dates, df1, df2, on=:date)
        df.province = fill(provinces[id], nrow(df))
        push!(dfs, df)
    end

    # save dataframe as CSV
    df_final = sort(vcat(dfs...), [:province, :date])
    CSV.write(fpath, df_final)
    df_final
end

end # module VnCdc