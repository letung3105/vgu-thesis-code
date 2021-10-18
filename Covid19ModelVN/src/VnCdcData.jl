module VnCdcData

using CSV
using Dates
using DataFrames
using JSON
using HTTP

function request_json(data...; kwargs...)
    res = HTTP.get(data..., kwargs...)
    JSON.parse(String(res.body))
end

"""
Send a request for the province's report for the specify time period

```
* `province_id::AbstractString`: the province identifier (one to two digits number)
* `start_date::Date`: early date in the timeseries
* `end_date::Date`: latest date in the timeseries
* `date_format`: format string for date values (default: "yyy-mm-dd")
* `host::AbstractString`: host of VnCDC API
* `path::AbstractString`: path to the data
```
"""
function request_province_cases_data(
    province_id::AbstractString,
    start_date::Date,
    end_date::Date;
    date_format=dateformat"yyyy-mm-dd",
    host::AbstractString,
    path::AbstractString,
)
    query = Dict(
        :start_time => Dates.format(start_date, date_format),
        :end_time => Dates.format(end_date, date_format),
        :province_ids => province_id
    )
    uri = HTTP.URI(scheme="https", host=host, path=path, query=query)
    return request_json(uri)
end

"""
Send a request for the province's identifiers from VnCDC

```
* `host::AbstractString`: host of VnCDC API
* `path::AbstractString`: path to the data
```
"""
function request_provinces_ids(;
    host::AbstractString="ncov.vncdc.gov.vn",
    path::AbstractString="/v2/vietnam/provinces"
)
    # send request for provinces that have confirmed cases
    ids_data = request_json(HTTP.URI(scheme="https", host=host, path=path))
    provinces = Dict([d["value"] => d["label"] for d in ids_data])
    return sort(collect(keys(provinces)))
end

"""
Parse a JSON array of pairs of a date value and a integer value

```json
[
    [<DATE VALUE IN UNIX NANO>, <INTEGER VALUE>],
    [<DATE VALUE IN UNIX NANO>, <INTEGER VALUE>],

    ...

    [<DATE VALUE IN UNIX NANO>, <INTEGER VALUE>]
]
```
"""
function parse_date_value_pairs(data)
    dates = Vector{Date}()
    values = Vector{Int}()
    for data_point in data
        push!(dates, Date(Dates.unix2datetime(data_point[1] // 1000)))
        push!(values, data_point[2])
    end
    dates, values
end

"""
Parse a JSON that contains the timeseries data of the number of confirmed cases and
the number deaths

```json
{
    "report": [
        { # Number of confirmed cases
            "data": [
                [<DATE VALUE IN UNIX NANO>, <INTEGER VALUE>],
                [<DATE VALUE IN UNIX NANO>, <INTEGER VALUE>],

                ...

                [<DATE VALUE IN UNIX NANO>, <INTEGER VALUE>]
            ]

            ...
        },
        { # Number of deaths
            "data": [
                [<DATE VALUE IN UNIX NANO>, <INTEGER VALUE>],
                [<DATE VALUE IN UNIX NANO>, <INTEGER VALUE>],

                ...

                [<DATE VALUE IN UNIX NANO>, <INTEGER VALUE>]
            ]

            ...
        }
    ]
}
```
"""
function parse_deaths_and_confirmed_report(data)
    column_names = ["confirmed", "deaths"]
    data = data["report"]
    dfs = Vector{DataFrame}()
    for (i, name) in enumerate(column_names)
        dates, cases = parse_date_value_pairs(data[i]["data"])
        df = DataFrame(["date" => dates, name => cases])
        push!(dfs, df)
    end
    innerjoin(dfs..., on=:date)
end

"""
Parse a JSON that contains the timeseries data of the number of confirmed cases that are classified
into different categories

```json
{
    "report": [
        { # Number of confirmed cases
            "data": [
                [<DATE VALUE IN UNIX NANO>, <INTEGER VALUE>],
                [<DATE VALUE IN UNIX NANO>, <INTEGER VALUE>],

                ...

                [<DATE VALUE IN UNIX NANO>, <INTEGER VALUE>]
            ]

            ...
        },
        { # Number of deaths
            "data": [
                [<DATE VALUE IN UNIX NANO>, <INTEGER VALUE>],
                [<DATE VALUE IN UNIX NANO>, <INTEGER VALUE>],

                ...

                [<DATE VALUE IN UNIX NANO>, <INTEGER VALUE>]
            ]

            ...
        }
    ]
}
```
"""
function parse_categorized_confirmed_report(data)
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


function save_cases_timeseries(
    fdir::AbstractString,
    fid::AbstractString,
    first_date::Date,
    last_date::Date;
    recreate=false,
    date_format=dateformat"yyyymmdd",
    host="ncov.vncdc.gov.vn",
    deaths_and_confirmed_report_path="/v2/vietnam/report-epi",
    categorized_confirmed_report_path="/v2/vietnam/provinces",

)
    fpath = joinpath(fdir, "$(Dates.format(first_date, date_format))-$(Dates.format(last_date, date_format))-$fid.csv")
    # file exists and don't need to be updated
    if isfile(fpath) && !recreate
        return CSV.read(fpath, DataFrame)
    end
    # create containing folder if not exists
    if !isdir(fdir)
        mkpath(fdir)
    end

    # send request for provinces that have confirmed cases
    province_ids = request_provinces_ids()
    # Get the report for the number of deaths and the number of confirmed cases for every province
    get_cases_and_deaths = @task asyncmap(
        id -> parse_deaths_and_confirmed_report(
            request_province_cases_data(id, first_date, last_date, host=host, path=deaths_and_confirmed_report_path)
        ),
        province_ids
    )
    # Get the report for the categorized number of confirmed cases for every province
    get_cases_compositions = @task asyncmap(
        id -> parse_categorized_confirmed_report(
            request_province_cases_data(id, first_date, last_date, host=host, path=categorized_confirmed_report_path)
        ),
        province_ids
    )
    # Send all HTTP requests at once
    schedule(get_cases_and_deaths)
    schedule(get_cases_compositions)
    # Wait for all HTTP requests to return and parsed
    dfs_cases_and_deaths = fetch(get_cases_and_deaths)
    dfs_cases_composition = fetch(get_cases_compositions)
    df_dates = DataFrame(date=first_date:Day(1):last_date)

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