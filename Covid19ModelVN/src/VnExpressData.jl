module VnExpressData

using Dates, DataDeps, DataFrames, CSV, HTTP

function __init__()
    register(
        DataDep(
            "vnexpress",
            """
            Dataset: VnExpress Covid-19 Dashboard API Data
            Website: https://vnexpress.net
            """,
            [
                "https://github.com/letung3105/coviddata/raw/master/vnexpress/timeseries-vietnam-combined.csv",
                "https://github.com/letung3105/coviddata/raw/master/vnexpress/timeseries-vietnam-provinces-confirmed.csv",
                "https://github.com/letung3105/coviddata/raw/master/vnexpress/timeseries-vietnam-provinces-confirmed-total.csv",
            ],
        ),
    )
    return nothing
end

"""
Request and parse Vietnam country-level Covid-19 timeseries from vnexpress into a dataframe

# Arguments

+ `url`: the API url
+ `last_date`: the latest date that can exist in the dataframe
"""
function get_timeseries_vietnam_combined(
    url::AbstractString = "https://vnexpress.net/microservice/sheet/type/covid19_2021_by_day";
    last_date::Date = today() - Day(1),
)
    # request data
    res = HTTP.get(url)
    df = CSV.read(res.body, DataFrame)
    select!(
        df,
        "day_full" => (x -> Date.(x, dateformat"Y/m/d")) => :date,
        "total_active" => :infective,
        "total_cases" => :confirmed_total,
        "total_deaths" => :deaths_total,
        "total_recovered_12" => :recovered_total,
    )
    # Filter date range
    subset!(df, :date => d -> d .<= last_date)

    @assert !hasmissing(df)
    df[!, Not(:date)] .= Int.(df[!, Not(:date)])
    return df
end

"""
Request and parse Vietnam province-level Covid-19 confirmed cases timeseries from vnexpress into a dataframe

# Arguments

+ `url`: the API url
+ `last_date`: the latest date that can exist in the dataframe
"""
function get_timeseries_vietnam_provinces_confirmed(
    url::AbstractString = "https://vnexpress.net/microservice/sheet/type/covid19_2021_by_location";
    last_date::Date = today() - Day(1),
)
    # request data
    res = HTTP.get(url)
    df = CSV.read(res.body, DataFrame)

    clean_provinces_confirmed_cases_timeseries!(df)
    # Filter date range
    subset!(df, :date => d -> d .<= last_date)
    # Replace missing with 0
    df = coalesce.(df, 0)

    @assert !hasmissing(df)
    df[!, Not(:date)] .= Int.(df[!, Not(:date)])
    return df
end

"""
Request and parse Vietnam province-level Covid-19 total confirmed cases timeseries from vnexpress into a dataframe

# Arguments

+ `url`: the API url
+ `last_date`: the latest date that can exist in the dataframe
"""
function get_timeseries_vietnam_provinces_confirmed_total(
    url::AbstractString = "https://vnexpress.net/microservice/sheet/type/covid19_2021_by_total";
    last_date::Date = today() - Day(1),
)
    # request data
    res = HTTP.get(url)
    df = CSV.read(res.body, DataFrame)

    clean_provinces_confirmed_cases_timeseries!(df)
    # Filter date range
    subset!(df, :date => d -> d .<= last_date)

    @assert !hasmissing(df)
    df[!, Not(:date)] .= Int.(df[!, Not(:date)])
    return df
end

"""
Check if there's any missing data cell

# Arguments

+ `df`: the dataframe to be checked
"""
hasmissing(df::AbstractDataFrame) =
    any(Iterators.flatten(map(row -> ismissing.(values(row)), eachrow(df))))

"""
Rename the provinces so that they match the data from GADM

# Arguments

+ `df`: the dataframe to be renamed
"""
function rename_vnexpress_cities_provinces_names_to_gadm!(df::AbstractDataFrame)
    rename!(
        df,
        "TP HCM" => "Hồ Chí Minh city",
        "Thừa Thiên Huế" => "Thừa Thiên - Huế",
        "Đăk Lăk" => "Đắk Lắk",
    )
    return df
end

"""
Clean the province-level data from vnexpress

# Arguments

+ `df`: the dataframe to be cleaned
"""
function clean_provinces_confirmed_cases_timeseries!(df::AbstractDataFrame)
    # removes extra rows and columns
    delete!(df, 1)
    select!(df, 1:63)
    rename_vnexpress_cities_provinces_names_to_gadm!(df)
    # Convert string to date and only select needed columns
    select!(
        df,
        "Ngày" => (x -> Date.(x .* "/2021", dateformat"d/m/Y")) => :date,
        Not("Ngày"),
    )
    return df
end

end
