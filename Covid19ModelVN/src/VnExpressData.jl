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
                "https://github.com/letung3105/vgu-thesis-datasets/raw/master/vnexpress/timeseries-vietnam-provinces-confirmed.csv"
                "https://github.com/letung3105/vgu-thesis-datasets/raw/master/vnexpress/timeseries-vietnam-provinces-confirmed-total.csv"
            ],
        ),
    )
    return nothing
end

function get_timeseries_vietnam_provinces_confirmed(
    url = "https://vnexpress.net/microservice/sheet/type/covid19_2021_by_location";
    last_date = today() - Day(1),
)
    # request data
    res = HTTP.get(url)
    df = CSV.read(res.body, DataFrame)

    clean_provinces_confirmed_cases_timeseries!(df)
    # Filter date range
    filter!(:date => d -> d >= d <= last_date, df)
    # Replace missing with 0
    df = coalesce.(df, 0)

    @assert !hasmissing(df)
    df[!, Not(:date)] .= Int.(df[!, Not(:date)])
    return df
end

function get_timeseries_vietnam_provinces_confirmed_total(
    url = "https://vnexpress.net/microservice/sheet/type/covid19_2021_by_total";
    last_date = today() - Day(1),
)
    # request data
    res = HTTP.get(url)
    df = CSV.read(res.body, DataFrame)

    clean_provinces_confirmed_cases_timeseries!(df)
    # Filter date range
    filter!(:date => d -> d <= last_date, df)

    @assert !hasmissing(df)
    df[!, Not(:date)] .= Int.(df[!, Not(:date)])
    return df
end

hasmissing(df) =
    any(Iterators.flatten(map(row -> ismissing.(values(row)), eachrow(df))))

function rename_vnexpress_cities_provinces_names_to_gso!(df)
    rename!(
        df,
        "TP HCM" => "Hồ Chí Minh city",
        "Thừa Thiên Huế" => "Thừa Thiên - Huế",
        "Đăk Lăk" => "Đắk Lắk",
    )
    return df
end

function clean_provinces_confirmed_cases_timeseries!(df)
    # removes extra rows and columns
    delete!(df, 1)
    select!(df, 1:63)
    rename_vnexpress_cities_provinces_names_to_gso!(df)
    # Convert string to date and only select needed columns
    select!(
        df,
        "Ngày" => (x -> Date.(x .* "/2021", dateformat"d/m/Y")) => :date,
        Not("Ngày"),
    )
    return df
end

end
