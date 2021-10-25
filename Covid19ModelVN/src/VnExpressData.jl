module VnExpressData

using Dates, DataFrames, CSV, HTTP

hasmissing(df::DataFrame) =
    any(Iterators.flatten(map(row -> ismissing.(values(row)), eachrow(df))))

function timeseries_vietnam(
    url = "https://vnexpress.net/microservice/sheet/type/covid19_2021_by_day";
    last_date = today() - Day(1),
)
    # request data
    res = HTTP.get(url)
    df = CSV.read(res.body, DataFrame)

    select!(
        df,
        "day_full" => (x -> Date.(x, dateformat"Y/m/d")) => :date,
        "new_cases" => :confirmed,
        "total_cases" => :confirmed_total,
        "new_deaths" => :deaths,
        "total_deaths" => :deaths_total,
        "new_recovered" => :recovered,
        "total_recovered_12" => :recovered_total,
    )
    filter!(:date => d -> d <= last_date, df)
    sort!(df, :date)
    transform!(
        df,
        [:confirmed_total, :deaths_total, :recovered_total] =>
            ((x, y, z) -> x - y - z) => :infective,
    )

    @assert !hasmissing(df)
    df[!, Not(:date)] .= Int.(df[!, Not(:date)])
    return df
end

function rename_vnexpress_cities_provinces_names_to_gso!(df::DataFrame)
    rename!(
        df,
        "TP HCM" => "Hồ Chí Minh city",
        "Thừa Thiên Huế" => "Thừa Thiên - Huế",
        "Đăk Lăk" => "Đắk Lắk",
    )
    return df
end

function clean_provinces_confirmed_cases_timeseries!(df::DataFrame)
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

function timeseries_provinces_confirmed(
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

function save_provinces_confirmed_cases_timeseries(
    fdir::AbstractString,
    fid::AbstractString,
    first_date::Date,
    last_date::Date;
    url = "https://vnexpress.net/microservice/sheet/type/covid19_2021_by_location",
    date_format = dateformat"yyyymmdd",
    recreate = false,
)
    fpath = joinpath(
        fdir,
        "$(Dates.format(first_date, date_format))-$(Dates.format(last_date, date_format))-$fid.csv",
    )
    # file exists and don't need to be updated
    if isfile(fpath) && !recreate
        return CSV.read(fpath, DataFrame)
    end
    # create containing folder if not exists
    if !isdir(fdir)
        mkpath(fdir)
    end

    # request data
    res = HTTP.get(url)
    df = CSV.read(res.body, DataFrame)

    clean_provinces_confirmed_cases_timeseries!(df)
    # Filter date range
    filter!(:date => d -> d >= first_date && d <= last_date, df)
    # Replace missing with 0
    df = coalesce.(df, 0)

    @assert !hasmissing(df)
    df[!, Not(:date)] = Int.(df[!, Not(:date)])
    CSV.write(fpath, df)
    return df
end

function timeseries_provinces_confirmed_total(
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

function save_provinces_total_confirmed_cases_timeseries(
    fdir::AbstractString,
    fid::AbstractString,
    first_date::Date,
    last_date::Date;
    url = "https://vnexpress.net/microservice/sheet/type/covid19_2021_by_total",
    date_format = dateformat"yyyymmdd",
    recreate = false,
)
    fpath = joinpath(
        fdir,
        "$(Dates.format(first_date, date_format))-$(Dates.format(last_date, date_format))-$fid.csv",
    )
    # file exists and don't need to be updated
    if isfile(fpath) && !recreate
        return CSV.read(fpath, DataFrame)
    end
    # create containing folder if not exists
    if !isdir(fdir)
        mkpath(fdir)
    end

    # request data
    res = HTTP.get(url)
    df = CSV.read(res.body, DataFrame)

    clean_provinces_confirmed_cases_timeseries!(df)
    # Filter date range
    filter!(:date => d -> d >= first_date && d <= last_date, df)

    @assert !hasmissing(df)
    df[!, Not(:date)] = Int.(df[!, Not(:date)])
    CSV.write(fpath, df)
    return df
end

end
