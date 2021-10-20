module VnExpressData

using Dates, DataFrames, CSV, HTTP

hasmissing(df::DataFrame) =
    any(Iterators.flatten(map(row -> ismissing.(values(row)), eachrow(df))))

"""
Request, clean, and save the cases timeseries from VnExpress

* `fdir::AbstractString`: directory to save the file
* `fid::AbstractString`: the file identifier
* `first_date::Date`: earliest date to consider
* `last_date::Date`: latest date to consider
* `url"`: API path of VnExpress
* `date_format`: string format for dates
* `recreate`: true if we want to save a new file when one already exists
"""
function save_cases_timeseries(
    fdir::AbstractString,
    fid::AbstractString,
    first_date::Date,
    last_date::Date;
    url = "https://vnexpress.net/microservice/sheet/type/covid19_2021_by_day",
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

    select!(
        df,
        "day_full" => (x -> Date.(x, dateformat"Y/m/d")) => :date,
        "new_cases" => :confirmed,
        "total_cases" => :confirmed_total,
        "new_deaths" => :dead,
        "total_deaths" => :dead_total,
        "new_recovered" => :recovered,
        "total_recovered_12" => :recovered_total,
    )
    filter!(:date => d -> d >= first_date && d <= last_date, df)
    sort!(df, :date)
    transform!(
        df,
        [:confirmed_total, :dead_total, :recovered_total] =>
            ((x, y, z) -> x - y - z) => :infective,
    )

    @assert !hasmissing(df)
    df[!, Not(:date)] = Int.(df[!, Not(:date)])
    CSV.write(fpath, df)
    return df
end

function rename_vnexpress_cities_provinces_names_to_gso!(df::DataFrame)
    rename!(
        df,
        "TP HCM" => "Hồ Chí Minh city",
        "Thừa Thiên Huế" => "Thừa Thiên - Huế",
        "Đăk Lăk" => "Đắk Lắk",
    )
    return nothing
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
        Not("Ngày")
    )
    return nothing
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