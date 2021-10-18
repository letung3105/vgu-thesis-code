module VnExpressData

using Dates, DataFrames, CSV, HTTP

"""
Clean the data from VnExpress

# Arguments

* `df::DataFrame`: the `DataFrame` that contains the data
* `firstdate::Date`: date of the first entry in the returned data
* `lastdate::Date`: date of the last entry in the returned data
"""
function clean_cases_timeseries!(
    df::DataFrame,
    first_date::Date,
    last_date::Date
)
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
    transform!(df,
        [:confirmed_total, :dead_total, :recovered_total]
            => ((x, y, z) -> x - y - z)
            => :infective
    )
    return df
end

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
    url="https://vnexpress.net/microservice/sheet/type/covid19_2021_by_day",
    date_format=dateformat"yyyymmdd",
    recreate=false
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

    # request data
    res = HTTP.get(url)
    df = CSV.read(res.body, DataFrame)
    clean_cases_timeseries!(df, first_date, last_date)
    # save csv
    CSV.write(fpath, df)
    return df
end

end