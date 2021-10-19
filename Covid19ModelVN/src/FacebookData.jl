module FacebookData

using CSV, Dates, DataFrames, DelimitedFiles, Statistics

"""
Create a average country movement range by taking the mean of the data for
all the regions within the country, then save the data to a CSV file.

# Arguments

* `source_fpath::AbstractString`: path to the file for Facebook global movement range data
* `fdir::AbstractString`: directory where the new CSV will be saved
* `fid::AbstractString`: identifier of the new CSV file
* `country::AbstractString`: country code
* `recreate::Bool`: true if not recreating the file when it already exists
"""
function save_country_average_movement_range(
    source_fpath::AbstractString,
    fdir::AbstractString,
    fid::AbstractString,
    country::AbstractString;
    recreate::Bool = false,
)
    fpath = joinpath(fdir, "$country-$fid.csv")
    # file exists and don't need to be updated
    if isfile(fpath) && !recreate
        return CSV.read(fpath, DataFrame)
    end
    # create containing folder if not exists
    if !isdir(fdir)
        mkpath(fdir)
    end

    data, header = readdlm(source_fpath, '\t', header = true)
    df = identity.(DataFrame(data, vec(header)))
    filter!(x -> x.country == "VNM", df)
    transform!(df, :ds => x -> Date.(x), renamecols = false)

    df_final = combine(
        DataFrames.groupby(df, :ds),
        :all_day_bing_tiles_visited_relative_change => mean,
        :all_day_ratio_single_tile_users => mean,
        renamecols = false,
    )
    # save csv
    CSV.write(fpath, df_final)
    return df_final
end

"""
Get the moving average of the given movement range data

# Arguments

* `df::DataFrame`: the `DataFrame` that contains the data
* `n::Int`: number of samples for the moving average
"""
function get_movement_range_moving_average(df::DataFrame, n::Int = 7)
    moving_average(xs) = [mean(@view xs[i-n+1:i]) for i = n:length(xs)]
    return combine(
        df,
        :ds => x -> x[n:end],
        :all_day_bing_tiles_visited_relative_change => moving_average,
        :all_day_ratio_single_tile_users => moving_average,
        renamecols = false,
    )
end

end # module FacebookData
