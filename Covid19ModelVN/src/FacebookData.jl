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
    country::AbstractString,
    gadm1_id::Union{Nothing,Int} = nothing;
    recreate::Bool = false,
)
    fpath = if isnothing(gadm1_id)
        joinpath(fdir, "$country-$fid.csv")
    else
        joinpath(fdir, "$country$gadm1_id-$fid.csv")
    end
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
    if !isnothing(gadm1_id)
        filter!(x -> startswith(x.polygon_id, "$country.$gadm1_id"), df)
    end
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

function save_intra_country_gadm1_nuts2_connectedness_index(
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
    # only get friendship between cities/provinces in the country
    filter!(x -> startswith(x.user_loc, country) && startswith(x.fr_loc, country), df)

    # save csv
    CSV.write(fpath, df)
    return df
end

# https://arxiv.org/pdf/2109.12094.pdf
function calculate_social_proximity_to_cases_index(
    df_gadm1_population::DataFrame,
    df_gadm1_total_confirmed_cases::DataFrame,
    df_gadm1_intra_connected_index::DataFrame,
)
    df_incidence_rate = DataFrame()
    df_incidence_rate.date = df_gadm1_total_confirmed_cases.date
    # Go through each province, divide the time series data with that province
    # total population, and multiply to result with 10000 to get the incidence rate
    for name in df_gadm1_population.gadm1_name
        population =
            first(filter(:gadm1_name => x -> x == name, df_gadm1_population).avg_population)
        if name in names(df_gadm1_total_confirmed_cases)
            df_incidence_rate[!, name] =
                df_gadm1_total_confirmed_cases[!, name] ./ population .* 10000
        else
            df_incidence_rate[!, name] .= 0
        end
    end

    df_spc_index = DataFrame()
    df_spc_index.date = df_incidence_rate.date
    # Go through the connectedness index of each region and calculate to weighted sum
    # of the incidence rate. The weights are determined by the connectedness index between
    # province A and province B divided by the sum of all the connectedness indinces of province A
    for (key, df_region_sci) in pairs(groupby(df_gadm1_intra_connected_index, :user_loc))
        region_sci_sum = sum(df_region_sci.scaled_sci)
        region_name = first(
            filter(
                :gadm1_id => x -> x == parse(Int, key.user_loc[4:end]),
                df_gadm1_population,
            ).gadm1_name,
        )

        df_spc_index[!, region_name] .= 0
        for connected_region in eachrow(df_region_sci)
            connected_region_name = first(
                filter(
                    :gadm1_id => x -> x == parse(Int, connected_region.fr_loc[4:end]),
                    df_gadm1_population,
                ).gadm1_name,
            )
            df_spc_index[!, region_name] .+=
                df_incidence_rate[!, connected_region_name] .*
                connected_region.scaled_sci ./ region_sci_sum
        end
    end

    return df_spc_index
end

function save_social_proximity_to_cases_index(
    gadm1_population_fpath::AbstractString,
    gadm1_total_confirmed_cases_fpath::AbstractString,
    gadm1_intra_connected_index_fpath::AbstractString,
    fdir::AbstractString,
    fid::AbstractString;
    recreate::Bool = false,
)
    fpath = joinpath(fdir, "$fid.csv")
    # file exists and don't need to be updated
    if isfile(fpath) && !recreate
        return CSV.read(fpath, DataFrame)
    end
    # create containing folder if not exists
    if !isdir(fdir)
        mkpath(fdir)
    end

    df_gadm1_population = CSV.read(gadm1_population_fpath, DataFrame)
    df_gadm1_total_confirmed_cases = CSV.read(gadm1_total_confirmed_cases_fpath, DataFrame)
    df_gadm1_intra_connected_index = CSV.read(gadm1_intra_connected_index_fpath, DataFrame)
    df_social_proximity_to_cases = calculate_social_proximity_to_cases_index(
        df_gadm1_population,
        df_gadm1_total_confirmed_cases,
        df_gadm1_intra_connected_index,
    )

    CSV.write(fpath, df_social_proximity_to_cases)
    return df_social_proximity_to_cases
end

end # module FacebookData
