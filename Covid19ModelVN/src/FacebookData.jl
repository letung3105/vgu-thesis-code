module FacebookData

using CSV, DataDeps, Dates, DataFrames, DelimitedFiles, Statistics

function __init__()
    register(
        DataDep(
            "facebook",
            """
            Dataset: Facebook Data for Good
            Website:
            + https://dataforgood.facebook.com/dfg/tools/movement-range-maps
            + https://dataforgood.facebook.com/dfg/tools/social-connectedness-index
            """,
            [
                "https://github.com/letung3105/vgu-thesis-datasets/raw/master/facebook/movement-range-data-2021-10-09.zip"
                "https://github.com/letung3105/vgu-thesis-datasets/raw/master/facebook/gadm1_nuts2-gadm1_nuts2-fb-social-connectedness-index-october-2021.zip"
            ],
            post_fetch_method = unpack,
        ),
    )
    return nothing
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

function read_movement_range(fpath)
    data, header = readdlm(fpath, '\t', header = true)
    df = identity.(DataFrame(data, vec(header)))
    df[!, :ds] .= Date.(df[!, :ds])
    return df
end

function region_average_movement_range(
    df_movement_range,
    country_code,
    subdivision_id = nothing,
)
    df_movement_range_region =
        subset(df_movement_range, :country => x -> x .== country_code, view = true)
    if !isnothing(subdivision_id)
        df_movement_range_region = subset(
            df_movement_range_region,
            :polygon_id => x -> startswith.(x, "$country_code.$subdivision_id"),
            view = true,
        )
    end

    df_movement_range_region_avg = combine(
        DataFrames.groupby(df_movement_range_region, :ds),
        [:all_day_bing_tiles_visited_relative_change, :all_day_ratio_single_tile_users] .=> mean,
        renamecols = false,
    )
    return df_movement_range_region_avg
end

function read_social_connectedness(fpath)
    data, header = readdlm(fpath, '\t', header = true)
    df = identity.(DataFrame(data, vec(header)))
    return df
end

inter_province_social_connectedness(df_social_connectedness, country_code) = subset(
    df_social_connectedness,
    [:user_loc, :fr_loc] => ((x, y) -> startswith.(x, country_code) .& startswith.(y, country_code)),
    view = true,
)

end # module FacebookData
