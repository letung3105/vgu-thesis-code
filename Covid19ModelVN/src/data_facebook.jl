using DataDeps, Dates, DataFrames, DelimitedFiles, Statistics, Covid19ModelVN

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
                "https://github.com/letung3105/coviddata/raw/master/facebook/movement-range-data-2021-10-09.zip"
                "https://github.com/letung3105/coviddata/raw/master/facebook/gadm1_nuts2-gadm1_nuts2-fb-social-connectedness-index-october-2021.zip"
                "https://github.com/letung3105/coviddata/raw/master/facebook/us-counties-us-counties-fb-social-connectedness-index-october-2021.zip"
            ],
            post_fetch_method = unpack,
        ),
    )
    return nothing
end

"""
Calculate the Social Proximity to Cases [1] index based on the population at each area, the number of confirmed cases on each day,
at each area, and the social connectedness between the areas that are considered.

1. T. Kuchler, D. Russel, and J. Stroebel, “The Geographic Spread of COVID-19 Correlates with the Structure of Social Networks as Measured by Facebook,” National Bureau of Economic Research, Working Paper 26990, Apr. 2020. doi: 10.3386/w26990.

# Arguments

+ `df_population`: A table contains the following columns
    + `ID` is the preferred unique ID at level 1
    + `NAME_1` is the level one GADM region's official name in latin script
    + `VARNAME_1` for the level one GAME variant names, separated by pipes `|`
    + `AVGPOPULATION` for the level one GAME name of the region
+ `df_covid_timeseries_confirmed`: A table contains a column for each province in a country, and each row
contains a timestamp `date` and the number of confirmed cases recorded at `date` for each province.
+ `df_social_connectedness`: A table contains the following columns
    + `user_loc`: the id (ex: "VNM1") of the first province
    + `fr_loc`: the id of the second province
    + `scaled_sci`: the scaled Social Connedtedness Index between two provinces
"""
function calculate_social_proximity_to_cases(
    df_population::AbstractDataFrame,
    df_covid_timeseries_confirmed::AbstractDataFrame,
    df_social_connectedness::AbstractDataFrame,
)
    # gadm code
    getloc(id::AbstractString) =
        subset(df_population, :ID_1 => x -> x .== parse(Int, id[4:end]), view = true)
    # fips code
    getloc(id::Float64) = subset(df_population, :ID_1 => x -> x .== id, view = true)

    locs_with_confirmed = Set(names(df_covid_timeseries_confirmed))

    df_spc = DataFrame()
    df_spc.date = df_covid_timeseries_confirmed.date

    # go through each dataframe that is grouped by the first location
    for (key, df_group) ∈ pairs(groupby(df_social_connectedness, :user_loc))
        # check if population data for the first location is available, skip if not
        first_loc = getloc(key.user_loc)
        first_loc = isempty(first_loc) ? continue : first(first_loc)

        sum_sci = sum(df_group.scaled_sci)
        df_spc[!, first_loc.NAME_1] .= 0
        # go through each location that is connected with the first location
        for row ∈ eachrow(df_group)
            # check if population data for the second location is available, skip if not
            second_loc = getloc(row.fr_loc)
            second_loc = isempty(second_loc) ? continue : first(second_loc)
            # only calculate SPC for location that has confirmed cases
            if second_loc.NAME_1 ∈ locs_with_confirmed
                df_spc[!, first_loc.NAME_1] .+=
                    (
                        df_covid_timeseries_confirmed[!, second_loc.NAME_1] ./
                        second_loc.AVGPOPULATION .* 10000
                    ) .* row.scaled_sci ./ sum_sci
            end
        end
    end

    return df_spc
end

"""
Read the Movement Range Maps dataset from Facebook

# Arguments
+ `fpath`: path the tab-delimited file contains the movement range data
"""
function read_movement_range(fpath::AbstractString)
    data, header = readdlm(fpath, '\t', header = true)
    df = identity.(DataFrame(data, vec(header)))
    df[!, :ds] .= Date.(df[!, :ds])
    return df
end

"""
Calculate the average movement range of a region by taking the mean of the movement
ranges of all of its subregions.

# Arguments
+ `df_movement_range`: A table contains the movement range data with the same structure
as given by Facebook
+ `country_code`: ISO 3166-1 alpha-3 code of the country
+ `subdivision_id`: the "FIPS" code for US regions or "GADM" code for other countries
"""
function region_average_movement_range(
    df_movement_range::AbstractDataFrame,
    country_code::AbstractString,
    subdivision_id::Union{Nothing,<:Integer} = nothing,
)
    # gadm code
    parse_subdivision(x::AbstractString) = parse(Int, split(x, ".")[2])
    # fips code
    parse_subdivision(x::Int) = x

    df_movement_range_region =
        subset(df_movement_range, :country => x -> x .== country_code, view = true)
    if !isnothing(subdivision_id)
        df_movement_range_region = subset(
            df_movement_range_region,
            [:polygon_id, :polygon_source] =>
                (x, y) -> subdivision_id .== parse_subdivision.(x),
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

"""
Save a subset of the movement range maps dataset for a specific region to CSV files.

# Arguments

+ `fpath_outputs`: Paths to the output files
+ `country_codes`: List of ISO-3166 codes of the countries whose average movement range while be calculated
+ `subdivision_ids`: List of GADM or FIPS codes of the subdivion of the countries in `country_codes`
+ `fpath_movement_range`: Path to the movement range dataset
+ `recreate`: Whether to ovewrite an existing file
"""
function save_region_average_movement_range(
    fpath_outputs::AbstractVector{<:AbstractString},
    country_codes::AbstractVector{<:AbstractString},
    subdivision_ids::AbstractVector{<:Union{Nothing,<:Integer}};
    fpath_movement_range::AbstractString = datadep"facebook/movement-range-2021-10-09.txt",
    recreate::Bool = false,
)
    if all(isfile, fpath_outputs) && !recreate
        return nothing
    end

    df_movement_range = FacebookData.read_movement_range(fpath_movement_range)

    for (fpath, country_code, subdivision_id) ∈
        zip(fpath_outputs, country_codes, subdivision_ids)

        if isfile(fpath) && !recreate
            continue
        end

        @info "Generating '$fpath"
        df_region_movement_range = FacebookData.region_average_movement_range(
            df_movement_range,
            country_code,
            subdivision_id,
        )
        save_dataframe(df_region_movement_range, fpath)
    end

    return nothing
end

"""
Read the Social Connectedness Index dataset from Facebook

# Arguments
+ `fpath`: path the tab-delimited file contains the social connectedness data
"""
function read_social_connectedness(fpath::AbstractString)
    data, header = readdlm(fpath, '\t', header = true)
    df = identity.(DataFrame(data, vec(header)))
    return df
end

"""
Get the social connectedness between regions within a country

# Argument

+ `df_social_connectedness`: A table with the same structure as the social connectedness index
dataset from by Facebook
+ `country_code`: The ISO-3166 country code
"""
inter_province_social_connectedness(
    df_social_connectedness::AbstractDataFrame,
    country_code::AbstractString,
) = subset(
    df_social_connectedness,
    [:user_loc, :fr_loc] =>
        ((x, y) -> startswith.(x, country_code) .& startswith.(y, country_code)),
    view = true,
)

"""
Save a subset of the social connectedness index dataset for specific countries to CSV files.

# Arguments

+ `fpath_outputs`: Paths to the output files
+ `country_codes`: List of ISO-3166 codes of the countries whose average movement range while be calculated
+ `fpath_social_connectedness`: Path to the social connectness dataset
+ `recreate`: Whether to ovewrite an existing file
"""
function save_inter_province_social_connectedness(
    fpath_outputs::AbstractVector{<:AbstractString},
    country_codes::AbstractVector{<:AbstractString};
    fpath_social_connectedness::AbstractString = datadep"facebook/gadm1_nuts2_gadm1_nuts2.tsv",
    recreate::Bool = false,
)
    if all(isfile, fpath_outputs) && !recreate
        return nothing
    end

    df_social_connectedness = read_social_connectedness(fpath_social_connectedness)

    for (fpath, country_code) ∈ zip(fpath_outputs, country_codes)
        if isfile(fpath) && !recreate
            continue
        end

        @info "Generating '$fpath"
        df_country_social_connectedness =
            inter_province_social_connectedness(df_social_connectedness, country_code)
        save_dataframe(df_country_social_connectedness, fpath)
    end

    return nothing
end
