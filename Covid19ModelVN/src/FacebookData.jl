module FacebookData

using DataDeps, Dates, DataFrames, DelimitedFiles, Statistics, CSV

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
                "https://github.com/letung3105/coviddata/raw/master/facebook/movement-range-data-2021-11-24.zip"
                "https://github.com/letung3105/coviddata/raw/master/facebook/movement-range-data-2020-03-01-2020-12-31.zip"
                "https://github.com/letung3105/coviddata/raw/master/facebook/gadm1_nuts2-gadm1_nuts2-fb-social-connectedness-index-october-2021.zip"
                "https://github.com/letung3105/coviddata/raw/master/facebook/us-counties-us-counties-fb-social-connectedness-index-october-2021.zip"
            ];
            post_fetch_method=unpack,
        ),
    )
    return nothing
end

"""
    function calculate_social_proximity_to_cases(
        df_population::AbstractDataFrame,
        df_covid_timeseries_confirmed::AbstractDataFrame,
        df_social_connectedness::AbstractDataFrame,
    )

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
    function getloc(id::AbstractString)
        return subset(df_population, :ID_1 => x -> x .== parse(Int, id[4:end]); view=true)
    end
    # fips code
    getloc(id::AbstractFloat) = subset(df_population, :ID_1 => x -> x .== id; view=true)

    # initialize the dataframe with dates and zeros
    df_spc = DataFrame()
    df_spc.date = df_covid_timeseries_confirmed.date
    for locid in unique(df_social_connectedness.user_loc)
        loc = getloc(locid)
        loc = isempty(loc) ? continue : first(loc)
        df_spc[!, loc.NAME_1] .= 0
    end
    # names of all locations that have confirmed cases
    locs_with_confirmed = Set(names(df_covid_timeseries_confirmed))
    # go through each dataframe that is grouped by the first location
    Threads.@threads for (key, df_group) in
                         collect(pairs(groupby(df_social_connectedness, :user_loc)))
        # check if population data for the first location is available, skip if not
        first_loc = getloc(key.user_loc)
        first_loc = isempty(first_loc) ? continue : first(first_loc)
        sum_sci = sum(df_group.scaled_sci)
        # go through each location that is connected with the first location
        for row in eachrow(df_group)
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
    read_movement_range(fpath::AbstractString)

Read the Movement Range Maps dataset from Facebook

# Arguments
+ `fpath`: path the tab-delimited file contains the movement range data
"""
function read_movement_range(fpath::AbstractString)
    data, header = readdlm(fpath, '\t'; header=true)
    df = identity.(DataFrame(data, vec(header)))
    df[!, :ds] .= Date.(df[!, :ds])
    return df
end

"""
    region_average_movement_range(
        df_movement_range::AbstractDataFrame,
        country_code::AbstractString,
        subdivision_id::Union{Nothing,<:Integer} = nothing,
    )

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
    subdivision_id::Union{Nothing,<:Integer}=nothing,
)
    # gadm code
    parse_subdivision(x::AbstractString) = parse(Int, split(x, ".")[2])
    # fips code
    parse_subdivision(x::Int) = x

    df_movement_range_region = subset(
        df_movement_range, :country => x -> x .== country_code; view=true
    )
    if !isnothing(subdivision_id)
        df_movement_range_region = subset(
            df_movement_range_region,
            [:polygon_id, :polygon_source] =>
                (x, y) -> subdivision_id .== parse_subdivision.(x);
            view=true,
        )
    end

    df_movement_range_region_avg = combine(
        DataFrames.groupby(df_movement_range_region, :ds),
        [:all_day_bing_tiles_visited_relative_change, :all_day_ratio_single_tile_users] .=>
            mean;
        renamecols=false,
    )
    return df_movement_range_region_avg
end

"""
    RegionMovementRangeFile

+ `path`: Path to the file
+ `country`: ISO-3166 code of the country whose average movement range is contained in `path`
+ `subdivision`: GADM or FIPS code of the subdivion of `country`
"""
struct RegionMovementRangeFile
    path::AbstractString
    country::AbstractString
    subdivision::Union{<:Integer,Nothing}
end

"""
    save_region_average_movement_range(
        files::AbstractVector{RegionMovementRangeFile};
        fpath_movement_range::AbstractString = datadep"facebook/movement-range-2021-11-24.txt",
        recreate::Bool = false,
    )

Save a subset of the movement range maps dataset for a specific region to CSV files.

# Arguments

+ `files`: List of file configurations
+ `fpath_movement_range`: Path to the movement range dataset
+ `recreate`: Whether to ovewrite an existing file
"""
function save_region_average_movement_range(
    files::AbstractVector{RegionMovementRangeFile};
    fpath_movement_range::AbstractString=datadep"facebook/movement-range-2021-11-24.txt",
    recreate::Bool=false,
)
    if all(f -> isfile(f.path), files) && !recreate
        return nothing
    end

    @info("Reading Movement Range Maps dataset", fpath_movement_range)
    df_movement_range = FacebookData.read_movement_range(fpath_movement_range)

    Threads.@threads for f in files
        if isfile(f.path) && !recreate
            continue
        end
        if !isdir(dirname(f.path))
            mkpath(dirname(f.path))
        end

        @info(
            "Generating average stay put index and average change in movement",
            f.path,
            f.country,
            f.subdivision
        )
        df_region_movement_range = FacebookData.region_average_movement_range(
            df_movement_range, f.country, f.subdivision
        )
        CSV.write(f.path, df_region_movement_range)
    end

    return nothing
end

"""
    read_social_connectedness(fpath::AbstractString)

Read the Social Connectedness Index dataset from Facebook

# Arguments
+ `fpath`: path the tab-delimited file contains the social connectedness data
"""
function read_social_connectedness(fpath::AbstractString)
    data, header = readdlm(fpath, '\t'; header=true)
    df = identity.(DataFrame(data, vec(header)))
    return df
end

"""
    inter_province_social_connectedness(
        df_social_connectedness::AbstractDataFrame,
        country_code::AbstractString,
    )

Get the social connectedness between regions within a country

# Argument

+ `df_social_connectedness`: A table with the same structure as the social connectedness index
dataset from by Facebook
+ `country_code`: The ISO-3166 country code
"""
function inter_province_social_connectedness(
    df_social_connectedness::AbstractDataFrame, country_code::AbstractString
)
    return subset(
        df_social_connectedness,
        [:user_loc, :fr_loc] =>
            ((x, y) -> startswith.(x, country_code) .& startswith.(y, country_code));
        view=true,
    )
end

"""
    InterProvinceSocialConnectednessFile

+ `path`: Path to the file
+ `country`: ISO-3166 code of the country whose average social connectedness is contained in `path`
"""
struct InterProvinceSocialConnectednessFile
    path::AbstractString
    country::AbstractString
end

"""
    save_inter_province_social_connectedness(
        files::AbstractVector{InterProvinceSocialConnectednessFile};
        fpath_social_connectedness::AbstractString = datadep"facebook/gadm1_nuts2_gadm1_nuts2.tsv",
        recreate::Bool = false,
    )

Save a subset of the social connectedness index dataset for specific countries to CSV files.

# Arguments

+ `files`: List of file configurations
+ `fpath_social_connectedness`: Path to the social connectness dataset
+ `recreate`: Whether to ovewrite an existing file
"""
function save_inter_province_social_connectedness(
    files::AbstractVector{InterProvinceSocialConnectednessFile};
    fpath_social_connectedness::AbstractString=datadep"facebook/gadm1_nuts2_gadm1_nuts2.tsv",
    recreate::Bool=false,
)
    if all(f -> isfile(f.path), files) && !recreate
        return nothing
    end

    @info("Reading Social Connectedness Index dataset", fpath_social_connectedness)
    df_social_connectedness = read_social_connectedness(fpath_social_connectedness)

    Threads.@threads for f in files
        if isfile(f.path) && !recreate
            continue
        end
        if !isdir(dirname(f.path))
            mkpath(dirname(f.path))
        end

        @info("Generating average social connectedness index", f.path, f.country,)
        df_country_social_connectedness = inter_province_social_connectedness(
            df_social_connectedness, f.country
        )
        CSV.write(f.path, df_country_social_connectedness)
    end

    return nothing
end

end
