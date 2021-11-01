using CSV, DataDeps, DataFrames, Covid19ModelVN
import GeoDataFrames

function __init__()
    register(
        DataDep(
            "gadm2.8",
            """
            Dataset: General Administrative Division Map
            Website: https://gadm.org
            """,
            ["https://biogeo.ucdavis.edu/data/gadm2.8/gpkg/VNM_adm_gpkg.zip"],
            post_fetch_method = unpack,
        ),
    )
    register(
        DataDep(
            "gso",
            """
            Dataset: Vietnam General Statistic Office Average Population 2020
            Website: https://gso.gov.vn
            """,
            [
                "https://github.com/letung3105/coviddata/raw/master/gso/vietnam-2020-average-population-by-province.csv",
            ],
        ),
    )
    return nothing
end

"""
Combine the data for Vietnam GADM level 1 and the population data for Vietnam. The resulting dataframe
contain a row for each province and each row contains a list of IDs, Names, and Average Population of
that province.

# Arguments

+ `df_gadm`: the dataframe of the GADM data
+ `df_population`: the dataframe of the population data
"""
function combine_vietnam_province_level_gadm_and_gso_population(
    df_gadm::AbstractDataFrame,
    df_population::AbstractDataFrame,
)
    # remove rows of repeated provinces
    df_gadm = unique(df_gadm, :ID_1)
    # rename columns and transform data
    df_population = transform(
        df_population,
        # remove double spaces in names
        "Cities, provincies" => (x -> join.(split.(x), " ")) => :VARNAME_1,
        # times 1000 because the original dataset unit is in "thousands people"
        "2020 (*) Average population (Thous. pers.)" =>
            (x -> 1000 .* x) => :AVGPOPULATION,
        copycols = false,
    )

    # remove all spaces in names and make them lowercase
    standardize(s::AbstractString) = lowercase(filter(c -> !isspace(c), s))
    # x is the same as y if the standardized version of y contains the
    # standardized version of x
    issame(x::AbstractString, y::AbstractString) = contains(standardize(y), standardize(x))

    # rename the province so it matches data from GADM
    replace!(df_population.VARNAME_1, "Dak Nong" => "Dac Nong")
    # use provinces/cities names from GADM
    replace!(
        x -> begin
            name = filter(y -> issame(x, y), df_gadm.VARNAME_1)
            return isempty(name) ? x : first(name)
        end,
        df_population.VARNAME_1,
    )

    # get final table
    df = innerjoin(df_gadm, df_population, on = :VARNAME_1)
    select!(df, [:ID_1, :NAME_1, :VARNAME_1, :AVGPOPULATION])
    return df
end

"""
Read and combine the data for Vietnam GADM level 1 and the population data for Vietnam.

# Arguments
+ `fpath_output`: the output CSV file path
+ `fpath_gadm`: path to the GADM geopackage file
+ `fpath_population`: path to the population CSV
+ `recreate`: the existing file will be ovewritten if this is true
"""
function save_vietnam_province_level_gadm_and_gso_population(
    fpath_output::AbstractString;
    fpath_gadm::AbstractString = datadep"gadm2.8/VNM_adm.gpkg",
    fpath_population::AbstractString = datadep"gso/vietnam-2020-average-population-by-province.csv",
    recreate::Bool = false,
)
    if isfile(fpath_output) && !recreate
        return nothing
    end


    @info "Reading '$fpath_gadm' and '$fpath_population'"
    df_gadm = GeoDataFrames.read(fpath_gadm, 1)
    df_population = CSV.read(fpath_population, DataFrame)

    @info "Generating '$fpath_output'"
    df_combined =
        combine_vietnam_province_level_gadm_and_gso_population(df_gadm, df_population)
    save_dataframe(df_combined, fpath_output)

    return nothing
end
