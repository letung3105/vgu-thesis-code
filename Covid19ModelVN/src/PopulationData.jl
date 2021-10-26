module PopulationData

using CSV, DataDeps, DataFrames, Covid19ModelVN.Helpers
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
                "https://github.com/letung3105/vgu-thesis-datasets/raw/master/gso/vietnam-2020-average-population-by-province.csv",
            ],
        ),
    )
    return nothing
end

function combine_vietnam_province_level_gadm_and_gso_population(df_gadm, df_population)
    # remove rows of repeated provinces
    df_gadm = unique(df_gadm, :GID_1)
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
    standardize(s) = lowercase(filter(c -> !isspace(c), s))
    # x is the same as y if the standardized version of y contains the
    # standardized version of x
    issame(x, y) = contains(standardize(y), standardize(x))

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
    select!(df, [:GID_1, :NAME_1, :VARNAME_1, :AVGPOPULATION])
    return df
end

function save_vietnam_province_level_gadm_and_gso_population(
    fpath_output;
    fpath_gadm = datadep"gadm2.8/VNM_adm.gpkg",
    fpath_population = datadep"gso/vietnam-2020-average-population-by-province.csv",
    recreate = false,
)
    if isfile(fpath_output) && !recreate
        return nothing
    end

    df_gadm = GeoDataFrames.read(fpath_gadm, 1)
    df_population = CSV.read(fpath_population, DataFrame)

    df_combined = combine_vietnam_province_level_gadm_and_gso_population(
        df_gadm,
        df_population,
    )
    save_dataframe(df_combined, fpath_output)
    return nothing
end

end # module PopulationData
