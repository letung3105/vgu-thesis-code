module PopulationData

using DataFrames
import GeoDataFrames, CSV

function save_vietnam_gadm1_population(
    vn_adm_gpkg_fpath::AbstractString,
    gso_population_en_fpath::AbstractString,
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

    # read the Geopackage (GADM v2.8)
    df_vn_adm = unique(GeoDataFrames.read(vn_adm_gpkg_fpath, 1))
    select!(
        df_vn_adm,
        :NAME_1 => :gadm1_name,
        :VARNAME_1 => :gadm1_varname,
        :ID_1 => :gadm1_id,
    )

    # GSO's expected population for 2020 (EN version)
    df_vn_population = CSV.read(gso_population_en_fpath, DataFrame)
    # remove double spaces in names
    select!(
        df_vn_population,
        "Cities, provincies" => :gadm1_varname,
        "2020 (*) Average population (Thous. pers.)" => :avg_population_thousands,
    )
    # rename the province so it matches data from GADM
    replace!(df_vn_population.gadm1_varname, "Dak Nong" => "Dac Nong")
    transform!(
        df_vn_population,
        :gadm1_varname => x -> join.(split.(x), " "),
        :avg_population_thousands => (x -> 1000 .* x) => :avg_population,
        renamecols = false,
    )

    # remove all spaces in names and make them lowercase
    standardize(s) = lowercase(filter(c -> !isspace(c), s))
    # x is the same as y if the standardized version of y contains the
    # standardized version of x
    issame(x, y) = contains(standardize(y), standardize(x))
    # check if the name is a province or a city defined by GADM
    is_in_gadm(s, names) = any(map(name -> issame(s, name), names))

    # remove extra data from GSO
    filter!(:gadm1_varname => x -> is_in_gadm(x, df_vn_adm.gadm1_varname), df_vn_population)
    # use provinces/cities names from GADM
    replace!(
        varname -> first(filter(x -> issame(varname, x), df_vn_adm.gadm1_varname)),
        df_vn_population.gadm1_varname,
    )

    # get final table
    df = innerjoin(df_vn_adm, df_vn_population, on = :gadm1_varname)
    CSV.write(fpath, df)
    return df
end

end # module PopulationData
