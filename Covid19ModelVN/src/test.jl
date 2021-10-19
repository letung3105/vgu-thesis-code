# Activate the environment for running the script
if isfile("Project.toml") && isfile("Manifest.toml")
    import Pkg
    Pkg.activate(".")
end

using Dates, DataFrames, DelimitedFiles

function save_country_inter_province_connectedness_indices(
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

let
    source_fpath = "datasets/facebook/social-connectedness-index/gadm1_nuts2_gadm1_nuts2_aug2020.tsv"
    country_code = "VNM"

    data, header = readdlm(source_fpath, '\t', header = true)
    df = identity.(DataFrame(data, vec(header)))
    filter!(
        x -> startswith(x.user_loc, country_code) && endswith(x.fr_loc, country_code),
        df,
    )
end
