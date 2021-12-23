using Covid19ModelVN, ArgParse

function make_datasets(dir::AbstractString=""; recreate::Bool=false)
    make_population(dir; recreate)
    make_covid_timeseries(dir; recreate)
    make_movement_range(dir; recreate)
    return make_social_proximity(dir; recreate)
end

function mkdatasets(args=ARGS)
    s = ArgParseSettings()
    @add_arg_table s begin
        "--dir"
        help = "location to the directory where the datasets are saved"
        arg_type = String
        default = ".cache"

        "--recreate"
        help = "overwrite existing datasets if there is any"
        action = :store_true
    end
    parsed_args = parse_args(args, s; as_symbols=true)
    return make_datasets(parsed_args[:dir]; recreate=parsed_args[:recreate])
end

mkdatasets()
