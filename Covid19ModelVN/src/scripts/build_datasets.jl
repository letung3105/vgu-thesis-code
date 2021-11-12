# Activate the environment for running the script
if isfile("Project.toml") && isfile("Manifest.toml")
    import Pkg
    Pkg.activate(".")
end

using Covid19ModelVN

function build_datasets_main(dir::AbstractString = "")
    build_population(dir)
    build_covid_timeseries(dir)
    build_movement_range(dir)
    build_social_proximity(dir)
end

build_datasets_main(".cache")
