# Activate the environment for running the script
if isfile("Project.toml") && isfile("Manifest.toml")
    import Pkg
    Pkg.activate(".")
end

using Covid19ModelVN

function build_datasets_main(dir::AbstractString = ""; recreate::Bool = false)
    build_population(dir; recreate)
    build_covid_timeseries(dir; recreate)
    build_movement_range(dir; recreate)
    build_social_proximity(dir; recreate)
end

build_datasets_main(".cache", recreate = true)
