# Activate the environment for running the script
if isfile("Project.toml") && isfile("Manifest.toml")
    import Pkg
    Pkg.activate(".")
end

using Dates
import Covid19ModelVN.VnCdcData

df_cases_timeseries = VnCdcData.save_cases_timeseries(
    "datasets",
    "vncdc-timeseries-by-province",
    Date(2021, 4, 27),
    Date(2021, 10, 13),
    recreate=true,
)