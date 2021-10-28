using CSV, Dates, DataDeps, DataFrames, Covid19ModelVN

function __init__()
    register(
        DataDep(
            "jhu-csse",
            """
            Dataset: COVID-19 Data Repository by the Center for Systems Science and Engineering (CSSE) at Johns Hopkins University
            Email: jhusystems@gmail.com
            Website: https://github.com/CSSEGISandData/COVID-19

            This is the data repository for the 2019 Novel Coronavirus Visual Dashboard operated by the Johns Hopkins University Center for Systems Science and Engineering (JHU CSSE). Also, supported by ESRI Living Atlas Team and the Johns Hopkins University Applied Physics Lab (JHU APL).

            1. This data set is licensed under the Creative Commons Attribution 4.0 International (CC BY 4.0) by the Johns Hopkins University on behalf of its Center for Systems Science in Engineering. Copyright Johns Hopkins University 2020.
            2. Attribute the data as the "COVID-19 Data Repository by the Center for Systems Science and Engineering (CSSE) at Johns Hopkins University" or "JHU CSSE COVID-19 Data" for short, and the url: https://github.com/CSSEGISandData/COVID-19.
            3. For publications that use the data, please cite the following publication: "Dong E, Du H, Gardner L. An interactive web-based dashboard to track COVID-19 in real time.  Lancet Inf Dis. 20(5):533-534. doi: 10.1016/S1473-3099(20)30120-1"
            """,
            [
                "https://github.com/CSSEGISandData/COVID-19/raw/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv",
                "https://github.com/CSSEGISandData/COVID-19/raw/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_recovered_global.csv",
                "https://github.com/CSSEGISandData/COVID-19/raw/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv",
                "https://github.com/CSSEGISandData/COVID-19/raw/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_US.csv",
                "https://github.com/CSSEGISandData/COVID-19/raw/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_US.csv",
            ],
        ),
    )
    return nothing
end

"""
Turn the cases dataframe in wide-format to long-format

# Arguments

+ `df`: the dataframe
+ `value_name`: name of the value column of the stacked dataframe
"""
stack_timeseries(df::AbstractDataFrame, value_name::Union{<:AbstractString,Symbol}) =
    stack(df, All(), variable_name = :date, value_name = value_name, view = true)

"""
Combine 3 separated timeseries from JHU-CSSE into 1 single dataframe for a specific country

# Arguments

+ `df_confirmed`: the dataframe that contains the confirmed cases count
+ `df_recovered`: the dataframe that contains the recovered cases count
+ `df_deaths`: the dataframe that contains the deaths count
+ `country_name`: the name of the country whose data is extracted
"""
function combine_country_level_timeseries(
    df_confirmed::AbstractDataFrame,
    df_recovered::AbstractDataFrame,
    df_deaths::AbstractDataFrame,
    country_name::AbstractString,
)
    # select rows associated with the country
    filter_country(df::AbstractDataFrame) =
        subset(df, "Country/Region" => x -> x .== country_name, view = true)
    # drop unused columns
    drop_cols(df::AbstractDataFrame) = select(
        df,
        Not(["Province/State", "Country/Region", "Lat", "Long"]),
        copycols = false,
    )
    # sum the values of all the regions within a country
    sum_reduce_cols(df::AbstractDataFrame) =
        combine(df, names(df, All()) .=> sum, renamecols = false)

    df_confirmed = sum_reduce_cols(drop_cols(filter_country(df_confirmed)))
    df_recovered = sum_reduce_cols(drop_cols(filter_country(df_recovered)))
    df_deaths = sum_reduce_cols(drop_cols(filter_country(df_deaths)))

    # combine 3 separated dataframes into 1
    df_combined = innerjoin(
        stack_timeseries(df_confirmed, :confirmed_total),
        stack_timeseries(df_recovered, :recovered_total),
        stack_timeseries(df_deaths, :deaths_total),
        on = :date,
    )
    df_combined[!, :date] .=
        Date.(df_combined[!, :date], dateformat"mm/dd/yyyy") .+ Year(2000)
    df_combined.infective =
        df_combined.confirmed_total .- df_combined.recovered_total .-
        df_combined.deaths_total

    return df_combined
end

"""
Read and combine 3 separated timeseries from JHU-CSSE into 1 single dataframe for a specific country,
then save the resulting dataframe into a CSV file.

# Arguments

+ `fpath_outputs`: a list of output CSV files where data for each country in `country_names` is saved
to the corresponding path at the same index.
+ `country_names`: a list of country names.
+ `fpath_confirmed`: path to the confirmed CSV
+ `fpath_recovered`: path to the confirmed CSV
+ `fpath_deaths`: path to the deaths CSV
+ `recreate`: the existing file will be ovewritten if this is true
"""
function save_country_level_timeseries(
    fpath_outputs::AbstractVector{<:AbstractString},
    country_names::AbstractVector{<:AbstractString};
    fpath_confirmed::AbstractString = datadep"jhu-csse/time_series_covid19_confirmed_global.csv",
    fpath_recovered::AbstractString = datadep"jhu-csse/time_series_covid19_recovered_global.csv",
    fpath_deaths::AbstractString = datadep"jhu-csse/time_series_covid19_deaths_global.csv",
    recreate::Bool = false,
)
    if all(isfile, fpath_outputs) && !recreate
        return nothing
    end

    df_confirmed = CSV.read(fpath_confirmed, DataFrame)
    df_recovered = CSV.read(fpath_recovered, DataFrame)
    df_deaths = CSV.read(fpath_deaths, DataFrame)

    for (fpath, country_name) ∈ zip(fpath_outputs, country_names)
        if isfile(fpath) && !recreate
            continue
        end

        @info "Generating '$fpath"
        df_combined = combine_country_level_timeseries(
            df_confirmed,
            df_recovered,
            df_deaths,
            country_name,
        )
        save_dataframe(df_combined, fpath)
    end

    return nothing
end

"""
Read and combine 2 separated timeseries from JHU-CSSE into 1 single dataframe for a specific county,
then save the resulting dataframe into a CSV file.

# Arguments

+ `fpath_outputs`: a list of output CSV files where data for each county in `county_names` is saved
to the corresponding path at the same index.
+ `state_names`: a list of state names.
+ `county_names`: a list of county names.
+ `fpath_confirmed`: path to the confirmed CSV
+ `fpath_recovered`: path to the confirmed CSV
+ `fpath_deaths`: path to the deaths CSV
+ `recreate`: the existing file will be ovewritten if this is true
"""
function save_us_county_level_timeseries(
    fpath_outputs::AbstractVector{<:AbstractString},
    state_names::AbstractVector{<:AbstractString},
    county_names::AbstractVector{<:AbstractString};
    fpath_confirmed::AbstractString = datadep"jhu-csse/time_series_covid19_confirmed_US.csv",
    fpath_deaths::AbstractString = datadep"jhu-csse/time_series_covid19_deaths_US.csv",
    recreate::Bool = false,
)
    if all(isfile, fpath_outputs) && !recreate
        return nothing
    end

    df_confirmed = CSV.read(fpath_confirmed, DataFrame)
    df_deaths = CSV.read(fpath_deaths, DataFrame)

    for (fpath, state_name, county_name) ∈ zip(fpath_outputs, state_names, county_names)
        if isfile(fpath) && !recreate
            continue
        end

        @info "Generating '$fpath"
        df_combined = combine_us_county_level_timeseries(
            df_confirmed,
            df_deaths,
            state_name,
            county_name,
        )
        save_dataframe(df_combined, fpath)
    end

    return nothing
end

"""
Combine 2 separated timeseries from JHU-CSSE into 1 single dataframe for a specific US county

# Arguments

+ `df_confirmed`: the dataframe that contains the confirmed cases count
+ `df_deaths`: the dataframe that contains the deaths count
+ `state_name`: the name of the state in which the county is located
+ `county_name`: the name of the county whose data is extracted
"""
function combine_us_county_level_timeseries(
    df_confirmed::AbstractDataFrame,
    df_deaths::AbstractDataFrame,
    state_name::AbstractString,
    county_name::AbstractString,
)
    # select rows associated with the country
    filter_county(df::AbstractDataFrame) = subset(
        df,
        "Province_State" => x -> x .== state_name,
        "Admin2" => x -> x .== county_name,
        view = true,
    )
    # drop unused columns
    drop_cols(df::AbstractDataFrame) = select(
        df,
        Not([
            "UID",
            "iso2",
            "iso3",
            "code3",
            "FIPS",
            "Admin2",
            "Country_Region",
            "Province_State",
            "Lat",
            "Long_",
            "Combined_Key",
        ]),
        copycols = false,
    )

    # ignore counties with missing names
    df_confirmed = dropmissing(df_confirmed, "Admin2", view = true)
    df_deaths = dropmissing(df_deaths, "Admin2", view = true)
    # deaths dataframe has an extra population column
    df_deaths = select(df_deaths, Not("Population"), copycols = false)

    df_confirmed = drop_cols(filter_county(df_confirmed))
    df_deaths = drop_cols(filter_county(df_deaths))

    # combine 2 separated dataframes into 1
    df_combined = innerjoin(
        stack_timeseries(df_confirmed, :confirmed_total),
        stack_timeseries(df_deaths, :deaths_total),
        on = :date,
    )
    df_combined[!, :date] .= Date.(df_combined[!, :date], dateformat"mm/dd/yyyy")

    return df_combined
end

"""
Read the county's population from JHU-CSSE deaths timeseries

# Arguments

+ `df_deaths`: the dataframe that contains the deaths count
+ `state_name`: the name of the state in which the county is located
+ `county_name`: the name of the county whose data is extracted
"""
function get_us_county_population(
    df_deaths::AbstractDataFrame,
    state_name::AbstractString,
    county_name::AbstractString,
)
    # select rows associated with the country
    filter_county(df::AbstractDataFrame) = subset(
        df,
        "Province_State" => x -> x .== state_name,
        "Admin2" => x -> x .== county_name,
        view = true,
    )
    df_deaths = dropmissing(df_deaths, "Admin2", view = true)
    return first(filter_county(df_deaths).Population)
end
