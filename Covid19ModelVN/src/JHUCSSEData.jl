module JHUCSSEData

using Dates, DataDeps, DataFrames

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
                "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv",
                "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_recovered_global.csv",
                "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv",
                "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_US.csv",
                "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_US.csv",
            ],
        ),
    )
    return nothing
end

function combine_country_level_timeseries(df_confirmed, df_recovered, df_deaths, country_name)
    # select rows associated with the country
    filter_country(df) =
        subset(df, "Country/Region" => x -> x .== country_name, view = true)
    # drop unused columns
    drop_cols(df) = select(
        df,
        Not(["Province/State", "Country/Region", "Lat", "Long"]),
        copycols = false,
    )
    # sum the values of all the regions within a country
    sum_reduce_cols(df) = combine(df, names(df, All()) .=> sum, renamecols = false)

    df_confirmed = sum_reduce_cols(drop_cols(filter_country(df_confirmed)))
    df_recovered = sum_reduce_cols(drop_cols(filter_country(df_recovered)))
    df_deaths = sum_reduce_cols(drop_cols(filter_country(df_deaths)))

    # combine 3 separated dataframes into 1
    df_combined = innerjoin(
        stack(df_confirmed, All(), variable_name = :date, value_name = :confirmed_total),
        stack(df_recovered, All(), variable_name = :date, value_name = :recovered_total),
        stack(df_deaths, All(), variable_name = :date, value_name = :deaths_total),
        on = :date
    )
    df_combined[!, :date] .= Date.(df_combined[!, :date], dateformat"mm/dd/yyyy")

    return df_combined
end

function combine_us_county_level_timeseries(df_confirmed, df_deaths, state_name, county_name)
    # select rows associated with the country
    filter_county(df) = subset(
        df,
        "Province_State" => x -> x .== state_name,
        "Admin2" => x -> x .== county_name,
        view = true,
    )
    # drop unused columns
    drop_cols(df) = select(
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
        stack(df_confirmed, All(), variable_name = :date, value_name = :confirmed_total),
        stack(df_deaths, All(), variable_name = :date, value_name = :deaths_total),
        on = :date
    )
    df_combined[!, :date] .= Date.(df_combined[!, :date], dateformat"mm/dd/yyyy")

    return df_combined
end

function get_us_county_population(df_deaths, state_name, county_name)
    # select rows associated with the country
    filter_county(df) = subset(
        df,
        "Province_State" => x -> x .== state_name,
        "Admin2" => x -> x .== county_name,
        view = true,
    )
    df_deaths = dropmissing(df_deaths, "Admin2", view = true)
    return first(filter_county(df_deaths).Population)
end

end # module JHUCSSEData
