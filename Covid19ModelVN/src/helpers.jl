export TimeseriesDataset,
    TimeseriesConfig,
    train_test_split,
    load_timeseries,
    save_dataframe,
    lookup_saved_params,
    get_losses_plot_fpath,
    get_params_save_fpath,
    bound,
    bound!,
    moving_average,
    moving_average!

using Dates, Statistics, CSV, DataFrames

"""
A wrapper round `AbstractDataFrame` that contains additional information for the names
of the columns that contain the timeseries data and the column that contains the timestamps
of the data points

# Arguments

* `df`: the dataframe
* `date_col`: name of the column that contains the timestamps
* `data_cols`: names of the columns that contains the timeseries data
"""
struct TimeseriesConfig{Df<:AbstractDataFrame}
    df::Df
    date_col::String
    data_cols::Vector{String}
end

"""
Read the timeseries given by `config` and returns a matrix that contains the timeseries
data whose timestamps lie between `first_date` and `last_date`

# Arguments

* `config`: configuration for the timeseries data
* `first_date`: earliest date allowed in the timeseries
* `last_date`: latest date allowed in the timeseries
"""
function load_timeseries(config::TimeseriesConfig, first_date::Date, last_date::Date)
    df = bound(config.df, config.date_col, first_date, last_date, view = true)
    data = Array(df[!, Cols(config.data_cols)])
    return data'
end

"""
This contains the minimum required information for a timeseriese dataset that is used by UDEs

# Fields

* `data`: an array that holds the timeseries data
* `tspan`: the first and last time coordinates of the timeseries data
* `tsteps`: collocations points
"""
struct TimeseriesDataset{R<:Real,Ts}
    data::Matrix{R}
    tspan::Tuple{R,R}
    tsteps::Ts
end

"""
Split the timeseries data into 2 parts where the first contains data points whose timestamps
lie between `[first_date, split_date]` and the second contains data points whose timestamps
lie between `(split_date, last_date]`

# Arguments
* `config`: configuration for the timeseries data
* `first_date`: earliest date allowed in the first returned timeseries
* `split_date`: latest date allowed in the first returned timeseries
* `last_date`: latest date allowed in the second returned timeseries
"""
function train_test_split(
    config::TimeseriesConfig,
    first_date::Date,
    split_date::Date,
    last_date::Date,
)
    train_data = load_timeseries(config, first_date, split_date)
    train_tspan = Float64.((0, Dates.value(split_date - first_date)))
    train_tsteps = train_tspan[1]:1:train_tspan[2]
    train_dataset = TimeseriesDataset(train_data, train_tspan, train_tsteps)

    test_data = load_timeseries(config, split_date + Day(1), last_date)
    test_tspan = Float64.((0, Dates.value(last_date - first_date)))
    test_tsteps = (train_tspan[2]+1):1:test_tspan[2]
    test_dataset = TimeseriesDataset(test_data, test_tspan, test_tsteps)

    return train_dataset, test_dataset
end

"""
Save a dataframe as a CSV file

# Arguments

+ `df`: The dataframe to save
+ `fpath`: The path to save the file
"""
function save_dataframe(df::AbstractDataFrame, fpath::AbstractString)
    # create containing folder if not exists
    if !isdir(dirname(fpath))
        mkpath(dirname(fpath))
    end
    CSV.write(fpath, df)
    return fpath
end

"""
Get the file paths and uuids of all the saved parameters of an experiment

# Arguments

* `dir`: the directory that contains the saved parameters
"""
function lookup_saved_params(dir::AbstractString)
    params_files = filter(x -> endswith(x, ".jls"), readdir(dir))
    fpaths = map(f -> joinpath(dir, f), params_files)
    uuids = map(f -> first(rsplit(f, ".", limit = 3)), params_files)
    return fpaths, uuids
end

"""
Get default losses figure file path

# Arguments

* `fdir`: the root directory of the file
* `uuid`: the file unique identifier
"""
get_losses_plot_fpath(fdir::AbstractString, uuid::AbstractString) =
    joinpath(fdir, "$uuid.losses.png")

"""
Get default file path for saved parameters

# Arguments

* `fdir`: the root directory of the file
* `uuid`: the file unique identifier
"""
get_params_save_fpath(fdir::AbstractString, uuid::AbstractString) =
    joinpath(fdir, "$uuid.params.jls")

"""
Select a subset of the dataframe `df` such that values in `col` remain between `start_date` and `end_date`

# Arguments

+ `df`: An arbitrary dataframe
+ `col`: The key column used for filtering
+ `first`: The starting (smallest) value allowed
+ `last`: The ending (largest) value allowed
"""
bound(
    df::AbstractDataFrame,
    col::Union{Symbol,AbstractString},
    first::Any,
    last::Any;
    kwargs...,
) = subset(df, col => x -> (x .>= first) .& (x .<= last); kwargs...)

"""
Filter the dataframe `df` such that values in `col` remain between `start_date` and `end_date`

# Arguments

+ `df`: An arbitrary dataframe
+ `col`: The key column used for filtering
+ `first`: The starting (smallest) value allowed
+ `last`: The ending (largest) value allowed
"""
bound!(
    df::AbstractDataFrame,
    col::Union{Symbol,AbstractString},
    first::Any,
    last::Any;
    kwargs...,
) = subset!(df, col => x -> (x .>= first) .& (x .<= last); kwargs...)

"""
Calculate the moving average of the given list of numbers

# Arguments

+ `xs`: The list of number
+ `n`: Subset size to average over
"""
moving_average(xs::AbstractVector{<:Real}, n::Integer) =
    [mean(@view xs[(i >= n ? i - n + 1 : 1):i]) for i ∈ 1:length(xs)]

"""
Calculate the moving average of all the `cols` in `df`

# Arguments

+ `df`: A `DataFrame`
+ `cols`: Column names for calculating the moving average
+ `n`: Subset size to average over
"""
moving_average!(
    df::AbstractDataFrame,
    cols::Union{
        Symbol,
        AbstractString,
        AbstractVector{Symbol},
        AbstractVector{<:AbstractString},
    },
    n::Integer,
) = transform!(df, names(df, Cols(cols)) .=> x -> moving_average(x, n), renamecols = false)
