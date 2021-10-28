export UDEDataset,
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
This contains the minimum required information for a timeseriese dataset that is used by UDEs

# Fields

* `data`: an array that holds the timeseries data
* `tspan`: the first and last time coordinates of the timeseries data
* `tsteps`: collocations points
"""
struct UDEDataset
    data::AbstractMatrix{<:Real}
    tspan::Tuple{<:Real,<:Real}
    tsteps::Union{<:Real,AbstractVector{<:Real},StepRange,StepRangeLen}
end

"""
Create two `UDEDataset`s from the given dataframe, the first dataset contains data point whose `date_col`
value is in the range [first_date, split_date], and the second dataset contains data point whose `date_col`
value is in the range (split_date, last_date]

# Arguments

+ `df`: The dataframe
+ `data_cols`: The names of the columns whose data will be used for creating an array
+ `date_col`: The name of the column that contains the date of the data point
+ `first_date`: First date to take
+ `split_date`: Date where to dataframe is splitted in two
+ `last_date`: Last date to take
"""
function train_test_split(
    df::AbstractDataFrame,
    data_cols::Union{
        AbstractVector{<:AbstractString},
        AbstractVector{Symbol},
        <:AbstractString,
        Symbol,
    },
    date_col::Union{<:AbstractString,Symbol},
    first_date::Date,
    split_date::Date,
    last_date::Date,
)
    df_train = bound(df, date_col, first_date, split_date)
    df_test = bound(df, date_col, split_date + Day(1), last_date)

    train_tspan = Float64.((0, Dates.value(split_date - first_date)))
    test_tspan = Float64.((0, Dates.value(last_date - first_date)))

    train_tsteps = train_tspan[1]:1:train_tspan[2]
    test_tsteps = (train_tspan[2]+1):1:test_tspan[2]

    train_data = Float64.(Array(df_train[!, data_cols])')
    test_data = Float64.(Array(df_test[!, data_cols])')

    train_dataset = UDEDataset(train_data, train_tspan, train_tsteps)
    test_dataset = UDEDataset(test_data, test_tspan, test_tsteps)

    return train_dataset, test_dataset
end

"""
Load from the time series in the dataframe the data from `data_cols` columns, limiting
the data point `date_col` between [`first_date`, `last_date`].

# Arguments

+ `df`: The dataframe
+ `data_cols`: The names of the columns whose data will be used for creating an array
+ `date_col`: The name of the column that contains the date of the data point
+ `first_date`: First date to take
+ `last_date`: Last date to take
"""
function load_timeseries(
    df::AbstractDataFrame,
    data_cols::Union{
        AbstractVector{<:AbstractString},
        AbstractVector{Symbol},
        <:AbstractString,
        Symbol,
    },
    date_col::Union{<:AbstractString,Symbol},
    first_date::Date,
    last_date::Date,
)
    df = bound(df, date_col, first_date, last_date)
    return Array(df[!, Cols(data_cols)])
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
bound(df::AbstractDataFrame, col::Union{<:AbstractString,Symbol}, first::Any, last::Any) =
    subset(df, col => x -> (x .>= first) .& (x .<= last), view = true)

"""
Filter the dataframe `df` such that values in `col` remain between `start_date` and `end_date`

# Arguments

+ `df`: An arbitrary dataframe
+ `col`: The key column used for filtering
+ `first`: The starting (smallest) value allowed
+ `last`: The ending (largest) value allowed
"""
bound!(df::AbstractDataFrame, col::Union{<:AbstractString,Symbol}, first::Any, last::Any) =
    subset!(df, col => x -> (x .>= first) .& (x .<= last))

"""
Calculate the moving average of the given list of numbers

# Arguments

+ `xs`: The list of number
+ `n`: Subset size to average over
"""
moving_average(xs::AbstractVector{<:Real}, n::Integer) =
    [mean(@view xs[(i >= n ? i - n + 1 : 1):i]) for i = 1:length(xs)]

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
        AbstractVector{<:AbstractString},
        AbstractVector{Symbol},
        <:AbstractString,
        Symbol,
    },
    n::Integer,
) = transform!(df, names(df, Cols(cols)) .=> x -> moving_average(x, n), renamecols = false)
