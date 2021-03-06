"""
    TimeseriesConfig{DF<:AbstractDataFrame}

A wrapper round `AbstractDataFrame` that contains additional information for the names
of the columns that contain the timeseries data and the column that contains the timestamps
of the data points

# Arguments

* `df`: the dataframe
* `date_col`: name of the column that contains the timestamps
* `data_cols`: names of the columns that contains the timeseries data
"""
struct TimeseriesConfig{DF<:AbstractDataFrame}
    df::DF
    date_col::String
    data_cols::Vector{String}
end

"""
    load_timeseries(config::TimeseriesConfig, first_date::Date, last_date::Date)

Read the timeseries given by `config` and returns a matrix that contains the timeseries
data whose timestamps lie between `first_date` and `last_date`

# Arguments

* `config`: configuration for the timeseries data
* `first_date`: earliest date allowed in the timeseries
* `last_date`: latest date allowed in the timeseries
"""
function load_timeseries(config::TimeseriesConfig, first_date::Date, last_date::Date)
    df = bound(config.df, config.date_col, first_date, last_date; view=true)
    data = Array(df[!, Cols(config.data_cols)])
    return data'
end

"""
    TimeseriesDataset{R<:Real,Data<:AbstractMatrix{R},Tspan<:Tuple{R,R},Tsteps}

This contains the minimum required information for a timeseriese dataset that is used by UDEs

# Fields

* `data`: an array that holds the timeseries data
* `tspan`: the first and last time coordinates of the timeseries data
* `tsteps`: collocations points
"""
struct TimeseriesDataset{R<:Real,Data<:AbstractMatrix{R},Tspan<:Tuple{R,R},Tsteps}
    data::Data
    tspan::Tspan
    tsteps::Tsteps
end

struct TimeseriesDataLoader{DS<:TimeseriesDataset}
    dataset::DS
    batchsize::Int
    indices::Vector{Int}

    function TimeseriesDataLoader(dataset::DS, batchsize::Int) where {DS<:TimeseriesDataset}
        return new{DS}(dataset, batchsize, Vector(1:batchsize:size(dataset.data, 2)))
    end
end

Base.iterate(loader::TimeseriesDataLoader) = iterate(loader, 0)

function Base.iterate(loader::TimeseriesDataLoader, cursor)
    cursor += 1
    if cursor > length(loader.indices)
        return nothing
    end

    start = loader.indices[cursor]
    stop = min(start + loader.batchsize - 1, size(loader.dataset.data, 2))

    data = @view loader.dataset.data[:, start:stop]
    tsteps = loader.dataset.tsteps[start:stop]
    tspan = (loader.dataset.tspan[1], tsteps[end])

    return (data, tspan, tsteps), cursor
end

"""
    train_test_split(
        config::TimeseriesConfig,
        first_date::Date,
        split_date::Date,
        last_date::Date,
    )

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
    config::TimeseriesConfig, first_date::Date, split_date::Date, last_date::Date
)
    train_data = load_timeseries(config, first_date, split_date)
    train_tspan = Float64.((0, Dates.value(split_date - first_date)))
    train_tsteps = train_tspan[1]:1:train_tspan[2]
    train_dataset = TimeseriesDataset(train_data, train_tspan, train_tsteps)

    test_data = load_timeseries(config, split_date + Day(1), last_date)
    test_tspan = Float64.((0, Dates.value(last_date - first_date)))
    test_tsteps = (train_tspan[2] + 1):1:test_tspan[2]
    test_dataset = TimeseriesDataset(test_data, test_tspan, test_tsteps)

    return train_dataset, test_dataset
end

"""
    save_dataframe(df::AbstractDataFrame, fpath::AbstractString)

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
    lookup_saved_params(dir::AbstractString)

Get the file paths and uuids of all the saved parameters of an experiment

# Arguments

* `dir`: the directory that contains the saved parameters
"""
function lookup_saved_params(dir::AbstractString)
    params_files = filter(x -> endswith(x, ".jls"), readdir(dir))
    fpaths = map(f -> joinpath(dir, f), params_files)
    return fpaths
end

"""
    get_losses_save_fpath(fdir::AbstractString, uuid::AbstractString)

Get default losses figure file path

# Arguments

* `fdir`: the root directory of the file
* `uuid`: the file unique identifier
"""
function get_losses_save_fpath(fdir::AbstractString, uuid::AbstractString)
    return joinpath(fdir, "$uuid.losses.jls")
end

"""
    get_params_save_fpath(fdir::AbstractString, uuid::AbstractString)

Get default file path for saved parameters

# Arguments

* `fdir`: the root directory of the file
* `uuid`: the file unique identifier
"""
function get_params_save_fpath(fdir::AbstractString, uuid::AbstractString)
    return joinpath(fdir, "$uuid.params.jls")
end

"""
    get_forecasts_save_fpath(fdir::AbstractString, uuid::AbstractString)

Get default file path for forecasts made during training

# Arguments

* `fdir`: the root directory of the file
* `uuid`: the file unique identifier
"""
function get_forecasts_save_fpath(fdir::AbstractString, uuid::AbstractString)
    return joinpath(fdir, "$uuid.forecasts.jls")
end

"""
    bound(
        df::AbstractDataFrame,
        col::Union{Symbol,AbstractString},
        first::Any,
        last::Any;
        kwargs...,
    )

Select a subset of the dataframe `df` such that values in `col` remain between `start_date` and `end_date`

# Arguments

+ `df`: An arbitrary dataframe
+ `col`: The key column used for filtering
+ `first`: The starting (smallest) value allowed
+ `last`: The ending (largest) value allowed
"""
function bound(
    df::AbstractDataFrame,
    col::Union{Symbol,AbstractString},
    first::Any,
    last::Any;
    kwargs...,
)
    return subset(df, col => x -> (x .>= first) .& (x .<= last); kwargs...)
end

"""
    bound!(
        df::AbstractDataFrame,
        col::Union{Symbol,AbstractString},
        first::Any,
        last::Any;
        kwargs...,
    )

Filter the dataframe `df` such that values in `col` remain between `start_date` and `end_date`

# Arguments

+ `df`: An arbitrary dataframe
+ `col`: The key column used for filtering
+ `first`: The starting (smallest) value allowed
+ `last`: The ending (largest) value allowed
"""
function bound!(
    df::AbstractDataFrame,
    col::Union{Symbol,AbstractString},
    first::Any,
    last::Any;
    kwargs...,
)
    return subset!(df, col => x -> (x .>= first) .& (x .<= last); kwargs...)
end

"""
    moving_average(xs::AbstractVector{<:Real}, n::Integer)

Calculate the moving average of the given list of numbers

# Arguments

+ `xs`: The list of number
+ `n`: Subset size to average over
"""
function moving_average(xs::AbstractVector{<:Real}, n::Integer)
    return [mean(@view xs[(i >= n ? i - n + 1 : 1):i]) for i in 1:length(xs)]
end

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
    )

Calculate the moving average of all the `cols` in `df`

# Arguments

+ `df`: A `DataFrame`
+ `cols`: Column names for calculating the moving average
+ `n`: Subset size to average over
"""
function moving_average!(
    df::AbstractDataFrame,
    cols::Union{
        Symbol,AbstractString,AbstractVector{Symbol},AbstractVector{<:AbstractString}
    },
    n::Integer,
)
    return transform!(
        df, names(df, Cols(cols)) .=> x -> moving_average(x, n); renamecols=false
    )
end
