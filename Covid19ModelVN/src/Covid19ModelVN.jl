module Covid19ModelVN

include("PopulationData.jl")
include("JHUCSSEData.jl")
include("VnExpressData.jl")
include("VnCdcData.jl")
include("FacebookData.jl")

include("Helpers.jl")
include("Datasets.jl")
include("Models.jl")
include("TrainEval.jl")

end
