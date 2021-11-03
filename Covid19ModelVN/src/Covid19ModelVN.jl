module Covid19ModelVN

include("helpers.jl")
include("models.jl")
include("train_eval.jl")

module FacebookData
include("data_facebook.jl")
end

module JHUCSSEData
include("data_jhu_csse.jl")
end

module PopulationData
include("data_population.jl")
end

module VnExpressData
include("data_vnexpress.jl")
end

module VnCdcData
include("data_vncdc.jl")
end

include("data_prebuilt.jl")

end
