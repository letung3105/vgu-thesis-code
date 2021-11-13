include("cmd.jl")

let
    locations = [
        collect(keys(Covid19ModelVN.LOC_NAMES_VN))
        collect(keys(Covid19ModelVN.LOC_NAMES_US))
    ]
    args_list = [
        [
            "--locations",
            locations...,
            "--savedir",
            "snapshots/batchjob_fbmobility2_A",
            "fbmobility2",
            "fbmobility2",
        ],
        [
            "--locations",
            locations...,
            "--savedir",
            "snapshots/batchjob_fbmobility2_B",
            "--zeta",
            "0.005",
            "fbmobility2",
            "fbmobility2",
        ],
    ]
    @sync for args ∈ args_list
        @async runcmd(args)
    end
end
