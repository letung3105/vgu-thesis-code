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
            "snapshots/batchjob_fbmobility3_A",
            "fbmobility3",
            "fbmobility3",
        ],
        [
            "--locations",
            locations...,
            "--savedir",
            "snapshots/batchjob_fbmobility3_B",
            "--zeta",
            "0.005",
            "fbmobility3",
            "fbmobility3",
        ],
    ]
    @sync for args ∈ args_list
        @async runcmd(args)
    end
end
