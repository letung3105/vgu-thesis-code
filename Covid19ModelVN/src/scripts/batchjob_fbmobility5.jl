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
            "snapshots/batchjob_test",
            "fbmobility5_A",
            "fbmobility5",
        ],
        [
            "--locations",
            locations...,
            "--savedir",
            "snapshots/batchjob_test",
            "--zeta",
            "0.005",
            "fbmobility5_B",
            "fbmobility5",
        ],
    ]
    @sync for args ∈ args_list
        @async runcmd(args)
    end
end

