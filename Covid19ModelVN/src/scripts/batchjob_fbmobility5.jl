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
            "--savedir=snapshots/batchjob_test",
            "--adam-lr=0.015",
            "--adam-maxiters=500",
            "fbmobility5_B",
            "fbmobility5",
        ],
        [
            "--locations",
            locations...,
            "--savedir=snapshots/batchjob_test",
            "--adam-lr=0.001",
            "--adam-maxiters=1000",
            "fbmobility5_B",
            "fbmobility5",
        ]
    ]
    @sync for args ∈ args_list
        @async runcmd(args)
    end
end
