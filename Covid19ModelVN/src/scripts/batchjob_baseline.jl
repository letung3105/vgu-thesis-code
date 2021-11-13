include("cmd.jl")

let
    locations = [
        Covid19ModelVN.LOC_CODE_VIETNAM
        Covid19ModelVN.LOC_CODE_UNITED_STATES
        collect(keys(Covid19ModelVN.LOC_NAMES_VN))
        collect(keys(Covid19ModelVN.LOC_NAMES_US))
    ]
    args_list = [
        [
            "--locations",
            locations...,
            "--savedir",
            "snapshots/batchjob_baseline_A",
            "baseline",
            "baseline",
        ],
        [
            "--locations",
            locations...,
            "--savedir",
            "snapshots/batchjob_baseline_B",
            "--zeta",
            "0.005",
            "baseline",
            "baseline",
        ],
    ]
    @sync for args ∈ args_list
        @async runcmd(args)
    end
end
