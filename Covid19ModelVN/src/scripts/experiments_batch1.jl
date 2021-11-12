using Distributed
addprocs(exeflags = "--project")

const EXPERIMENTS_SCRIPTS = [
    "src/scripts/experiments_baseline.jl",
    "src/scripts/experiments_baselineA.jl",
    "src/scripts/experiments_fbmobility1.jl",
    "src/scripts/experiments_fbmobility1A.jl",
    "src/scripts/experiments_fbmobility2.jl",
    "src/scripts/experiments_fbmobility2A.jl",
    "src/scripts/experiments_fbmobility3.jl",
    "src/scripts/experiments_fbmobility3A.jl",
]

@sync @distributed for script in EXPERIMENTS_SCRIPTS
    include(script)
end
