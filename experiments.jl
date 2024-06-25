import CSV
import DelimitedFiles

include("examples/qrd.jl")
include("examples/qrd_ef.jl")
include("examples/cc_ea.jl")
include("examples/cc_qq.jl")
include("examples/gse.jl")
include("examples/qkd.jl")

# List of problems to test
problems = [
    "qrd",
    "qrd_ef",
    "cc_ea",
    "cc_qq",
    "gse",
    "qkd"
]

# Test settings
csv_name = "out.csv"
all_tests = false

header = [
    "problem",
    "description",
    "method",
    "status",
    "opt_val",
    "solve_time",
    "iter",
    "time_per_iter",
    "abs_gap",
    "rel_gap",
    "feas"
]
header = reshape(header, 1, length(header))
DelimitedFiles.writedlm(csv_name, header, ',')

for problem in problems
    if problem == "qrd"
        main_qrd(csv_name, all_tests)
    elseif problem == "qrd_ef"
        main_qrd_ef(csv_name, all_tests)
    elseif problem == "cc_ea"
        main_cc_ea(csv_name, all_tests)
    elseif problem == "cc_qq"
        main_cc_qq(csv_name, all_tests)
    elseif problem == "gse"
        main_gse(csv_name, all_tests)
    elseif problem == "qkd"
        main_qkd(csv_name, all_tests)
    end
end