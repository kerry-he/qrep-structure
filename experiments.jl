import CSV
import DelimitedFiles

# List of problems to test
problems = [
    # "qrd",
    # "qrd_ef",
    "cc_ea",
    "cc_qq",
    "gse",
    "qkd"
]

# Test settings
csv_name = "out.csv"
all_tests = true

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
    include("examples/" * problem * ".jl")
    main(csv_name, all_tests)
end