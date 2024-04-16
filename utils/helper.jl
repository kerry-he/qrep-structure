import CSV
import Hypatia.Solvers

function try_solve(model, solver, problem, description, method, csv_name)
    try
        Solvers.load(solver, model)
        Solvers.solve(solver)
        save_and_print_stats(solver, problem, description, method, csv_name)
    catch exception
        save_and_print_fail(exception, problem, description, method, csv_name)
    end
    
    # Free memory
    model = nothing
    solver = nothing
    GC.gc()

end

function save_and_print_stats(solver, problem, description, method, csv_name)
    worst_gap = min(solver.gap / solver.point.tau[], abs(solver.primal_obj_t - solver.dual_obj_t))
    max_tau_obj = max(solver.point.tau[], min(abs(solver.primal_obj_t), abs(solver.dual_obj_t)))
    total_time = Solvers.get_solve_time(solver) - solver.time_rescale - solver.time_initx - solver.time_inity
    opt_val = (Solvers.get_primal_obj(solver) + Solvers.get_dual_obj(solver)) / 2

    println("problem:       ", problem)
    println("method:        ", method)
    println("description:   ", description)
    println("status:        ", solver.status)
    println("opt_val:       ", opt_val)
    println("solve_time:    ", total_time)
    println("iter:          ", Solvers.get_num_iters(solver))
    println("time_per_iter: ", total_time / Solvers.get_num_iters(solver))
    println("abs_gap:       ", solver.gap)
    println("rel_gap:       ", worst_gap / max_tau_obj)
    println("feas:          ", max(solver.x_feas, solver.y_feas, solver.z_feas))
    println()    
    
    CSV.write(csv_name, (
        problem       = [problem],
        description   = [description],
        method        = [method],
        status        = [string(solver.status)],
        opt_val       = [opt_val],
        solve_time    = [total_time],
        iter          = [Solvers.get_num_iters(solver)],
        time_per_iter = [total_time / Solvers.get_num_iters(solver)],
        abs_gap       = [solver.gap],
        rel_gap       = [worst_gap / max_tau_obj],
        feas          = [max(solver.x_feas, solver.y_feas, solver.z_feas)]
    ), writeheader = false, append = true, sep = ',')
end

function save_and_print_fail(exception, problem, description, method, csv_name)
    println("problem:       ", problem)
    println("method:        ", method)
    println("description:   ", description)
    println("status:        ", exception)
    println()    
    
    CSV.write(csv_name, (
        problem       = [problem],
        description   = [description],
        method        = [method],
        status        = [string(exception)]
    ), writeheader = false, append = true, sep = ',')
end