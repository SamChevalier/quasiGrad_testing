# %% ============== zsus :) ===========
using QuasiGrad
using Revise
include("./test_functions.jl")

path    = "C:/Users/Samuel.HORACE/Dropbox (Personal)/Documents/Julia/GO3_testcases/C3S1_20221222/C3S1N00600D1/scenario_001.json"
InFile1 = path
jsn     = QuasiGrad.load_json(InFile1)

# initialize
adm, cgd, ctg, flw, grd, idx, lbf, mgd, ntk, prm, qG, scr, stt, sys, upd = QuasiGrad.base_initialization(jsn, perturb_states=true, pert_size=1.0);
QuasiGrad.update_states_and_grads!(cgd, ctg, flw, grd, idx, mgd, ntk, prm, qG, scr, stt, sys)

# %% ===
qG.num_threads = 1
@time QuasiGrad.solve_ctgs!(cgd, ctg, flw, grd, idx, mgd, ntk, prm, qG, scr, stt, sys)

# %% ===
qG.num_threads = 12
@time QuasiGrad.solve_ctgs!(cgd, ctg, flw, grd, idx, mgd, ntk, prm, qG, scr, stt, sys)

# %% test solution
adm, cgd, ctg, flw, grd, idx, lbf, mgd, ntk, prm, qG, scr, stt, sys, upd = QuasiGrad.base_initialization(jsn, perturb_states=true, pert_size=1.0);
QuasiGrad.update_states_and_grads!(cgd, ctg, flw, grd, idx, mgd, ntk, prm, qG, scr, stt, sys)

QuasiGrad.project!(100.0, idx, prm, qG, stt, sys, upd, final_projection = false)
QuasiGrad.project!(100.0, idx, prm, qG, stt, sys, upd, final_projection = true)

# %%

qG.num_threads = 12
@time QuasiGrad.solve_ctgs!(cgd, ctg, flw, grd, idx, mgd, ntk, prm, qG, scr, stt, sys)

# %%
QuasiGrad.write_solution("solution.jl", prm, qG, stt, sys)


# %% 0)
qG.num_threads = 12
@time QuasiGrad.economic_dispatch_initialization!(cgd, ctg, flw, grd, idx, mgd, ntk, prm, qG, scr, stt, sys, upd)

# %% 1)
@btime QuasiGrad.flush_gradients!(grd, mgd, prm, qG, sys)

# %% 2)
#QuasiGrad.clip_all!(prm, qG, stt, sys)
qG.num_threads = 12
@btime QuasiGrad.clip_all!(prm, qG, stt, sys)

# %% 3)
qG.num_threads = 1
@btime QuasiGrad.acline_flows!(grd, idx, prm, qG, stt, sys)

qG.num_threads = 12
@btime QuasiGrad.acline_flows!(grd, idx, prm, qG, stt, sys)

# %% 4)
qG.num_threads = 1
@btime QuasiGrad.xfm_flows!(grd, idx, prm, qG, stt, sys)

qG.num_threads = 12
@btime QuasiGrad.xfm_flows!(grd, idx, prm, qG, stt, sys)

# %% 5)
qG.num_threads = 1
@btime QuasiGrad.shunts!(grd, idx, prm, qG, stt)

qG.num_threads = 12
@btime QuasiGrad.shunts!(grd, idx, prm, qG, stt)

# %% 6)
qG.num_threads = 1
@btime QuasiGrad.all_device_statuses_and_costs!(grd, prm, qG, stt)

qG.num_threads = 12
@btime QuasiGrad.all_device_statuses_and_costs!(grd, prm, qG, stt)

# %% 7)
qG.num_threads = 1
@btime QuasiGrad.device_startup_states!(grd, idx, mgd, prm, qG, stt, sys)

qG.num_threads = 12
@btime QuasiGrad.device_startup_states!(grd, idx, mgd, prm, qG, stt, sys)

# %% 8)
qG.num_threads = 1
@btime QuasiGrad.device_active_powers!(idx, prm, qG, stt, sys)

qG.num_threads = 12
@btime QuasiGrad.device_active_powers!(idx, prm, qG, stt, sys)

# %% 9) 
qG.num_threads = 1
@btime QuasiGrad.device_reactive_powers!(idx, prm, qG, stt)

qG.num_threads = 12
@btime QuasiGrad.device_reactive_powers!(idx, prm, qG, stt)

# %% 10) 
qG.num_threads = 1
@btime QuasiGrad.energy_costs!(grd, prm, qG, stt, sys)

qG.num_threads = 12
@btime QuasiGrad.energy_costs!(grd, prm, qG, stt, sys)

# %% 11)
qG.num_threads = 1
@btime QuasiGrad.energy_penalties!(grd, idx, prm, qG, scr, stt, sys)

qG.num_threads = 12
@btime QuasiGrad.energy_penalties!(grd, idx, prm, qG, scr, stt, sys)

# %% 12)
qG.num_threads = 1
@btime QuasiGrad.penalized_device_constraints!(grd, idx, mgd, prm, qG, scr, stt, sys)

qG.num_threads = 12
@btime QuasiGrad.penalized_device_constraints!(grd, idx, mgd, prm, qG, scr, stt, sys)

# %% 13)
qG.num_threads = 1
@btime QuasiGrad.device_reserve_costs!(prm, qG, stt)

qG.num_threads = 12
@btime QuasiGrad.device_reserve_costs!(prm, qG, stt)

# %% 13)
qG.num_threads = 1
@btime QuasiGrad.power_balance!(grd, idx, prm, qG, stt, sys)

qG.num_threads = 12
@btime QuasiGrad.power_balance!(grd, idx, prm, qG, stt, sys)

# %% 14)
qG.num_threads = 1
@btime QuasiGrad.reserve_balance!(idx, prm, qG, stt, sys)

qG.num_threads = 12
@btime QuasiGrad.reserve_balance!(idx, prm, qG, stt, sys)

# %% 15)
qG.num_threads = 1
@btime QuasiGrad.master_grad_solve_pf!(cgd, grd, idx, mgd, prm, qG, stt, sys)

qG.num_threads = 12
@btime QuasiGrad.master_grad_solve_pf!(cgd, grd, idx, mgd, prm, qG, stt, sys)

# %% compute the master grad
qG.num_threads = 1
@btime QuasiGrad.master_grad!(cgd, grd, idx, mgd, prm, qG, stt, sys)

qG.num_threads = 12
@btime QuasiGrad.master_grad!(cgd, grd, idx, mgd, prm, qG, stt, sys)

# %% 16)
qG.num_threads = 1
@btime QuasiGrad.initialize_ctg(sys, prm, qG, idx);

qG.num_threads = 12
@btime QuasiGrad.initialize_ctg(sys, prm, qG, idx);

# %% 17)
qG.num_threads = 1
@btime QuasiGrad.solve_pf_lbfgs!(pf_lbfgs, pf_lbfgs_diff, pf_lbfgs_idx, pf_lbfgs_map, pf_lbfgs_step, mgd, prm, qG, stt, upd, zpf)

qG.num_threads = 12
@btime QuasiGrad.solve_pf_lbfgs!(pf_lbfgs, pf_lbfgs_diff, pf_lbfgs_idx, pf_lbfgs_map, pf_lbfgs_step, mgd, prm, qG, stt, upd, zpf)



# %% test power flow solving!!
path     = "C:/Users/Samuel.HORACE/Dropbox (Personal)/Documents/Julia/GO3_testcases/C3S1_20221222/C3S1N00600D1/scenario_001.json"
InFile1  = path
jsn      = QuasiGrad.load_json(InFile1)
Division = 1

adm, cgd, ctg, flw, grd, idx, lbf, mgd, ntk, prm, qG, scr, stt, sys, upd = QuasiGrad.base_initialization(jsn, Div=Division);

# %% I3. run an economic dispatch and update the states
# @time QuasiGrad.economic_dispatch_initialization!(cgd, ctg, flw, grd, idx, mgd, ntk, prm, qG, scr, stt, sys, upd)

# %% ===
qG.print_projection_success = false
final_projection = false
@time QuasiGrad.solve_Gurobi_projection!(final_projection, idx, prm, qG, stt, sys, upd)

# %% ===
qG.print_projection_success = false
final_projection = false
@time QuasiGrad.solve_Gurobi_projection_parallel!(final_projection, idx, prm, qG, stt, sys, upd)

# %%
adm, cgd, ctg, flw, grd, idx, lbf, mgd, ntk, prm, qG, scr, stt, sys, upd = QuasiGrad.base_initialization(jsn, Div=Division);

qG.print_projection_success = false
final_projection = false

QuasiGrad.solve_Gurobi_projection!(final_projection, idx, prm, qG, stt, sys, upd)

# %%
@time QuasiGrad.soft_reserve_cleanup!(idx, prm, qG, stt, sys, upd);

# %%
@time QuasiGrad.soft_reserve_cleanup_not_parallel!(idx, prm, qG, stt, sys, upd);

# %%
@time QuasiGrad.reserve_cleanup!(idx, prm, qG, stt, sys, upd);

# %%
@time QuasiGrad.soft_reserve_cleanup_not_parallel!(idx, prm, qG, stt, sys, upd);
# %%
@time QuasiGrad.reserve_cleanup_not_parallel!(idx, prm, qG, stt, sys, upd);
# %%
gurobi_env = Gurobi.Env()

# %%

model = Model(optimizer_with_attributes(() -> Gurobi.Optimizer(gurobi_env), "Threads" => 1, "OutputFlag" => 0))

# %% solve lbfgs pf
QuasiGrad.solve_power_flow!(adm, cgd, ctg, flw, grd, idx, lbf, mgd, ntk, prm, qG, scr, stt, sys, upd)

# %% Test 1: benchmark
@time QuasiGrad.solve_linear_pf_with_Gurobi!(idx, ntk, prm, qG, stt, sys)

# %% Test 2: all threads!
@time QuasiGrad.solve_linear_pf_with_Gurobi!(idx, ntk, prm, qG, stt, sys)

# %% Test 3: multi-threading
@time QuasiGrad.solve_linear_pf_with_Gurobi_parallel!(idx, ntk, prm, qG, stt, sys)

# %%
tii = :t1
@time Ybus_real, Ybus_imag = QuasiGrad.update_Ybus(idx, ntk, prm, stt, sys, tii);

# %% ============================
using QuasiGrad
using FLoops

# count threads
Threads.nthreads()
Threads.nthreads()

# how many threads are available?
Sys.CPU_THREADS

# %% now:
function ff(N::Int64)
    #s = zeros(300000)
    s = 0.0
    @floop ThreadedEx(basesize = 300000 ÷ N) for (ii,x) in enumerate(1:300000)
            #s[ii] = sin(x^2)
            @reduce s += sin(x^2)
        end
    return s
end

# %% now:
function f()
    #s = zeros(300000)
    s = 0
    for (ii,x) in enumerate(1:300000)
        #s[ii] = sin(x^2)
        s += sin(x^2)
    end
    return s
end

function f3(N::Int64)
    #s = zeros(300000)
    s = 0.0
    @floop ThreadedEx(basesize = 5 ÷ N) for (ii,x) in enumerate([:t1; :t2; :t3; :t4; :t5])
            #s[ii] = sin(x^2)
            @reduce s += sin(ii^2)
        end
    return s
end

# %%
s_fast = ff(7);
s_slow = f();
# %%
@btime ff(12);
@btime f();
