using QuasiGrad
using Revise

path    = "C:/Users/Samuel.HORACE/Dropbox (Personal)/Documents/Julia/GO3_testcases/C3S1_20221222/C3S1N01576D1/scenario_001.json"
InFile1 = path
jsn     = QuasiGrad.load_json(InFile1)

# initialize
adm, cgd, ctg, flw, grd, idx, lbf, mgd, ntk, prm, qG, scr, stt, sys, upd = QuasiGrad.base_initialization(jsn, perturb_states=false, pert_size=1.0)

QuasiGrad.economic_dispatch_initialization!(cgd, ctg, flw, grd, idx, mgd, ntk, prm, qG, scr, stt, sys, upd)
QuasiGrad.project!(100.0, idx, prm, qG, stt, sys, upd, final_projection = false)
QuasiGrad.project!(100.0, idx, prm, qG, stt, sys, upd, final_projection = true)

# %%

qG.initial_pf_lbfgs_step = 0.05
QuasiGrad.solve_power_flow!(adm, cgd, ctg, flw, grd, idx, lbf, mgd, ntk, prm, qG, scr, stt, sys, upd)
# ^ grb off
stt0 = deepcopy(stt)

# %% reset ======================
stt = deepcopy(stt0)

# %% solve pf with GRB
QuasiGrad.solve_parallel_linear_pf_with_Gurobi!(flw, grd, idx, ntk, prm, qG, stt, sys)


# %% basically, lower the active power weight, so power moves more

QuasiGrad.update_states_and_grads!(cgd, ctg, flw, grd, idx, mgd, ntk, prm, qG, scr, stt, sys)

scr[:encs]
scr[:enpr]
scr[:zp]  
scr[:zq]  
scr[:acl] 
scr[:xfm] 
