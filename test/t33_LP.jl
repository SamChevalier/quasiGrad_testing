using QuasiGrad
using Revise

# files
path = "C:/Users/chev8/Dropbox/Documents/Julia/GO3_testcases/C3S1.1_20230807/D1/C3S1N01576D1/scenario_001.json"
path = "C:/Users/chev8/Dropbox/Documents/Julia/GO3_testcases/C3S1.1_20230807/D1/C3S1N04200D1/scenario_001.json"

path = "C:/Users/chev8/Dropbox/Documents/Julia/GO3_testcases/C3S1.1_20230807/D1/C3S1N06049D1/scenario_001.json"

jsn  = QuasiGrad.load_json(path)

# initialize
adm, cgd, ctg, flw, grd, idx, lbf, mgd, ntk, prm, qG, scr, stt, sys, upd = QuasiGrad.base_initialization(jsn, perturb_states=true);

# ================
qG.num_threads = 10
QuasiGrad.update_states_and_grads!(cgd, ctg, flw, grd, idx, mgd, ntk, prm, qG, scr, stt, sys)

# %% run copper plate ED
qG.num_threads = 10
QuasiGrad.economic_dispatch_initialization!(cgd, ctg, flw, grd, idx, mgd, ntk, prm, qG, scr, stt, sys, upd, include_sus=true)

# %% ===

QuasiGrad.solve_economic_dispatch!(idx, prm, qG, scr, stt, sys, upd; include_sus_in_ed=true)
