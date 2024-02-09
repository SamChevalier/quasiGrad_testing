using QuasiGrad
using Revise

# files
path = "C:/Users/Samuel.HORACE/Dropbox (Personal)/Documents/Julia/GO3_testcases/C3S1_20221222/C3S1N01576D1/scenario_001.json"
jsn  = QuasiGrad.load_json(path)

# initialize
adm, cgd, ctg, flw, grd, idx, lbf, mgd, ntk, prm, qG, scr, stt, sys, upd = QuasiGrad.base_initialization(jsn, perturb_states=true);

# ================
qG.num_threads = 10
QuasiGrad.update_states_and_grads!(cgd, ctg, flw, grd, idx, mgd, ntk, prm, qG, scr, stt, sys)

# %% run copper plate ED
qG.num_threads = 10
@btime QuasiGrad.economic_dispatch_initialization!(cgd, ctg, flw, grd, idx, mgd, ntk, prm, qG, scr, stt, sys, upd, include_sus=true)
