using Makie
using GLMakie
using QuasiGrad

# %% common folder for calling

# call plotting tools 
include("../informs/informs_plotting.jl")

# files -- 1576 system
tfp  = "C:/Users/chev8/Dropbox/Documents/Julia/GO3_testcases/"
path = tfp*"C3E3.1_20230629/D1/C3E3N01576D1/scenario_027.json"

# solve ED
InFile1 = path
jsn = QuasiGrad.load_json(InFile1)

# initialize
adm, cgd, ctg, flw, grd, idx, lbf, mgd, ntk, prm, qG, scr, stt, sys, upd = QuasiGrad.base_initialization(jsn)

# solve ed
QuasiGrad.economic_dispatch_initialization!(cgd, ctg, flw, grd, idx, mgd, ntk, prm, qG, scr, stt, sys, upd)
stt0 = deepcopy(stt);

# %% solve a single time period acpf
stt = deepcopy(stt0);
