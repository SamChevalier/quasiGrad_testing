using quasiGrad

# %% identify the data
InFile1 = "./data/scenario_027.json"

# call the jsn data
jsn = quasiGrad.load_json(InFile1)

# initialize the network 
adm, cgd, ctg, flw, grd, idx, lbf, mgd, ntk, prm, qG, scr, stt, sys, upd = quasiGrad.base_initialization(jsn, Div=1, hpc_params=false);

# solve a single time period power flow with adam
quasiGrad.economic_dispatch_initialization!(cgd, ctg, flw, grd, idx, mgd, ntk, prm, qG, scr, stt, sys, upd)
stt0 = deepcopy(stt);

# %% ============== runs tests here
#
# reset the state
stt = deepcopy(stt0)
qG.adam_max_time = 60.0

# choose adam step sizes for power flow (initial)
vm_pf_t0      = 1e-6
va_pf_t0      = 1e-6
phi_pf_t0     = 1e-6
tau_pf_t0     = 1e-6
dc_pf_t0      = 1e-4
power_pf_t0   = 1e-4
bin_pf_t0     = 1e-4 # bullish!!!

qG.alpha_pf_t0[:vm]     = vm_pf_t0
qG.alpha_pf_t0[:va]     = va_pf_t0
# xfm
qG.alpha_pf_t0[:phi]    = phi_pf_t0
qG.alpha_pf_t0[:tau]    = tau_pf_t0
# dc
qG.alpha_pf_t0[:dc_pfr] = dc_pf_t0
qG.alpha_pf_t0[:dc_qto] = dc_pf_t0
qG.alpha_pf_t0[:dc_qfr] = dc_pf_t0
# powers
qG.alpha_pf_t0[:dev_q]  = power_pf_t0
qG.alpha_pf_t0[:p_on]   = power_pf_t0/10.0 # downscale active power!!!!
# bins
qG.alpha_pf_t0[:u_step_shunt] = bin_pf_t0

# choose adam step sizes for power flow (final)
vm_pf_tf    = 5e-7
va_pf_tf    = 5e-7
phi_pf_tf   = 5e-7
tau_pf_tf   = 5e-7
dc_pf_tf    = 1e-7
power_pf_tf = 1e-7
bin_pf_tf   = 1e-7 # bullish!!!

qG.alpha_pf_tf[:vm]     = vm_pf_tf
qG.alpha_pf_tf[:va]     = va_pf_tf
# xfm
qG.alpha_pf_tf[:phi]    = phi_pf_tf
qG.alpha_pf_tf[:tau]    = tau_pf_tf
# dc
qG.alpha_pf_tf[:dc_pfr] = dc_pf_tf
qG.alpha_pf_tf[:dc_qto] = dc_pf_tf
qG.alpha_pf_tf[:dc_qfr] = dc_pf_tf
# powers
qG.alpha_pf_tf[:dev_q]  = power_pf_tf
qG.alpha_pf_tf[:p_on]   = power_pf_tf/10.0 # downscale active power!!!!
# bins
qG.alpha_pf_tf[:u_step_shunt] = bin_pf_tf

quasiGrad.jack_solves_adam_pf!(adm, cgd, ctg, flw, grd, idx, mgd, ntk, prm, qG, scr, stt, sys, upd; first_solve = false)

# %% === test the current stepping routine
using Plots

t0       = 10.0
tf       = 35.0
tnow     = t0:0.01:tf
alpha_t0 = 10.0   # first step
alpha_tf = 0.001  # last step

tnorm          = @. 2.0*(tnow-t0)/(tf - t0) - 1.0 # scale between -1 and 1
beta           = @. exp(4.0*tnorm)/(0.6 + exp(4.0*tnorm))
log_stp_ratio  = @. log10(alpha_t0/alpha_tf)
alpha_tnow     = @. 10.0 ^ (-beta*log_stp_ratio + log10(alpha_t0))

# Plots.plot(tnow, alpha_tnow, xlabel="time (sec)")
Plots.plot(tnow, alpha_tnow, xlabel="time (sec)", ylabel="step size", yaxis=:log)


