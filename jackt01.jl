using Plots
using Revise
using QuasiGrad

# identify the data
InFile1 = "./data/c3s1_d1_600_scenario_001.json"

# call the jsn data
jsn = QuasiGrad.load_json(InFile1)

# initialize the network 
adm, cgd, ctg, flw, grd, idx, lbf, mgd, ntk, prm, qG, scr, stt, sys, upd = QuasiGrad.base_initialization(jsn, Div=1, hpc_params=false);

# solve a single time period power flow with adam
QuasiGrad.economic_dispatch_initialization!(cgd, ctg, flw, grd, idx, mgd, ntk, prm, qG, scr, stt, sys, upd)
stt0 = deepcopy(stt);

# %% ============== runs tests here
#
# reset the state -- this basically resests all variables, so you can run a new test
stt = deepcopy(stt0)

# define the size of the initial steps -- 7 variables
voltage_magnitude_step_t0  = 100*1e-5
voltage_phase_step_t0      = 100*1e-5
transformer_phi_step_t0    = 100*1e-5
transfomer_tau_step_t0     = 100*1e-5
hvdc_line_step_t0          = 100*1e-4
device_power_step_t0       = 100*1e-4
binary_step_t0             = 100*1e-4

# define the size of the final steps -- 7 variables
voltage_magnitude_step_tf  = 5e-10
voltage_phase_step_tf      = 5e-10
transformer_phi_step_tf    = 5e-10
transfomer_tau_step_tf     = 5e-10
hvdc_line_step_tf          = 1e-10
device_power_step_tf       = 1e-10
binary_step_tf             = 1e-10

# build dicts
step_t0_dict = Dict(:vm      => voltage_magnitude_step_t0,
               :va           => voltage_phase_step_t0,    
               :phi          => transformer_phi_step_t0,  
               :tau          => transfomer_tau_step_t0,   
               :dc_pfr       => hvdc_line_step_t0,
               :dc_qto       => hvdc_line_step_t0,
               :dc_qfr       => hvdc_line_step_t0,
               :dev_q        => device_power_step_t0,
               :p_on         => device_power_step_t0,
               :u_step_shunt => binary_step_t0)

step_tf_dict = Dict(:vm      => voltage_magnitude_step_tf,
               :va           => voltage_phase_step_tf,    
               :phi          => transformer_phi_step_tf,  
               :tau          => transfomer_tau_step_tf,   
               :dc_pfr       => hvdc_line_step_tf,
               :dc_qto       => hvdc_line_step_tf,
               :dc_qfr       => hvdc_line_step_tf,
               :dev_q        => device_power_step_tf,
               :p_on         => device_power_step_tf,
               :u_step_shunt => binary_step_tf)

# beta parameters
beta1 = 0.9
beta2 = 0.99

# sigmoid parameters
p1 = 4.0
p2 = 0.6

# run time
adam_max_time = 20.0

QuasiGrad.update_states_and_grads_for_adam_pf!(cgd, grd, idx, mgd, prm, qG, scr, stt, sys; clip_pq_based_on_bins=false)

# run adam
penalties = QuasiGrad.jack_solves_adam_pf!(beta1, beta2, step_t0_dict, step_tf_dict, adam_max_time, p1, p2, adm, cgd, ctg, flw, grd, idx, mgd, ntk, prm, qG, scr, stt, sys, upd)

# plot the progress -- plot the negatives, that we try to drive to 0 :)
plot(-penalties, ylabel = "penalty value", xlabel = "adam step number", yaxis=:log)

# %% === test the current stepping routine
using Plots

t0       = 10.0
tf       = 35.0
tnow     = t0:0.01:tf
alpha_t0 = 10.0   # first step
alpha_tf = 0.001  # last step


x = -1:0.01:1
y = @. -cos(exp(-x)) .+ 1

plot!(x,y)



# %%

tnorm          = @. 2.0*(tnow-t0)/(tf - t0) - 1.0 # scale between -1 and 1
beta           = @. exp(4.0*tnorm)/(0.6 + exp(4.0*tnorm))
log_stp_ratio  = @. log10(alpha_t0/alpha_tf)
alpha_tnow     = @. 10.0 ^ (-beta*log_stp_ratio + log10(alpha_t0))

# Plots.plot(tnow, alpha_tnow, xlabel="time (sec)")
Plots.plot(tnow, alpha_tnow, xlabel="time (sec)", ylabel="step size", yaxis=:log)
