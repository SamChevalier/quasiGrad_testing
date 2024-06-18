using Makie
using GLMakie
using QuasiGrad

# call plotting tools 
include("./informs/informs_plotting.jl")

# %% identify the data
InFile1 = "./data/c3s1_d1_600_scenario_001.json"

# call the jsn data
jsn = QuasiGrad.load_json(InFile1)

# initialize the network 
adm, cgd, ctg, flw, grd, idx, lbf, mgd, ntk, prm, qG, scr, stt, sys, upd = QuasiGrad.base_initialization(jsn, Div=1, hpc_params=false);

# solve a single time period power flow with adam
QuasiGrad.economic_dispatch_initialization!(cgd, ctg, flw, grd, idx, mgd, ntk, prm, qG, scr, stt, sys, upd)
stt0 = deepcopy(stt);

# %% 1. call stt0 and solve
stt = deepcopy(stt0);

# write locally
qG.write_location   = "local"
qG.eval_grad        = true
qG.always_solve_ctg = true
qG.skip_ctg_eval    = false

# turn off all printing
qG.print_zms                     = false # print zms at every adam iteration?
qG.print_final_stats             = false # print stats at the end?
qG.print_lbfgs_iterations        = false
qG.print_projection_success      = false
qG.print_linear_pf_iterations    = false
qG.print_reserve_cleanup_success = false

# ===============
vm_t0      = 2.5e-5
va_t0      = 2.5e-5
phi_t0     = 2.5e-5
tau_t0     = 2.5e-5
dc_t0      = 1e-2
power_t0   = 1e-2
reserve_t0 = 1e-2
bin_t0     = 1e-2 # bullish!!!
qG.alpha_t0 = Dict(
               :vm     => vm_t0,
               :va     => va_t0,
               # xfm
               :phi    => phi_t0,
               :tau    => tau_t0,
               # dc
               :dc_pfr => dc_t0,
               :dc_qto => dc_t0,
               :dc_qfr => dc_t0,
               # powers
               :dev_q  => power_t0,
               :p_on   => power_t0,
               # reserves
               :p_rgu     => reserve_t0,
               :p_rgd     => reserve_t0,
               :p_scr     => reserve_t0,
               :p_nsc     => reserve_t0,
               :p_rrd_on  => reserve_t0,
               :p_rrd_off => reserve_t0,
               :p_rru_on  => reserve_t0,
               :p_rru_off => reserve_t0,
               :q_qrd     => reserve_t0,
               :q_qru     => reserve_t0,
               # bins
               :u_on_xfm     => bin_t0,
               :u_on_dev     => bin_t0,
               :u_step_shunt => bin_t0,
               :u_on_acline  => bin_t0)
vm_tf      = 5e-7 # 2e-6#
va_tf      = 5e-7 # 2e-6#
phi_tf     = 5e-7 # 2e-6#
tau_tf     = 5e-7 # 2e-6#
dc_tf      = 1e-4 # 1e-3#
power_tf   = 1e-4 # 1e-3#
reserve_tf = 1e-4 # 1e-3#
bin_tf     = 1e-4 # 1e-3#
qG.alpha_tf = Dict(
                :vm     => vm_tf,
                :va     => va_tf,
                # xfm
                :phi    => phi_tf,
                :tau    => tau_tf,
                # dc
                :dc_pfr => dc_tf,
                :dc_qto => dc_tf,
                :dc_qfr => dc_tf,
                # powers
                :dev_q  => power_tf,
                :p_on   => power_tf,
                # reserves
                :p_rgu     => reserve_tf,
                :p_rgd     => reserve_tf,
                :p_scr     => reserve_tf,
                :p_nsc     => reserve_tf,
                :p_rrd_on  => reserve_tf,
                :p_rrd_off => reserve_tf,
                :p_rru_on  => reserve_tf,
                :p_rru_off => reserve_tf,
                :q_qrd     => reserve_tf,
                :q_qru     => reserve_tf,
                # bins
                :u_on_xfm     => bin_tf,
                :u_on_dev     => bin_tf,
                :u_step_shunt => bin_tf,
                :u_on_acline  => bin_tf)

# plot initial solutions!
qG.print_zms     = true
stt              = deepcopy(stt0);
x_lim            = 100
ax, fig, z_plt   = initialize_plot(qG, scr, x_lim)
qG.skip_ctg_eval = false

# stt = deepcopy(stt0);
qG.adam_max_time = 100.0
x_lim = run_adam_and_plot!(ax, fig, z_plt, adm, cgd, ctg, flw, grd, idx, mgd, ntk, prm, qG, scr, stt, sys, upd, x_lim, fp=true)
