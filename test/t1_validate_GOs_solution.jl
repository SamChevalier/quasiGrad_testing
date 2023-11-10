# load a valid solution and test it with my constraints :)
include("../src/quasiGrad_dual.jl")

# file -- 3 bus
data_dir  = "./test/data/c3/C3S0_20221208/D1/C3S0N00003/"
file_name = "scenario_003.json"
soln      = "scenario_003_solution.json"

# file -- 73 bus
#data_dir  = "./test/data/c3/C3S0_20221208/D3/C3S0N00073/"
#file_name = "scenario_002.json"
#soln      = "scenario_002_solution.json"

# read and parse the input data
jsn, prm, idx, sys = quasiGrad.load_and_parse_json(data_dir*file_name);
qG                 = quasiGrad.initialize_qG(prm)
qG.eval_grad     = false

# initialize states -- @btime
cgd, GRB, grd, mgd, scr, stt = quasiGrad.initialize_states(idx, prm, sys);

# %% -- load the solution
jsn_soln = quasiGrad.load_json(data_dir*soln)

# loop and populate the stt
#
# ac_line
ac_lines = jsn_soln["time_series_output"]["ac_line"]
for line in ac_lines
    id       = line["uid"]
    line_ind = findfirst(x -> x == id, prm.acline.id)
    for tii in prm.ts.time_keys
        stt.u_on_acline[tii][line_ind] = line["on_status"][tii]
    end
end

# shunt
shunts = jsn_soln["time_series_output"]["shunt"]
for sh in shunts
    id     = sh["uid"]
    sh_ind = findfirst(x -> x == id, prm.shunt.id)
    for tii in prm.ts.time_keys
        stt.u_step_shunt[tii][sh_ind] = sh["step"][tii]
    end
end

# xfm
xfms = jsn_soln["time_series_output"]["two_winding_transformer"]
for xfm in xfms
    id      = xfm["uid"]
    xfm_ind = findfirst(x -> x == id, prm.xfm.id )
    for tii in prm.ts.time_keys
        stt.tau[tii][xfm_ind]      = xfm["tm"][tii]
        stt.phi[tii][xfm_ind]      = xfm["ta"][tii]
        stt.u_on_xfm[tii][xfm_ind] = xfm["on_status"][tii]
    end
end

# bus
buses = jsn_soln["time_series_output"]["bus"]
for bus in buses
    id      = bus["uid"]
    bus_ind = findfirst(x -> x == id, prm.bus.id)
    for tii in prm.ts.time_keys
        stt.vm[tii][bus_ind] = bus["vm"][tii]
        stt.va[tii][bus_ind] = bus["va"][tii]
    end
end

# dc
dclines = jsn_soln["time_series_output"]["dc_line"]
for line in dclines
    id      = line["uid"]
    line_ind = findfirst(x -> x == id, prm.dc.id)
    for tii in prm.ts.time_keys
        stt.dc_pfr[tii][line_ind] = line["pdc_fr"][tii]
        stt.dc_pto[tii][line_ind] = line["pdc_fr"][tii]
        stt.dc_qfr[tii][line_ind] = line["qdc_fr"][tii]
        stt.dc_qto[tii][line_ind] = line["qdc_to"][tii]
    end
end

# devices
devices = jsn_soln["time_series_output"]["simple_dispatchable_device"]
for device in devices
    id       = device["uid"]
    dev_ind = findfirst(x -> x == id, prm.dev.id)
    for tii in prm.ts.time_keys
        stt.u_on_dev[tii][dev_ind]  = device["on_status"][tii]
        stt.p_on[tii][dev_ind]      = device["p_on"][tii]
        stt.dev_q[tii][dev_ind]     = device["q"][tii]
        stt.p_rgu[tii][dev_ind]     = device["p_reg_res_up"][tii]
        stt.p_rgd[tii][dev_ind]     = device["p_reg_res_down"][tii]
        stt.p_scr[tii][dev_ind]     = device["p_syn_res"][tii]
        stt.p_nsc[tii][dev_ind]     = device["p_nsyn_res"][tii]
        stt.p_rru_on[tii][dev_ind]  = device["p_ramp_res_up_online"][tii]
        stt.p_rrd_on[tii][dev_ind]  = device["p_ramp_res_down_online"][tii]
        stt.p_rru_off[tii][dev_ind] = device["p_ramp_res_up_offline"][tii]
        stt.p_rrd_off[tii][dev_ind] = device["p_ramp_res_down_offline"][tii]
        stt.q_qru[tii][dev_ind]     = device["q_res_up"][tii]
        stt.q_qrd[tii][dev_ind]     = device["q_res_down"][tii]
    end
end

# %% now, test (hard) constraint violations!
include("../src/quasiGrad_dual.jl")
sys = quasiGrad.build_sys(jsn)

quasiGrad.acline_flows!(grd, idx, prm, qG, stt, sys)
quasiGrad.xfm_flows!(grd, idx, prm, qG, stt, sys)
quasiGrad.shunts!(grd, idx, prm, qG, stt)
quasiGrad.all_device_statuses_and_costs!(grd, prm, qG, stt)
quasiGrad.device_startup_states!(prm, idx, stt, grd, qG, sys)
quasiGrad.device_active_powers!(idx, prm, qG, stt, sys)
quasiGrad.device_reactive_powers!(idx, prm, qG, stt)
quasiGrad.energy_costs!(grd, prm, qG, stt, sys)
quasiGrad.energy_penalties!(grd, idx, prm, qG, scr, stt, sys)
quasiGrad.penalized_device_constraints!(prm, idx, stt, grd, qG, sys)

# %% issues!
println(maximum(maximum.(values(stt.zhat_mndn))))
println(maximum(maximum.(values(stt.zhat_mnup))))
println(maximum(maximum.(values(stt.zhat_rup))))
println(maximum(maximum.(values(stt.zhat_rd))))
println(maximum(maximum.(values(stt.zhat_rgu))))
println(maximum(maximum.(values(stt.zhat_rgd))))
println(maximum(maximum.(values(stt.zhat_scr))))
println(maximum(maximum.(values(stt.zhat_nsc))))
println(maximum(maximum.(values(stt.zhat_rruon))))
println(maximum(maximum.(values(stt.zhat_rruoff))))
println(maximum(maximum.(values(stt.zhat_rrdon))))
println(maximum(maximum.(values(stt.zhat_rrdoff))))
println(maximum(maximum.(values(stt.zhat_pmax))))
println(maximum(maximum.(values(stt.zhat_pmin))))
println(maximum(maximum.(values(stt.zhat_pmaxoff))))
println(maximum(maximum.(values(stt.zhat_qmax))))
println(maximum(maximum.(values(stt.zhat_qmin))))
println(maximum(maximum.(values(stt.zhat_qmax_beta))))
println(maximum(maximum.(values(stt.zhat_qmin_beta))))

# %% Let's pass this solution to Gurobi and see if it find a zero objective!
quasiGrad.Gurobi_projection!(prm, idx, stt, grd, qG, sys, vst, GRB)

