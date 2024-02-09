using QuasiGrad
using Revise

# load the json
path = "C:/Users/Samuel.HORACE/Dropbox (Personal)/Documents/Julia/GO3_testcases/C3S0_20221208/D2/C3S0N00073/scenario_002.json"

# call
jsn = QuasiGrad.load_json(path)

# %% init
adm, cgd, ctg, flw, grd, idx, lbf, mgd, ntk, prm, qG, scr, stt, sys, upd = QuasiGrad.base_initialization(jsn, false, 1.0);

# solve
QuasiGrad.update_states_and_grads!(cgd, ctg, flw, grd, idx, mgd, ntk, prm, qG, scr, stt, sys)

# run an ED
ED = QuasiGrad.solve_economic_dispatch(GRB, idx, prm, qG, scr, stt, sys, upd);
QuasiGrad.apply_economic_dispatch_projection!(ED, idx, prm, qG, stt, sys);

# recompute the state
qG.eval_grad = false
QuasiGrad.update_states_and_grads!(cgd, ctg, flw, grd, idx, mgd, ntk, prm, qG, scr, stt, sys)
qG.eval_grad = true

# ===== new score?
QuasiGrad.dcpf_initialization!(flw, idx, ntk, prm, qG, stt, sys)
QuasiGrad.update_states_and_grads!(cgd, ctg, flw, grd, idx, mgd, ntk, prm, qG, scr, stt, sys)

# intialize lbfgs
dpf0, pf_lbfgs, pf_lbfgs_diff, pf_lbfgs_idx, pf_lbfgs_map, pf_lbfgs_step, zpf = QuasiGrad.initialize_pf_lbfgs(mgd, prm, stt, sys, upd);

# %% score
QuasiGrad.update_states_and_grads_for_solve_pf_lbfgs!(cgd, grd, idx, lbf, mgd, prm, qG, stt, sys)
zp1 = sum(sum([zpf[:zp][tii] for tii in prm.ts.time_keys]))
zq1 = sum(sum([zpf[:zq][tii] for tii in prm.ts.time_keys]))

# correct
QuasiGrad.correct_reactive_injections!(idx::QuasiGrad.Index, prm::QuasiGrad.Param, qG::QuasiGrad.QG, stt::QuasiGrad.State, sys::QuasiGrad.System)

# rescore :)
QuasiGrad.update_states_and_grads_for_solve_pf_lbfgs!(dpf0, grd, idx, mgd, prm, qG, stt, sys, zpf)
zp2 = sum([zpf[:zp][tii] for tii in prm.ts.time_keys])
zq2 = sum([zpf[:zq][tii] for tii in prm.ts.time_keys])

# %% ============
QuasiGrad.dcvm_initialization!(flw, idx, ntk, prm, qG, stt, sys)


# %% solve pf
#qG.scale_c_pbus_testing = 1e-4
#qG.scale_c_qbus_testing = 1e-4
#qG.cdist_psolve = 1e3

#prm.vio.p_bus   = 1e3
#prm.vio.q_bus   = 1e3

# loop -- lbfgs
for ii in 1:1500
    # take an lbfgs step 
    QuasiGrad.solve_pf_lbfgs!(pf_lbfgs, pf_lbfgs_diff, pf_lbfgs_idx, pf_lbfgs_map, pf_lbfgs_step, mgd, prm, qG, stt, upd, zpf)                                                                              

    # save zpf BEFORE updating with the new state
    for tii in prm.ts.time_keys
        pf_lbfgs_step[:zpf_prev][tii] = (zpf[:zp][tii]+zpf[:zq][tii]) 
    end

    # compute all states and grads
    QuasiGrad.update_states_and_grads_for_solve_pf_lbfgs!(cgd, grd, idx, lbf, mgd, prm, qG, stt, sys)

    # print
    zp = round(sum(sum([zpf[:zp][tii] for tii in prm.ts.time_keys])); sigdigits = 3)
    zq = round(sum(sum([zpf[:zq][tii] for tii in prm.ts.time_keys])); sigdigits = 3)
    stp = sum(pf_lbfgs_step[:step][tii] for tii in prm.ts.time_keys)/sys.nT
    println("P penalty is $(zp), Q penalty is $(zq) and average step is $(stp)!")

    #println(stt.vm[:t2][3])
end

# %% write solution
QuasiGrad.solve_Gurobi_projection!(idx, prm, qG, stt, sys, upd)
QuasiGrad.apply_Gurobi_projection!(idx, prm, qG, stt, sys)

# one last clip + state computation -- no grad needed!
qG.eval_grad = false
QuasiGrad.update_states_and_grads!(cgd, ctg, flw, grd, idx, mgd, ntk, prm, qG, scr, stt, sys)

# write the final solution
soln_dict = QuasiGrad.prepare_solution(prm, stt, sys)
QuasiGrad.write_solution("solution.jl", qG, soln_dict, scr)

# %% ================== 
# sum(sum([qG.cdist_psolve*(stt.p_on[tii] - dpf0[:p_on][tii]).^2 for tii in prm.ts.time_keys]))
# z = sum(sum([zpf[:zq][tii] for tii in prm.ts.time_keys]))

