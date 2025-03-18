# Load and Export the base 
using Plots
using Revise
using QuasiGrad
using LinearAlgebra

# identify the data
InFile1 = "./data/c3s1_d1_600_scenario_001.json"

# call the jsn data
jsn = QuasiGrad.load_json(InFile1)

# initialize the network 
adm, cgd, ctg, flw, grd, idx, lbf, mgd, ntk, prm, qG, scr, stt, sys, upd = QuasiGrad.base_initialization(jsn, Div=1, hpc_params=false);

# %% === ===========

# choose a randnom set of injections which balance
p_inj_test    = randn(sys.nb)
p_inj_test[1] = -sum(p_inj_test[2:end])

# get the flows
ac_b_params = -[prm.acline.b_sr; prm.xfm.b_sr]
Ybs         = QuasiGrad.spdiagm(ac_b_params)
Yflow       = Ybs*ntk.E # flow matrix
Yb          = ntk.Yb    # Ybus matrix
Ybr         = Yb[2:end,2:end]  # use @view ? 
E           = ntk.E
Er          = E[:,2:end]
Yfr         = Ybs*Er
ptdf        = Ybs*Er*inv(Matrix(Er'*Ybs*Er))
ptdf        = [zeros(sys.nl + sys.nx) ptdf]

# base case flow
pflow_base = ptdf*p_inj_test

# test
theta = Yb[2:end,2:end]\p_inj_test[2:end]
theta = [0; theta]
pflow_base_indirect = Ybs*ntk.E*theta

# all good -- the ptdf works :)
println(norm(pflow_base - pflow_base_indirect))

# %% === ===========
u_k = [zeros(sys.nb-1) for ctg_ii in 1:sys.nctg]
g_k = zeros(sys.nctg)
z_k = [zeros(sys.nac) for ctg_ii in 1:sys.nctg]

for ctg_ii in 1:sys.nctg
    ln_ind          = ntk.ctg_out_ind[ctg_ii][1]
    ac_b_params_ctg = -[prm.acline.b_sr; prm.xfm.b_sr]
    ac_b_params_ctg[ln_ind] = 0

    Ybs_ctg       = QuasiGrad.spdiagm(ac_b_params_ctg)
    Yfr_ctg       = Ybs_ctg*Er # flow matrix

    # apply sparse 
    ei = Array(Er[ln_ind,:])
    
    # compute u, g, and z!
    u_k[ctg_ii]  = Ybr\ei
    g_k[ctg_ii]  = -ac_b_params[ln_ind]/(1.0+(dot(ei,u_k[ctg_ii]))*-ac_b_params[ln_ind])
    mul!(z_k[ctg_ii], Yfr_ctg, u_k[ctg_ii])
end

# %% ok, compute the contingency flows
ctg_err = zeros(sys.nctg)
for ctg_ii in 1:sys.nctg

    pflow_base_copy = copy(pflow_base)
    pflow_base_copy[ntk.ctg_out_ind[ctg_ii][1]] = 0

    # compute updated flows 
    pflow_ctg = pflow_base_copy .- z_k[ctg_ii].*(g_k[ctg_ii]*dot(u_k[ctg_ii], p_inj_test[2:end]))

    # now, set the flow on the removed line to 0
    #pflow_ctg[ntk.ctg_out_ind[ctg_ii][1]] = 0.0

    # compare this to a direct contingency power flow computation

    ac_b_params_ctg = -[prm.acline.b_sr; prm.xfm.b_sr]
    ac_b_params_ctg[ntk.ctg_out_ind[ctg_ii][1]] = 0

    Ybs_ctg         = QuasiGrad.spdiagm(ac_b_params_ctg)
    Yflow_ctg       = Ybs_ctg*E # flow matrix
    Yb_ctg          = E'*Ybs_ctg*E
    ptdf_ctg        = Ybs_ctg*Er*inv(Matrix(Er'*Ybs_ctg*Er))
    ptdf_ctg        = [zeros(sys.nl + sys.nx) ptdf_ctg]

    # base case flow
    pflow_ctg_direct = ptdf_ctg*p_inj_test

    # compare
    ctg_err[ctg_ii] = norm(pflow_ctg_direct - pflow_ctg)
    println(ctg_ii)
end
