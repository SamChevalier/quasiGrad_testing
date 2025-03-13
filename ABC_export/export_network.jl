# Load and Export the base 
using Plots
using Revise
using QuasiGrad

# identify the data
InFile1 = "./data/c3s1_d1_600_scenario_001.json"

# call the jsn data
jsn = QuasiGrad.load_json(InFile1)

# initialize the network 
adm, cgd, ctg, flw, grd, idx, lbf, mgd, ntk, prm, qG, scr, stt, sys, upd = QuasiGrad.base_initialization(jsn, Div=1, hpc_params=false);

function kys(param)
    tuple = fieldnames(typeof(param))
    for el in tuple
        println(el)
    end
end

# %% costs
function compute_dev_cost(nbk, pbk, pcm, cst, p)
    c = 0.0
    for ii in 2:nbk
        if p > pcm[ii]
            c += pbk[ii]*cst[ii]
        else
            c += (p - pcm[ii-1])*cst[ii]
            break
        end
    end

    return c
end

# %% set up a SCOPF problem -- assume all generators are on :)
using JuMP
using Gurobi

# time index
tii  = 1
dt   = prm.ts.duration[tii] # duration
p_lb = zeros(sys.ndev)
p_ub = zeros(sys.ndev)

# upper and lower bounds for loads/gens
for dev in prm.dev.dev_keys
    p_lb[dev] = prm.dev.p_lb[dev][tii]
    p_ub[dev] = prm.dev.p_ub[dev][tii]
end

# what matrices do we need?
ac_b_params = -[prm.acline.b_sr; prm.xfm.b_sr]
Ybs         = QuasiGrad.spdiagm(ac_b_params)
Yflow       = Ybs*ntk.E # flow matrix
Yb          = ntk.Yb    # Ybus matrix
Er          = ntk.E[:,2:end]
ptdf        = Ybs*Er*inv(Matrix(Er'*Ybs*Er))
ptdf        = [zeros(sys.nl + sys.nx) ptdf]

# flow limits
pf_max_base = 0.083*[prm.acline.mva_ub_nom; prm.xfm.mva_ub_nom]

N_d2inj = zeros(sys.nb,sys.ndev)
# loop over consumers (loads)
for bus in 1:sys.nb
    for cs in idx.cs[bus]
        N_d2inj[bus,cs] = -1
    end

    # loop over producers (generators)
    for pr in idx.pr[bus]
        N_d2inj[bus,pr] = +1
    end
end

# %% plot to show the laod and gen cost curves
dev_gen  = 1
cst      = prm.dev.cum_cost_blocks[dev_gen][tii][1]
pbk      = prm.dev.cum_cost_blocks[dev_gen][tii][2]
pcm      = prm.dev.cum_cost_blocks[dev_gen][tii][3]
nbk      = length(pbk)
npts     = 1000
pvec     = collect(LinRange(0, pcm[end], npts))
c        = zeros(npts)

for jj in eachindex(pvec)
    c[jj] = compute_dev_cost(nbk, pbk, pcm, cst, pvec[jj])
end
plot(pvec, c, xlim = (0, 500))


# %% ===================
""" Model 1: pose a simple, flow constrained DCOPF problem"""
model = Model(Gurobi.Optimizer)

# device power bounds!
@variable(model, dev_power[1:sys.ndev])
@constraint(model, p_lb .<= dev_power .<= p_ub)

# phase angle variables
@variable(model, theta[1:sys.nb])
@constraint(model, theta[1] == 0)

pinj = Vector{AffExpr}(undef, sys.nb)
for i in eachindex(pinj)
    pinj[i] = AffExpr(0.0)
end

# get the flows
pflow = Yflow*theta
@constraint(model, -pf_max_base .<= pflow .<= pf_max_base)

# for each device, assign a set of power blocks, each one bounded
dev_power_blocks = [@variable(model, [blk = 1:length(prm.dev.cost[dev][tii])], lower_bound = 0, upper_bound = prm.dev.cum_cost_blocks[dev][tii][2][blk+1]) for dev in 1:sys.ndev]


dev_cost = Vector{AffExpr}(undef, sys.ndev)
for dev in 1:sys.ndev
    dev_cost[dev] = AffExpr(0.0)
    # the device power is the sum of the blocks
    @constraint(model, dev_power[dev] == sum(dev_power_blocks[dev]))

    # now, get the cost!
    cst = prm.dev.cum_cost_blocks[dev][tii][1][2:end]
    if prm.dev.device_type[dev] == "producer"
        # this is a generator
        dev_cost[dev] = -dt*sum(dev_power_blocks[dev].*cst)
    elseif prm.dev.device_type[dev] == "consumer"
        # this is a load
        dev_cost[dev] = +dt*sum(dev_power_blocks[dev].*cst)
    else
        println("device not found!")
    end
end

# loop over consumers (loads)
for bus in 1:sys.nb
    for cs in idx.cs[bus]
        pinj[bus] -= dev_power[cs]
    end

    # loop over producers (generators)
    for pr in idx.pr[bus]
        pinj[bus] += dev_power[pr]
    end
end

# power flow!
@constraint(model, Yb*theta .== pinj)

market_surplus = sum(dev_cost)
@objective(model, Max, market_surplus)

# optimize
optimize!(model)

println("========")
println(objective_value(model))
println("========")

# %% inspect the solution -- flows
t = value.(theta)
f = Yflow*t
plot(f)
plot!(pf_max_base)
plot!(-pf_max_base)

# %% inspect the solution -- device power
plot(value.(dev_power))
plot!(p_lb)
plot!(p_ub)

# %% inspect solution: costs
dev = 4
println(value(dev_cost[dev]))
cst = prm.dev.cum_cost_blocks[dev][tii][1][2:end]
println("========")
println(value.(dev_power_blocks[dev]))
println(-dt*sum(value.(dev_power_blocks[dev]).*cst))
println(value(dev_cost[dev]))


# %% next!
""" Model 2: Use a PTDF formulation, penalize injection imbalance and flows, use 
             mapping matrices to get injections...
"""
model = Model(Gurobi.Optimizer)

# device power bounds!
@variable(model,            dev_power[1:sys.ndev])
@constraint(model, p_lb .<= dev_power .<= p_ub)

# get injections
pinj = N_d2inj * dev_power

# for each device, assign a set of power blocks, each one bounded
dev_power_blocks = [@variable(model, [blk = 1:length(prm.dev.cost[dev][tii])], lower_bound = 0, upper_bound = prm.dev.cum_cost_blocks[dev][tii][2][blk+1]) for dev in 1:sys.ndev]

dev_cost = Vector{AffExpr}(undef, sys.ndev)
for dev in 1:sys.ndev
    dev_cost[dev] = AffExpr(0.0)
    # the device power is the sum of the blocks
    @constraint(model, dev_power[dev] == sum(dev_power_blocks[dev]))

    # now, get the cost!
    cst = prm.dev.cum_cost_blocks[dev][tii][1][2:end]
    if prm.dev.device_type[dev] == "producer"
        # this is a generator
        dev_cost[dev] = -dt*sum(dev_power_blocks[dev].*cst)
    elseif prm.dev.device_type[dev] == "consumer"
        # this is a load
        dev_cost[dev] = dt*sum(dev_power_blocks[dev].*cst)
    else
        println("device not found!")
    end
end

# power flow!
@variable(model, t[1:(sys.nl + sys.nx)], lower_bound = 0.0)
@constraint(model,  ptdf*pinj - pf_max_base  .<= t)
@constraint(model, -ptdf*pinj - pf_max_base  .<= t)

# also penalize power imbalance
@variable(model, tb, lower_bound = 0.0)
@constraint(model,   sum(pinj) .<= tb)
@constraint(model,  -sum(pinj) .<= tb)

market_surplus = sum(dev_cost) - dt*prm.vio.s_flow*sum(t) - dt*prm.vio.p_bus*tb
@objective(model, Max, market_surplus)

# optimize
optimize!(model)

println("========")
println(objective_value(model))
println("========")

# %% === for saving data with HDF5
using HDF5

random_data = randn(100)

fid = h5open("data_file.h5", "w") do file
    write(file, "random_data", random_data)
end

# %% === for reading
using HDF5

fid = h5open("data_file.h5", "r") do file
    global random_data    = read(file, "random_data")
end