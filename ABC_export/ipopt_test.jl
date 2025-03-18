using Plots
using Revise
using QuasiGrad
using LinearAlgebra

# %% =======
using JuMP

model = Model(Ipopt.Optimizer)

@variable(model, x)
@constraint(model, -1 <= x <= 1)
@objective(model, Min, x^2 + x - 1)

optimize!(model)
