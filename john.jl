#Starter Tasks
#Sin Cos Product Plots
using Plots
function sincosproduct()
    #x = range(-10*pi, 10*pi, length=6283) #done in length, not steps
    xrange = -10pi:0.01:10pi #done in steps of 0.01
    prod = sin.(xrange) .* cos.(xrange)
    display(Plots.plot(xrange, prod, label="sin(x) * cos(x)", title="Plot of product of sin and cos", xlabel="x", ylabel="sin(x) * cos(x)"))
end


# %% 

#-----------------------------------------------------------------------------------
#Linear System Solution
using Random
using LinearAlgebra #imports the backslash command that solves linear equations
function linsystem()
    A = randn(10,10) #10x10 matrix of random elements from the stnd normal distribution (mean 0, variance 1)
    b = randn(10) #same as A, but it's a 10x1 matrix, or an array
    lin = A \ b #backslash operator solves linear system
    display(scatter(1:10, lin, label="Solution x", title="Solution of Ax = b"))
end
#-------------------------------------------------------------------------------------
#Gradient Solution
using ForwardDiff #takes ForwardDiff from JuliaDiff package, Forward finds gradient function and results of the function
function gradsolution()
    f(x::AbstractVector) = x[1]^2 #function to minimize, treats x as a vector
    gradfunc = x -> ForwardDiff.gradient(f, [x])[1] #sets the gradient function, sends x as a 1-element array, and then pulls out the one element of the gradient after
    function grad_descent(gradfunc, x0, s, max) #sets iteration function to find the optimized solution
        x = x0
        for i in 1:max
            x -= s * gradfunc(x)
        end
        return x
    end
    x0 = 10 #starting point
    s = 0.01 #step size
    max = 1000 #maximum num of iterations
    opt = grad_descent(gradfunc, x0, s, max) #finds optimized solution using function
    println("Optimized solution x = $opt")
end
sincosproduct()
linsystem()
gradsolution()

# %% 
sincosproduct()

# %% 
function sum_of_numbers(v::Vector{Float64})
    s = sum(v)
    return s
end

# %% ==========
s = sum_of_numbers(Float32.(rand(5000)))

# %% 
