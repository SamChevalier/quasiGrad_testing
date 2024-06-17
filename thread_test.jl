g = randn(10000000,50)

for i in 1:50
    g[:,i] .= cos.(sin.(g[:,i]).^5)
    println(i)
end

# %% ===
Threads.@threads for i in 1:50
    g[:,i] .= cos.(sin.(g[:,i]).^5)
    println(i)
end