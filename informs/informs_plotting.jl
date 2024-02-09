function run_adam_and_plot!(ax::Makie.Axis, fig::Makie.Figure, z_plt::Dict{Symbol, Dict{Symbol, Float64}}, adm::QuasiGrad.Adam, cgd::QuasiGrad.ConstantGrad, ctg::QuasiGrad.Contingency, flw::QuasiGrad.Flow, grd::QuasiGrad.Grad, idx::QuasiGrad.Index, mgd::QuasiGrad.MasterGrad, ntk::QuasiGrad.Network, prm::QuasiGrad.Param, qG::QuasiGrad.QG, scr::Dict{Symbol, Float64}, stt::QuasiGrad.State, sys::QuasiGrad.System, upd::Dict{Symbol, Vector{Vector{Int64}}}, x_lim::Int64; clip_pq_based_on_bins::Bool=false, fp::Bool=false)
    # NOTE -- "clip_pq_based_on_bins = true" is only used once all binaries have been fixed!
    #         so, use in on the very last adam iteration after binaries have been set.
    # 
    # here we go!
    @info "Running adam for $(qG.adam_max_time) seconds!"

    # flush adam just once!
    QuasiGrad.flush_adam!(adm, flw, prm, upd)

    # loop and solve adam twice: once for an initialization, and once for a true run
    qG.skip_ctg_eval = false

    # re-initialize
    if fp == true
        local_step  = 0
        qG.adm_step = 0
    else
        local_step = 0
    end
    qG.beta1_decay   = 1.0
    qG.beta2_decay   = 1.0
    qG.one_min_beta1 = 1.0 - qG.beta1 # here for testing, in case beta1 is changed before a run
    qG.one_min_beta2 = 1.0 - qG.beta2 # here for testing, in case beta2 is changed before a run
    run_adam         = true

    # start the timer!
    adam_start = time()
    plot_freq  = 1

    # loop over adam steps
    while run_adam

        # increment
        qG.adm_step += 1
        local_step  += 1

        if local_step < 125
            plot_freq  = 1
        else
            plot_freq  = 10
        end

        # update limits
        if qG.adm_step > x_lim
            x_lim = x_lim + 150
            Makie.xlims!(ax, [1, x_lim])
            ax.xticks = Int64.(round.(LinRange(0, x_lim, 4)))
        end

        # step decay
        QuasiGrad.adam_step_decay!(qG, time(), adam_start, adam_start+qG.adam_max_time)

        # decay beta and pre-compute
        qG.beta1_decay         = qG.beta1_decay*qG.beta1
        qG.beta2_decay         = qG.beta2_decay*qG.beta2
        qG.one_min_beta1_decay = (1.0-qG.beta1_decay)
        qG.one_min_beta2_decay = (1.0-qG.beta2_decay)

        # update weight parameters?
        if qG.apply_grad_weight_homotopy == true
            QuasiGrad.update_penalties!(prm, qG, time(), adam_start, adam_start+qG.adam_max_time)
        end

        # compute all states and grads
        QuasiGrad.update_states_and_grads!(cgd, ctg, flw, grd, idx, mgd, ntk, prm, qG, scr, stt, sys, clip_pq_based_on_bins=clip_pq_based_on_bins)

        # take an adam step
        QuasiGrad.adam!(adm, mgd, prm, qG, stt, upd)
        GC.safepoint()

        if qG.adm_step % plot_freq == 0
            update_plot!(ax, fig, qG, scr, z_plt, fp, plot_freq)
            if qG.adm_step > 1
                fp = false
            end
            display(fig)
            sleep(1e-10) 
        end

        # stop?
        run_adam = QuasiGrad.adam_termination(adam_start, qG, run_adam, qG.adam_max_time)
    end

    return x_lim
end

function update_plot!(ax::Makie.Axis, fig::Makie.Figure, qG::QuasiGrad.QG, scr::Dict{Symbol, Float64}, z_plt::Dict{Symbol, Dict{Symbol, Float64}}, fp::Bool, plot_freq::Int64)
    #
    # first, set the current values
    z_plt[:now][:zms]  = QuasiGrad.scale_z(scr[:zms])
    z_plt[:now][:pzms] = QuasiGrad.scale_z(scr[:zms_penalized])      
    z_plt[:now][:zhat] = QuasiGrad.scale_z(scr[:zt_penalty] - qG.constraint_grad_weight*scr[:zhat_mxst])
    z_plt[:now][:ctg]  = QuasiGrad.scale_z(scr[:zctg_min] + scr[:zctg_avg])
    z_plt[:now][:emnx] = QuasiGrad.scale_z(scr[:emnx])
    z_plt[:now][:zp]   = QuasiGrad.scale_z(scr[:zp])
    z_plt[:now][:zq]   = QuasiGrad.scale_z(scr[:zq])
    z_plt[:now][:acl]  = QuasiGrad.scale_z(scr[:acl])
    z_plt[:now][:xfm]  = QuasiGrad.scale_z(scr[:xfm])
    z_plt[:now][:zoud] = QuasiGrad.scale_z(scr[:zoud])
    z_plt[:now][:zone] = QuasiGrad.scale_z(scr[:zone])
    z_plt[:now][:rsv]  = QuasiGrad.scale_z(scr[:rsv])
    z_plt[:now][:enpr] = QuasiGrad.scale_z(scr[:enpr])
    z_plt[:now][:encs] = QuasiGrad.scale_z(scr[:encs])
    z_plt[:now][:zsus] = QuasiGrad.scale_z(scr[:zsus])

    # now plot!
    #
    # add an economic dipatch upper bound
    l0  = Makie.lines!(ax, [0, 1e4], [log10(scr[:ed_obj]) - 3.0, log10(scr[:ed_obj]) - 3.0], color = :coral1, linestyle = :dash, linewidth = 7.0)
    l1  = Makie.lines!(ax, [qG.adm_step-plot_freq, qG.adm_step], [z_plt[:prev][:zms],  z_plt[:now][:zms] ], color = :cornflowerblue, linewidth = 10.0)
    l2  = Makie.lines!(ax, [qG.adm_step-plot_freq, qG.adm_step], [z_plt[:prev][:pzms], z_plt[:now][:pzms]], color = :mediumblue,     linewidth = 3.0)
    l3  = Makie.lines!(ax, [qG.adm_step-plot_freq, qG.adm_step], [z_plt[:prev][:zhat], z_plt[:now][:zhat]], color = :goldenrod1, linewidth = 2.0)
    l4  = Makie.lines!(ax, [qG.adm_step-plot_freq, qG.adm_step], [z_plt[:prev][:ctg] , z_plt[:now][:ctg] ], color = :lightslateblue)
    l5  = Makie.lines!(ax, [qG.adm_step-plot_freq, qG.adm_step], [z_plt[:prev][:zp]  , z_plt[:now][:zp]  ], color = :firebrick, linewidth = 3.5)
    l6  = Makie.lines!(ax, [qG.adm_step-plot_freq, qG.adm_step], [z_plt[:prev][:zq]  , z_plt[:now][:zq]  ], color = :salmon1,   linewidth = 2.0)
    l7  = Makie.lines!(ax, [qG.adm_step-plot_freq, qG.adm_step], [z_plt[:prev][:acl] , z_plt[:now][:acl] ], color = :darkorange1, linewidth = 3.5)
    l8  = Makie.lines!(ax, [qG.adm_step-plot_freq, qG.adm_step], [z_plt[:prev][:xfm] , z_plt[:now][:xfm] ], color = :orangered1,  linewidth = 2.0)
    l9  = Makie.lines!(ax, [qG.adm_step-plot_freq, qG.adm_step], [z_plt[:prev][:zoud], z_plt[:now][:zoud]], color = :grey95, linewidth = 3.5, linestyle = :solid)
    l10 = Makie.lines!(ax, [qG.adm_step-plot_freq, qG.adm_step], [z_plt[:prev][:zone], z_plt[:now][:zone]], color = :gray89, linewidth = 3.0, linestyle = :dot)
    l11 = Makie.lines!(ax, [qG.adm_step-plot_freq, qG.adm_step], [z_plt[:prev][:rsv] , z_plt[:now][:rsv] ], color = :gray75, linewidth = 2.5, linestyle = :dash)
    l12 = Makie.lines!(ax, [qG.adm_step-plot_freq, qG.adm_step], [z_plt[:prev][:emnx], z_plt[:now][:emnx]], color = :grey38, linewidth = 2.0, linestyle = :dashdot)
    l13 = Makie.lines!(ax, [qG.adm_step-plot_freq, qG.adm_step], [z_plt[:prev][:zsus], z_plt[:now][:zsus]], color = :grey0,  linewidth = 1.5, linestyle = :dashdotdot)
    l14 = Makie.lines!(ax, [qG.adm_step-plot_freq, qG.adm_step], [z_plt[:prev][:enpr], z_plt[:now][:enpr]], color = :forestgreen, linewidth = 3.5)
    l15 = Makie.lines!(ax, [qG.adm_step-plot_freq, qG.adm_step], [z_plt[:prev][:encs], z_plt[:now][:encs]], color = :darkgreen,   linewidth = 2.0)
    
    if fp == true
        fp = false # toggle

        # define trace lables
        label = Dict(
            :zms  => "market surplus",
            :pzms => "penalized market surplus",       
            :zhat => "constraint penalties", 
            :ctg  => "contingency penalties",
            :zp   => "active power balance",
            :zq   => "reactive power balance",
            :acl  => "acline flow",  
            :xfm  => "xfm flow", 
            :zoud => "on/up/down costs",
            :zone => "zonal reserve penalties",
            :rsv  => "local reserve penalties",
            :enpr => "energy costs (pr)",   
            :encs => "energy revenues (cs)",   
            :emnx => "min/max energy violations",
            :zsus => "start-up state discount",
            :ed   => "economic dispatch (bound)")

        # build legend ==================
        Makie.Legend(fig[1, 2], [l0, l1, l2, l3, l4, l5, l6, l7, l8, l9, l10, l11, l12, l13, l14, l15],
                        [label[:ed],  label[:zms],  label[:pzms], label[:zhat], label[:ctg], label[:zp],   label[:zq],   label[:acl],
                            label[:xfm], label[:zoud], label[:zone], label[:rsv], label[:emnx], label[:zsus], label[:enpr], 
                            label[:encs]],
                            halign = :right, valign = :top, framevisible = false)
    end

    # update the previous values!
    z_plt[:prev][:zms]  = QuasiGrad.scale_z(scr[:zms])
    z_plt[:prev][:pzms] = QuasiGrad.scale_z(scr[:zms_penalized])      
    z_plt[:prev][:zhat] = QuasiGrad.scale_z(scr[:zt_penalty] - qG.constraint_grad_weight*scr[:zhat_mxst])
    z_plt[:prev][:ctg]  = QuasiGrad.scale_z(scr[:zctg_min] + scr[:zctg_avg])
    z_plt[:prev][:emnx] = QuasiGrad.scale_z(scr[:emnx])
    z_plt[:prev][:zp]   = QuasiGrad.scale_z(scr[:zp])
    z_plt[:prev][:zq]   = QuasiGrad.scale_z(scr[:zq])
    z_plt[:prev][:acl]  = QuasiGrad.scale_z(scr[:acl])
    z_plt[:prev][:xfm]  = QuasiGrad.scale_z(scr[:xfm])
    z_plt[:prev][:zoud] = QuasiGrad.scale_z(scr[:zoud])
    z_plt[:prev][:zone] = QuasiGrad.scale_z(scr[:zone])
    z_plt[:prev][:rsv]  = QuasiGrad.scale_z(scr[:rsv])
    z_plt[:prev][:enpr] = QuasiGrad.scale_z(scr[:enpr])
    z_plt[:prev][:encs] = QuasiGrad.scale_z(scr[:encs])
    z_plt[:prev][:zsus] = QuasiGrad.scale_z(scr[:zsus])
end

function initialize_plot(qG::QuasiGrad.QG, scr::Dict{Symbol, Float64}, x_lim::Int64)
    # now, initialize
    fig = Makie.Figure(resolution=(1200, 600), fontsize=26) 
    ax  = Makie.Axis(fig[1, 1], xlabel = "adam iteration", ylabel = "score values (z)", xlabelfont = :bold, ylabelfont = :bold)
    Makie.xlims!(ax, [1, x_lim])
    ax.xticks = Int64.(round.(LinRange(0, x_lim, 4)))

    # set ylims -- this is tricky, since we use "1000" as zero, so the scale goes,
    # -10^4 -10^3 0 10^3 10^4... data_log[:pzms][1]
    min_y     = (-log10(abs(-1e8) + 0.1) - 1.5) + 3.0
    min_y_int = ceil(min_y)

    max_y     = (+log10(scr[:ed_obj]  + 0.1) + 0.25) - 3.0
    max_y_int = floor(max_y)

    # since "1000" is our reference -- see scaling function notes
    y_vec = collect((min_y_int):2:(max_y_int))
    Makie.ylims!(ax, [min_y, max_y])
    tick_name = String[]

    for yv in y_vec
        if yv == 0
            push!(tick_name,"0")
        elseif yv < 0
            push!(tick_name,"-10^"*string(Int(abs(yv - 3.0))))
        else
            push!(tick_name,"+10^"*string(Int(abs(yv + 3.0))))
        end
    end
    ax.yticks = (y_vec, tick_name)
    # => ax.xticks = [2500, 5000, 7500, 10000]
    display(fig)

    # define current and previous dicts
    z_plt = Dict(:prev => Dict(
                            :zms  => 0.0,
                            :pzms => 0.0,       
                            :zhat => 0.0,
                            :ctg  => 0.0,
                            :zp   => 0.0,
                            :zq   => 0.0,
                            :acl  => 0.0,
                            :xfm  => 0.0,
                            :zoud => 0.0,
                            :zone => 0.0,
                            :rsv  => 0.0,
                            :enpr => 0.0,
                            :encs => 0.0,
                            :emnx => 0.0,
                            :zsus => 0.0),
                :now => Dict(
                            :zms  => 0.0,
                            :pzms => 0.0,     
                            :zhat => 0.0,
                            :ctg  => 0.0,
                            :zp   => 0.0,
                            :zq   => 0.0,
                            :acl  => 0.0,
                            :xfm  => 0.0,
                            :zoud => 0.0,
                            :zone => 0.0,
                            :rsv  => 0.0,
                            :enpr => 0.0,
                            :encs => 0.0,
                            :emnx => 0.0,
                            :zsus => 0.0))

    # update the previous values!
    z_plt[:prev][:zms]  = QuasiGrad.scale_z(scr[:zms])
    z_plt[:prev][:pzms] = QuasiGrad.scale_z(scr[:zms_penalized])      
    z_plt[:prev][:zhat] = QuasiGrad.scale_z(scr[:zt_penalty] - qG.constraint_grad_weight*scr[:zhat_mxst])
    z_plt[:prev][:ctg]  = QuasiGrad.scale_z(scr[:zctg_min] + scr[:zctg_avg])
    z_plt[:prev][:emnx] = QuasiGrad.scale_z(scr[:emnx])
    z_plt[:prev][:zp]   = QuasiGrad.scale_z(scr[:zp])
    z_plt[:prev][:zq]   = QuasiGrad.scale_z(scr[:zq])
    z_plt[:prev][:acl]  = QuasiGrad.scale_z(scr[:acl])
    z_plt[:prev][:xfm]  = QuasiGrad.scale_z(scr[:xfm])
    z_plt[:prev][:zoud] = QuasiGrad.scale_z(scr[:zoud])
    z_plt[:prev][:zone] = QuasiGrad.scale_z(scr[:zone])
    z_plt[:prev][:rsv]  = QuasiGrad.scale_z(scr[:rsv])
    z_plt[:prev][:enpr] = QuasiGrad.scale_z(scr[:enpr])
    z_plt[:prev][:encs] = QuasiGrad.scale_z(scr[:encs])
    z_plt[:prev][:zsus] = QuasiGrad.scale_z(scr[:zsus])

    return ax, fig, z_plt
end