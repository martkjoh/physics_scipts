# Figures and movie for the 2D Ising data written by `ising.jl`.
#
#     julia plot.jl          reads data/, writes fig/
#
# Requires: CairoMakie, LaTeXStrings, ProgressMeter — see README.md
#     using Pkg; Pkg.add(["CairoMakie", "LaTeXStrings", "ProgressMeter"])
#
# Three ways of showing the critical point:
#   critical.pdf   the four observables vs. T for several lattice sizes
#   scaling.pdf    the χ data collapse — the quantitative finite-size statement
#                  behind "there is a critical point"
#   snapshots.pdf  what the configurations look like below, at, and above T_c
#   ising.mp4      Metropolis dynamics at the same three temperatures

using CairoMakie, DelimitedFiles, Serialization, LaTeXStrings, Printf, ProgressMeter

const TC = 2 / log1p(sqrt(2))
const DATA = joinpath(@__DIR__, "data")
const FIG = joinpath(@__DIR__, "fig")
const SPIN = [RGBf(0.96, 0.96, 0.93), RGBf(0.13, 0.15, 0.22)]   # s = -1, +1

set_theme!(Theme(
    fontsize = 13,
    figure_padding = 7,
    Axis = (xminorticksvisible = true, yminorticksvisible = true,
            xtickalign = 1, ytickalign = 1,
            xminortickalign = 1, yminortickalign = 1,
            xgridvisible = false, ygridvisible = false,
            titlesize = 12, titlefont = :regular, titlealign = :left),
    Lines = (linewidth = 1.6,),
    Scatter = (markersize = 6,),
))

"""Onsager's exact spontaneous magnetisation for the infinite lattice."""
onsager(T) = T < TC ? (1 - sinh(2 / T)^-4)^(1/8) : 0.0

"""Columns of observables.csv, by name."""
const COL = (L = 1, T = 2, e = 3, m = 4, c = 5, χ = 6)

function load()
    raw, _ = readdlm(joinpath(DATA, "observables.csv"), ',', Float64; header = true)
    Ls = sort(unique(Int.(raw[:, COL.L])))
    byL = map(Ls) do L
        r = raw[Int.(raw[:, COL.L]) .== L, :]
        r[sortperm(r[:, COL.T]), :]                 # sort in temperature
    end
    colors = [get(cgrad(:viridis), x) for x in range(0.05, 0.8, length(Ls))]
    return Ls, byL, colors
end

tcline!(ax) = vlines!(ax, TC; color = (:black, 0.4), linestyle = :dash, linewidth = 1)

# ------------------------------------------------- observables vs. T --------

function fig_critical(Ls, byL, colors)
    fig = Figure(size = (760, 560))
    panels = ((COL.m, L"|m|",  "(a)  magnetisation"),
              (COL.χ, L"\chi", "(b)  susceptibility"),
              (COL.c, L"c",    "(c)  specific heat"))

    for (n, (col, lab, title)) in enumerate(panels)
        ax = Axis(fig[(n - 1) ÷ 2 + 1, (n - 1) % 2 + 1];
                  xlabel = L"T", ylabel = lab, title = title)
        tcline!(ax)
        if col == COL.m         # exact infinite-system result for comparison
            T = range(minimum(byL[1][:, COL.T]), TC; length = 300)
            lines!(ax, T, onsager.(T); color = :firebrick, linestyle = :dot,
                   linewidth = 2.2, label = "Onsager")
        end
        for (i, L) in enumerate(Ls)
            scatterlines!(ax, byL[i][:, COL.T], byL[i][:, col];
                          color = colors[i], label = L"L=%$L")
        end
    end

    # three panels on a 2×2 grid: the legend takes the free cell
    # tellwidth/tellheight false: otherwise the legend, being narrow, would
    # shrink the whole second column and squash panel (b)
    Legend(fig[2, 2], content(fig[1, 1]); framevisible = false, merge = true,
           tellwidth = false, tellheight = false,
           halign = :center, valign = :center, patchsize = (24, 12))
    Label(fig[0, 1:2],
          L"\text{2D Ising model: the transition sharpens as }L\text{ grows, at }T_c = 2/\ln(1+\sqrt{2})",
          fontsize = 14)
    rowgap!(fig.layout, 6)
    save(joinpath(FIG, "critical.pdf"), fig)
    # save(joinpath(FIG, "critical.png"), fig; px_per_unit = 3)
end

# --------------------------------------------- finite-size scaling ----------

function fig_scaling(Ls, byL, colors)
    fig = Figure(size = (430, 340))

    # χ = L^{γ/ν} f((T-T_c) L^{1/ν}) with the exact 2D exponents γ/ν = 7/4,
    # ν = 1: the curves collapse onto one scaling function.
    ax = Axis(fig[1, 1]; xlabel = L"(T - T_c)\, L^{1/\nu}", ylabel = L"\chi\, L^{-\gamma/\nu}",
              title = "susceptibility data collapse"
    )
    for (i, L) in enumerate(Ls)
        scatterlines!(ax, (byL[i][:, COL.T] .- TC) .* L, byL[i][:, COL.χ] ./ L^(7/4); 
                      color = colors[i], label = L"L=%$L"
        )
    end
    xlims!(ax, -25, 40)
    axislegend(ax; framevisible = false, position = :rt, rowgap = 0)
    text!(ax, 0.97, 0.62; 
         text = L"\gamma/\nu = 7/4,\; \nu = 1", space = :relative, align = (:right, :top), fontsize = 12
    )
    save(joinpath(FIG, "scaling.pdf"), fig)
    # save(joinpath(FIG, "scaling.png"), fig; px_per_unit = 3)
end

# ------------------------------------------------------- configurations -----

paneltitle(T) = L"T = %$(round(T, digits = 3))\;\; (%$(round(T/TC, digits = 2))\, T_c)"

function spinaxis(fig, pos, title)
    ax = Axis(fig[pos...]; aspect = DataAspect(), title = title, titlealign = :center, titlesize = 13)
    hidedecorations!(ax); hidespines!(ax)
    return ax
end

function fig_snapshots(mov)
    fig = Figure(size = (790, 305))
    for (k, T) in enumerate(mov.T)
        ax = spinaxis(fig, (1, k), paneltitle(T))
        heatmap!(ax, mov.frames[:, :, end, k]; colormap = SPIN, colorrange = (-1, 1), rasterize = 4)
    end
    Label(fig[0, 1:3],
          L"\text{Equilibrium configurations, }L = %$(mov.L)\text{. At }T_c\text{, ordered regions appear on every length scale.}",
          fontsize = 13)
    rowgap!(fig.layout, 3)
    save(joinpath(FIG, "snapshots.pdf"), fig)
    # save(joinpath(FIG, "snapshots.png"), fig; px_per_unit = 3)
end

function make_movie(mov)
    nframes = size(mov.frames, 3)
    fig = Figure(size = (900, 365))
    obs = map(enumerate(mov.T)) do (k, T)
        ax = spinaxis(fig, (1, k), paneltitle(T))
        o = Observable(mov.frames[:, :, 1, k])
        heatmap!(ax, o; colormap = SPIN, colorrange = (-1, 1))
        o
    end
    
    Label(fig[0, :], "Metropolis dynamics, L = $(mov.L)", fontsize = 14)
    rowgap!(fig.layout, 3)
    rowsize!(fig.layout, 1, Aspect(1, 1.0))
    resize_to_layout!(fig)

    prog = Progress(nframes; desc = "recording    ")
    # nframes = 2
    record(fig, joinpath(FIG, "ising.mp4"), 1:nframes; framerate = 25) do f
        for (k, o) in enumerate(obs)
            o[] = mov.frames[:, :, f, k]
        end
        next!(prog)
    end
end

function main()
    mkpath(FIG)
    Ls, byL, colors = load()
    mov = deserialize(joinpath(DATA, "frames.jls"))
    
    # println("critical.pdf");  fig_critical(Ls, byL, colors)
    # println("scaling.pdf");   fig_scaling(Ls, byL, colors)
    # println("snapshots.pdf"); fig_snapshots(mov)

    println("ising.mp4");     make_movie(mov)
    
    println("→ all written to ", FIG)
end

main()
