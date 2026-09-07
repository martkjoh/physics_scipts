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
#
# and, if NR_ising_dynamics.jl has been run (data/nr_frames.jls):
#   nr_ising.mp4   the nonreciprocal Ising model, coloured by θ

using CairoMakie, DelimitedFiles, Serialization, LaTeXStrings, Printf, ProgressMeter

const TC = 2 / log1p(sqrt(2))
const DATA = joinpath(@__DIR__, "data")
const FIG = joinpath(@__DIR__, "fig")
const SPIN = [RGBf(0.96, 0.96, 0.93), RGBf(0.13, 0.15, 0.22)]   # s = -1, +1

# The four states of a site of the nonreciprocal model, in the order the swap
# cycles through them: (+,+) → (+,-) → (-,-) → (-,+) → (+,+). θ is an angle, so
# the colours have to close the loop as well — a non-cyclic colormap would put
# a false seam across the lattice and hide the spirals.
const THETA = [RGBf(0.20, 0.35, 0.75), RGBf(0.96, 0.93, 0.82),
               RGBf(0.90, 0.55, 0.15), RGBf(0.45, 0.20, 0.55)]

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
    ax = Axis(fig[pos...]; 
              aspect=DataAspect(), title=title, titlealign=:center, titlesize=13)
    hidedecorations!(ax); hidespines!(ax)
    return ax
end

function fig_snapshots(mov)
    fig = Figure(size = (790, 305))
    for (k, T) in enumerate(mov.T)
        ax = spinaxis(fig, (1, k), paneltitle(T))
        heatmap!(ax, mov.frames[:, :, end, k]; colormap = SPIN, colorrange = (-1, 1), rasterize = 4)
    end
    Label(fig[0, 1:3],"Metropolis dynamics, L = $(mov.L)",fontsize = 13)
    rowgap!(fig.layout, 3)
    save(joinpath(FIG, "snapshots.pdf"), fig)
    # save(joinpath(FIG, "snapshots.png"), fig; px_per_unit = 3)
end

"""
    make_movie(mov; file, colormap, colorrange, titles, label, legend, ncols)

Animate `mov.frames[:, :, f, k]` — one panel per k, one video frame per f,
panels laid out in a `ncols`-wide grid (rows filled in as needed — defaults
to a roughly square grid). The defaults are the equilibrium movie (±1 spins,
one panel per temperature); a different model only has to say what its
values mean, which is what `make_movie_nr` below does. `legend`, if given, is
one label per colour in `colormap`.
"""
function make_movie(mov; file="vid.mp4", legend=nothing, 
                    compression=35, ncols=ceil(Int, sqrt(length(titles))))
    colormap=SPIN
    colorrange=(-1, 1)
    label = "Metropolis dynamics, L = $(mov.L)"
    titles=paneltitle.(mov.T),

    nframes = size(mov.frames, 3)
    npanels = length(titles)
    nrows = ceil(Int, npanels / ncols)

    fig = Figure(size = (220 * ncols + (legend === nothing ? 40 : 180), 220 * nrows + 60))
    obs = map(eachindex(titles)) do k
        row, col = (k - 1) ÷ ncols + 1, (k - 1) % ncols + 1
        ax = spinaxis(fig, (row, col), titles[k])
        o = Observable(mov.frames[:, :, 1, k])
        heatmap!(ax, o; colormap = colormap, colorrange = colorrange)
        o
    end

    legend === nothing || Legend(fig[1:nrows, ncols + 1],
        [PolyElement(color = c) for c in colormap], legend;
        framevisible = false, tellheight = false, patchsize = (14, 14), rowgap = 2)

    Label(fig[0, 1:ncols], label, fontsize = 14)
    rowgap!(fig.layout, 3)
    for r in 1:nrows rowsize!(fig.layout, r, Aspect(1, 1.0)) end
    resize_to_layout!(fig)

    prog = Progress(nframes; desc = "recording    ")
    record(fig, joinpath(FIG, file), 1:nframes; framerate=25, compression=compression) do f
        for (k, o) in enumerate(obs)
            o[] = mov.frames[:, :, f, k]
        end
        next!(prog)
    end
end

"""
Movie of the nonreciprocal Ising model from data/nr_frames.jls, written by
NR_ising_dynamics.jl. Same frame layout as the equilibrium movie, but the
values are θ indices 1:4 — the four (σ^A, σ^B) states — rather than spins, so
this passes the cyclic θ palette and one panel title per (J̃, K̃) point.
"""
function make_movie_nr(mov; file = "nr_ising.mp4", ncols = ceil(Int, sqrt(length(mov.points))))
    make_movie(mov;
        file = file, colormap = THETA, colorrange = (1, 4), ncols = ncols,
        titles = [L"\tilde{J} = %$J,\; \tilde{K} = %$K" for (J, K) in mov.points],
        label = "nonreciprocal Ising model, L = $(mov.L)",
        legend = ["↑↑", "↑↓", "↓↓", "↓↑"])
end

"""
Loads the parallel runs written by `ising_dynamics_NR.jl`'s `make_movie_runs`
— `data/<dir>/<n>/frames.jls` for n = 1:length(points), each holding one
run's own (L, J̃, K̃, sweeps_per_frame, frames) — and stacks them into the same
(L, L, nframes, npanels) layout the other movies use, so the result can be
passed straight to `make_movie_nr`.
"""
function load_runs(dir)
    rundir = joinpath(DATA, dir)
    ns = sort(parse.(Int, readdir(rundir)))
    runs = [deserialize(joinpath(rundir, string(n), "frames.jls")) for n in ns]
    frames = cat((r.frames for r in runs)...; dims = 4)
    return (; L = runs[1].L, points = [(r.J̃, r.K̃) for r in runs], frames)
end

function main()
    # mkpath(FIG)
    # Ls, byL, colors = load()
    # mov = deserialize(joinpath(DATA, "frames.jls"))
    
    # fig_critical(Ls, byL, colors)
    # fig_scaling(Ls, byL, colors)
    # fig_snapshots(mov)

    # name = "nr_glau"
    name = "nr_kawa1"

    mov = load_runs(name)
    make_movie_nr(mov, file=name*".mp4", ncols=8)

end

main()
