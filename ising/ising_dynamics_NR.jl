# 2D nonreciprocal Ising model — Avni et al., PRL 134, 117103 (2025).
# Metropolis dynamics only. Records the frames of a movie; plot.jl draws it.
#
# Two spins per site, σ^A and σ^B = ±1, on an L×L lattice with periodic
# boundaries. Each spin minimises its own *selfish* energy
#
#     E_i^α = -J Σ_{j nn of i} σ_i^α σ_j^α - K ε^{αβ} σ_i^α σ_i^β ,
#
# with ε^{AB} = +1 = -ε^{BA}: within a species the coupling is the ordinary
# ferromagnetic one, but across species A wants to align with B while B wants
# to anti-align with A. There is no single Hamiltonian the two of them share,
# so detailed balance is broken and the model can end up in a time-dependent
# "swap" state where both magnetisations flip over and over.
#
# The flip of one spin changes only its own selfish energy, by
#
#     ΔE_i^α = 2 σ_i^α (J Σ_{j nn} σ_j^α + K ε^{αβ} σ_i^β) ,
#
# and is accepted with the Metropolis probability min(1, exp(-ΔE_i^α/(k_B T))).
# (The paper uses the Glauber rule instead; it reports no qualitative
# difference between update rules.)
#
# Couplings are the paper's dimensionless ones,
#
#     J̃ = 2dJ/(k_B T) = 4J/(k_B T)  in d = 2 ,     K̃ = K/(k_B T) ,
#
# so βJ = J̃/4 and βK = K̃, and the temperature never appears on its own — the
# phase diagrams of the paper are drawn in exactly these (J̃, K̃). Setting
# K̃ = 0 decouples the species and leaves two copies of the equilibrium Ising
# model, whose transition sits at J̃_c = 4/T_c = 2/log(1+√2) ≈ 1.763.
#
# Note that in 2D the swap phase is a finite-size effect: the paper finds that
# spiral defects destroy it as L grows, and that static order is destroyed by
# growing droplets, so at large L only disorder survives.
#
# Run:  julia ising_dynamics_NR.jl    then   julia plot.jl

using Serialization, ProgressMeter

const DATA = joinpath(@__DIR__, "data")

@inline up(i, L) = ifelse(i == L, 1, i + 1)
@inline dn(i, L) = ifelse(i == 1, L, i - 1)

"""All spins up: s[i, j, 1] = σ^A, s[i, j, 2] = σ^B."""
lattice(L::Int) = ones(Int8, L, L, 2)

"""
    metropolis!(s, J̃, K̃, nsweeps = 1)

`nsweeps` Metropolis sweeps, one sweep being 2L² attempted single-spin flips —
one per spin in the lattice, both species counted. Site and species are drawn
at random rather than swept in order, which keeps the chain from falling into
a limit cycle of its own on small lattices.
"""
function metropolis!(s::Array{Int8,3}, J̃::Float64, K̃::Float64, nsweeps::Int = 1)
    L = size(s, 1)
    βJ = J̃ / 4                                  # 2d = 4 in two dimensions
    @inbounds for _ in 1:(nsweeps * length(s))
        i, j, α = rand(1:L), rand(1:L), rand(1:2)
        nn = Int(s[up(i, L), j, α]) + Int(s[dn(i, L), j, α]) +
             Int(s[i, up(j, L), α]) + Int(s[i, dn(j, L), α])
        ε = ifelse(α == 1, 1, -1)                # ε^{AB} = +1, ε^{BA} = -1
        βΔE = 2 * Int(s[i, j, α]) * (βJ * nn + K̃ * ε * Int(s[i, j, 3 - α]))
        (βΔE <= 0 || rand() < exp(-βΔE)) && (s[i, j, α] = -s[i, j, α])
    end
    return s
end

"""
    movie_frames(L, J̃, K̃; nframes, sweeps_per_frame, prog = nothing) -> Array{Int8,3}

`nframes` snapshots, `sweeps_per_frame` sweeps apart, from the ordered start.
Nothing is equilibrated away first: the droplets and spirals that eat the
initial order are exactly what the movie is about.

A frame stores the angle variable θ of each site as an index 1:4 — 1 = (+,+),
2 = (+,-), 3 = (-,-), 4 = (-,+) — the four quadrants of the angle in the
(σ^A, σ^B) plane taken in order, so that the index is cyclic like θ itself.
That is what plot.jl's cyclic colour code expects: it turns a swap into a
colour cycle and shows the spiral defects as the points where all four colours
meet (Fig. 2 of the paper).

`prog`, if given, is ticked once per frame — one bar spans every panel, so it
belongs to `make_movie` rather than being created here.
"""
function movie_frames(L::Int, J̃::Float64, K̃::Float64;
                      nframes::Int, sweeps_per_frame::Int, prog = nothing)
    s = lattice(L)
    frames = Array{Int8}(undef, L, L, nframes)
    for f in 1:nframes
        metropolis!(s, J̃, K̃, sweeps_per_frame)
        @inbounds for j in 1:L, i in 1:L
            a, b = s[i, j, 1] > 0, s[i, j, 2] > 0
            frames[i, j, f] = a ? (b ? 1 : 2) : (b ? 4 : 3)
        end
        prog === nothing || next!(prog)
    end
    return frames
end

"""
    make_movie(points; L, nframes, sweeps_per_frame)

One panel per (J̃, K̃) in `points`, serialised for plot.jl in the same
(L, L, nframes, npanels) layout as the equilibrium movie — plot.jl only needs
to be told that the values are θ indices rather than spins.

None of the arguments has a default: every knob lives in `main`, so changing
one there is the only thing that decides what gets recorded.
"""
function make_movie(points; L::Int, nframes::Int, sweeps_per_frame::Int)
    prog = Progress(length(points) * nframes; desc = "movie frames ")
    frames = cat((movie_frames(L, J̃, K̃; nframes, sweeps_per_frame, prog)
                  for (J̃, K̃) in points)...; dims = 4)
    serialize(joinpath(DATA, "nr_frames.jls"), (; L, points, sweeps_per_frame, frames))
    println("  → data/nr_frames.jls  ", size(frames))
end

function main()
    mkpath(DATA)

    L                = 2^10
    nframes          = 500
    sweeps_per_frame = 10
    points           = [(1.6, 0.3), (2.0, 0.3), (2.8, 0.3)]

    make_movie(points; L, nframes, sweeps_per_frame)
end

main()
