# 2D Ising model by Metropolis Monte Carlo — the short version.
#
#     H = -Σ_<ij> s_i s_j ,   s_i = ±1 ,   periodic boundaries,   k_B = J = 1.
#
# Run with all cores:   julia -t auto ising_simple.jl
#
# Requires: ProgressMeter  (`using Pkg; Pkg.add("ProgressMeter")`), see README.md
#
# ---------------------------------------------------------------------------
# Numerical approach
# ---------------------------------------------------------------------------
#
# One update rule only. A spin picked at random is proposed for flipping and
# accepted with probability min(1, exp(-βΔE)), which satisfies detailed
# balance with respect to P(s) ∝ exp(-βH); repeating it samples the Boltzmann
# distribution. Only the four neighbours enter, so
#
#     ΔE = 2 s_i Σ_nn ∈ {-8, -4, 0, 4, 8}
#
# takes five values: the acceptance ratios are tabulated once per call rather
# than calling `exp` per spin, and downhill moves are taken outright without
# drawing a random number. A "sweep" is L² such attempts.
#
# Averages over the sampled configurations give, with N = L²,
#
#     e = <E>/N                              energy density
#     m = <|M|>/N                            magnetisation (|·| because the
#                                            Z2 symmetry is unbroken at finite L)
#     c = β² (<E²> - <E>²) / N               specific heat
#     χ = β  (<M²> - <|M|>²) / N             susceptibility
#
# c and χ come from fluctuations (fluctuation-dissipation), not from numerical
# derivatives. On a finite lattice they peak just above T_c and the peaks drift
# towards it as L grows; that is how the critical point is located from finite
# lattices. The exact answer is T_c = 2/ln(1+√2) ≈ 2.269 (Onsager).
#
# The cost of dropping the cluster algorithm: single-spin flips move the
# lattice a correlation length only after τ ~ L^z sweeps with z ≈ 2.17, so
# near T_c successive configurations stay correlated ("critical slowing
# down") and the error bars there grow steeply with L. That is why this
# script keeps to L ≤ 32 and takes many sweeps, while ising.jl reaches L = 128
# with the Wolff cluster update. Both write the same files, so plot.jl and
# test.jl work with either.
#
# Output (in data/, overwriting what ising.jl may have written):
#     observables.csv   L, T, e, m, c, chi — one row per (L, T)
#     frames.jls        serialised (; L, T, sweeps_per_frame, frames) with
#                       frames::Array{Int8,4} of size (L, L, nframes, nT)

using Base.Threads, DelimitedFiles, Serialization, Printf, ProgressMeter

const TC = 2 / log1p(sqrt(2))            # Onsager critical temperature
const DATA = joinpath(@__DIR__, "data")

# Periodic neighbours, branch-free so the index arithmetic stays in registers.
@inline up(i, L) = ifelse(i == L, 1, i + 1)
@inline dn(i, L) = ifelse(i == 1, L, i - 1)

"""Sum of the four nearest neighbours of site (i, j)."""
@inline function nnsum(s, i, j, L)
    @inbounds Int(s[up(i, L), j]) + Int(s[dn(i, L), j]) + Int(s[i, up(j, L)]) + Int(s[i, dn(j, L)])
end

# Int8 keeps the lattice eight times smaller than Int64, so it stays in cache.
lattice(L) = rand((Int8(-1), Int8(1)), L, L)

"""
    metropolis!(s, β, nsweeps = 1)

`nsweeps` Metropolis sweeps, one sweep being L² attempted single-spin flips.
With ΔE = 2·sh, where sh = s_i · Σ_nn ∈ {-4,-2,0,2,4}, the five acceptance
ratios exp(-2βsh) are tabulated once, and only uphill moves (sh > 0) consume
a random number.

Sites are drawn at random rather than swept in order. Order matters here:
downhill and ΔE = 0 moves are accepted with probability one, so a sequential
sweep is deterministic whenever no site is uphill, and the chain can fall
into a limit cycle it never leaves — on small lattices that really happens
(a 4×4 lattice gets stuck at E = 0). Random sites keep the chain ergodic.
"""
function metropolis!(s::Matrix{Int8}, β::Float64, nsweeps::Int = 1)
    L = size(s, 1)
    w = ntuple(k -> exp(-2β * (2k - 6)), 5)      # index k ↔ sh = 2k - 6
    @inbounds for _ in 1:(nsweeps * length(s))
        i, j = rand(1:L), rand(1:L)
        sh = Int(s[i, j]) * nnsum(s, i, j, L)
        (sh <= 0 || rand() < w[(sh + 6) >> 1]) && (s[i, j] = -s[i, j])
    end
    return s
end

"""
    measure(s) -> (E, M)

Total energy and magnetisation. Only the right and upper bond of each site is
counted, so every bond enters exactly once.
"""
function measure(s::Matrix{Int8})
    L = size(s, 1)
    E = M = 0
    @inbounds for j in 1:L, i in 1:L
        sij = Int(s[i, j])
        M += sij
        E -= sij * (Int(s[up(i, L), j]) + Int(s[i, up(j, L)]))
    end
    return E, M
end

"""
    simulate(L, T) -> (; e, m, c, χ)

Equilibrium averages at temperature `T` on an L×L lattice. The chain starts
from a random (T = ∞) configuration, `nequil` sweeps are discarded, and one
measurement is taken per sweep afterwards.
"""
function simulate(L::Int, T::Float64; nequil::Int = 20_000, nmeas::Int = 200_000)
    β, N = 1 / T, L^2
    s = lattice(L)
    metropolis!(s, β, nequil)
    e1 = e2 = m1 = m2 = 0.0
    for _ in 1:nmeas
        metropolis!(s, β, 1)
        E, M = measure(s)
        e1 += E;       e2 += Float64(E)^2
        m1 += abs(M);  m2 += Float64(M)^2
    end
    e1 /= nmeas; e2 /= nmeas; m1 /= nmeas; m2 /= nmeas
    return (e = e1 / N,
            m = m1 / N,
            c = β^2 * (e2 - e1^2) / N,          # fluctuation-dissipation
            χ = β * (m2 - m1^2) / N)
end

const CHUNK = 500                        # sweeps between progress updates

"""Snapshots of the local dynamics, after `nequil` sweeps of warm-up."""
function movie_frames(L::Int, T::Float64; nframes::Int = 300,
                      sweeps_per_frame::Int = 1, nequil::Int = 1_000_000,
                      prog = nothing)
    β = 1 / T
    s = lattice(L)
    for done in 1:CHUNK:nequil
        n = min(CHUNK, nequil - done + 1)
        metropolis!(s, β, n)
        prog === nothing || next!(prog; step = n)
    end
    frames = Array{Int8}(undef, L, L, nframes)
    for f in 1:nframes
        metropolis!(s, β, sweeps_per_frame)
        @views frames[:, :, f] .= s
        prog === nothing || next!(prog; step = sweeps_per_frame)
    end
    return frames
end

"""
Sweep in temperature. The (L, T) points are independent chains, so they are
spread over the available threads; `rand()` uses Julia's task-local RNG and is
safe to call from each of them.
"""
function sweep_in_T(Ls, Ts)
    jobs = [(L, T) for L in Ls for T in Ts]
    rows = Matrix{Float64}(undef, length(jobs), 6)
    prog = Progress(length(jobs); desc = "sweep in T   ")
    @threads for k in eachindex(jobs)
        L, T = jobs[k]
        o = simulate(L, T)
        rows[k, :] .= (L, T, o.e, o.m, o.c, o.χ)
        next!(prog)
    end
    return rows
end

function run_sweep()
     # Denser temperature grid around T_c, where the observables vary fastest.
    Ts = vcat(
            range(1.60, 2.05; length = 10),
            range(2.10, 2.45; length = 29),
            range(2.50, 3.20; length = 10)
    )
    
    rows = sweep_in_T((8, 16, 32), Ts)
    
    open(joinpath(DATA, "observables.csv"), "w") do io
        println(io, "L,T,e,m,c,chi")
        writedlm(io, rows, ',')
    end
    
    println("  → data/observables.csv  ", size(rows, 1), " rows")
end

function make_movie()
    # Configurations below, at, and above T_c, for the snapshots and the movie.
    # The bar counts sweeps, since the warm-up dominates the time here.
    Lmov, spf, nframes, nequil = 2^8, 1, 300, 50_000
    temps = [0.90TC, TC, 1.10TC]
    prog = Progress(length(temps) * (nequil + nframes * spf); desc = "movie sweeps ")
    frames = cat((movie_frames(Lmov, T; nframes, sweeps_per_frame = spf, nequil, prog) for T in temps)...; dims = 4)
    serialize(joinpath(DATA, "frames.jls"), (; L = Lmov, T = temps, sweeps_per_frame = spf, frames))
    println("  → data/frames.jls  ", size(frames))
end

function main()
    mkpath(DATA)
    nthreads() == 1 && @warn "single threaded — use `julia -t auto ising_simple.jl`"

    # run_sweep()

    make_movie()
end

# Run the simulation when this file is executed (`julia ising_simple.jl`) or sent to the
# REPL by an editor's run-file shortcut, which `include`s it. test.jl defines
# ISING_LOAD_ONLY first, to borrow the functions without starting a long run.
@isdefined(ISING_LOAD_ONLY) || main()
