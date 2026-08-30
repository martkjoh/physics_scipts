# Monte Carlo simulation of the 2D Ising model on a square lattice.
#
#     H = -Σ_<ij> s_i s_j ,   s_i = ±1 ,   periodic boundaries,   k_B = J = 1.
#
# Run with all cores:   julia -t auto ising.jl
#
# Requires: ProgressMeter  (`using Pkg; Pkg.add("ProgressMeter")`), see README.md
#
# ---------------------------------------------------------------------------
# Numerical approach
# ---------------------------------------------------------------------------
#
# Configurations are sampled from the Boltzmann distribution P(s) ∝ exp(-βH)
# by a Markov chain built from two complementary update rules.
#
# 1. Metropolis (`metropolis!`).  A spin picked at random is proposed for
#    flipping and accepted with probability min(1, exp(-βΔE)).  Only the four
#    neighbours enter, so ΔE = 2 s_i Σ_nn ∈ {-8,-4,0,4,8} takes five values
#    and the acceptance ratios are tabulated once per sweep instead of calling
#    `exp` per spin.  Downhill moves (ΔE ≤ 0) are always accepted, so no
#    random number is drawn for them — which is also why the sites are visited
#    in random rather than lexicographic order (see `metropolis!`).  This is a
#    *local* update: it mimics physical relaxation dynamics, which makes it the
#    right choice for the movie, but near T_c the correlation time diverges as
#    τ ~ ξ^z with z ≈ 2.17 ("critical slowing down"), so it is a poor sampler
#    there.
#
# 2. Wolff single-cluster (`wolff!`).  A seed spin is picked at random and a
#    cluster is grown over neighbours of equal sign, each bond being added
#    with probability p = 1 - exp(-2β); the whole cluster is then flipped.
#    The construction satisfies detailed balance with acceptance one, and the
#    cluster size tracks the correlation volume, so the algorithm is almost
#    free of critical slowing down (z ≈ 0.25).  Cluster growth uses an
#    explicit stack instead of recursion.
#
# Equilibrium averages use Wolff exclusively.  One cluster contains a variable
# number of spins, so `calibrate` measures the mean cluster size once the chain
# has equilibrated and fixes how many flips make up one "sweep": about one spin
# per site, which is work comparable to a Metropolis sweep and leaves
# successive samples nearly uncorrelated at every temperature (at high T
# clusters are single spins, at low T they span the lattice).  During the
# measurements that count stays fixed — adapting it to the running
# configuration biases the average; see `wolff_sweep!`.
#
# Measured per temperature, from E and M of the whole lattice (N = L²):
#
#     e = <E>/N                              energy density
#     m = <|M|>/N                            magnetisation (|·| because the
#                                            Z2 symmetry is unbroken at finite L)
#     c = β² (<E²> - <E>²) / N               specific heat
#     χ = β  (<M²> - <|M|>²) / N             susceptibility
#
# c and χ follow from the fluctuation-dissipation relations, i.e. they are
# obtained from fluctuations rather than from numerical derivatives.  On a
# finite lattice they peak slightly above T_c, and the peaks sharpen and drift
# towards it as L grows; that drift is how the critical point is located from
# finite lattices.  The exact answer is T_c = 2/ln(1+√2) ≈ 2.269 (Onsager).
#
# Output (in data/):
#     observables.csv   L, T, e, m, c, chi — one row per (L, T)
#     frames.jls        serialised (; L, T, sweeps_per_frame, frames) with
#                       frames::Array{Int8,4} of size (L, L, nframes, nT)

using Base.Threads, DelimitedFiles, Serialization, Printf, ProgressMeter

const TC = 2 / log1p(sqrt(2))            # Onsager critical temperature
const DATA = joinpath(@__DIR__, "data")

# ---------------------------------------------------------------- lattice ---

# Periodic neighbours. `ifelse` is branch-free and the compiler keeps the
# index arithmetic in registers, which is why no wrap-around array is needed.
@inline up(i, L) = ifelse(i == L, 1, i + 1)
@inline dn(i, L) = ifelse(i == 1, L, i - 1)

"""Sum of the four nearest neighbours of site (i, j)."""
@inline function nnsum(s, i, j, L)
    @inbounds Int(s[up(i, L), j]) + Int(s[dn(i, L), j]) +
              Int(s[i, up(j, L)]) + Int(s[i, dn(j, L)])
end

# Int8 keeps the lattice eight times smaller than Int64, so even a 256×256
# lattice (64 kB) stays in cache.
lattice(L) = rand((Int8(-1), Int8(1)), L, L)

# ---------------------------------------------------------------- updates ---

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
    wolff!(s, β, stack) -> cluster size

Grow one Wolff cluster from a random seed, adding aligned neighbours with
probability 1 - exp(-2β), and flip it. Spins are flipped as they are added,
which simultaneously marks them as visited, so no separate bookkeeping array
is needed. `stack` holds linear indices and is reused between calls.
"""
function wolff!(s::Matrix{Int8}, β::Float64, stack::Vector{Int})
    L = size(s, 1)
    p = -expm1(-2β)                              # 1 - exp(-2β), accurate
    i0, j0 = rand(1:L), rand(1:L)
    σ = s[i0, j0]
    s[i0, j0] = -σ
    empty!(stack); push!(stack, i0 + L * (j0 - 1))
    n = 1
    @inbounds while !isempty(stack)
        k = pop!(stack)
        i, j = mod1(k, L), (k - 1) ÷ L + 1
        for (a, b) in ((up(i, L), j), (dn(i, L), j), (i, up(j, L)), (i, dn(j, L)))
            if s[a, b] == σ && rand() < p
                s[a, b] = -σ
                push!(stack, a + L * (b - 1))
                n += 1
            end
        end
    end
    return n
end

"""`n` Wolff cluster flips."""
function wolff!(s, β, stack, n::Int)
    for _ in 1:n
        wolff!(s, β, stack)
    end
    return s
end

"""
Flip clusters until ~one spin per site has been touched.

The number of clusters depends on the configuration, so this may only be used
to *equilibrate*, never between measurements: it is a state-dependent stopping
rule, and ordered configurations — which produce large clusters — end the loop
sooner and would end up over-represented. Use `calibrate` + `wolff!(…, n)` for
the measurement phase.
"""
function wolff_sweep!(s, β, stack)
    n = 0
    while n < length(s)
        n += wolff!(s, β, stack)
    end
    return n
end

"""
    calibrate(s, β, stack) -> nclust

How many cluster flips touch about one spin per site, estimated from the mean
cluster size of a short burn-in. Call this on an *equilibrated* configuration
and then keep the returned count fixed for every measurement, so that the
sampling effort carries no information about the state being measured.
"""
function calibrate(s, β, stack; nburn::Int = 500)
    tot = 0
    for _ in 1:nburn
        tot += wolff!(s, β, stack)
    end
    return clamp(round(Int, nburn * length(s) / tot), 1, length(s))
end

# ------------------------------------------------------------ measurement ---

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

Equilibrium averages at temperature `T` on an L×L lattice, sampled with Wolff
updates. The chain starts from a random (T = ∞) configuration, `nequil`
sweeps are discarded, and one measurement is taken per sweep afterwards.
"""
function simulate(L::Int, T::Float64; nequil::Int = 1_000, nmeas::Int = 10_000)
    β, N = 1 / T, L^2
    s, stack = lattice(L), Int[]
    for _ in 1:nequil                       # adaptive here: nothing is measured
        wolff_sweep!(s, β, stack)
    end
    nclust = calibrate(s, β, stack)         # fixed from here on
    e1 = e2 = m1 = m2 = 0.0
    for _ in 1:nmeas
        wolff!(s, β, stack, nclust)
        E, M = measure(s)
        e1 += E;       e2 += Float64(E)^2
        m1 += abs(M);  m2 += Float64(M)^2
    end
    e1 /= nmeas; e2 /= nmeas; m1 /= nmeas; m2 /= nmeas
    return (e = e1 / N, m = m1 / N, c = β^2 * (e2 - e1^2) / N, χ = β * (m2 - m1^2) / N)
end

"""
    movie_frames(L, T) -> Array{Int8,3}

Equilibrate quickly with Wolff, then record `nframes` snapshots of *local*
Metropolis dynamics: cluster flips are non-local and would look unphysical.
"""
function movie_frames(L::Int, T::Float64; nframes::Int = 300,
                      sweeps_per_frame::Int = 1, nequil::Int = 200, prog = nothing)
    β = 1 / T
    s, stack = lattice(L), Int[]
    for _ in 1:nequil
        wolff_sweep!(s, β, stack)
    end
    frames = Array{Int8}(undef, L, L, nframes)
    for f in 1:nframes
        metropolis!(s, β, sweeps_per_frame)
        @views frames[:, :, f] .= s
        prog === nothing || next!(prog)
    end
    return frames
end

# ------------------------------------------------------------------- runs ---

"""
Finite-size sweep through the transition. The (L, T) points are independent
chains, so they are simply spread over the available threads; `rand()` uses
Julia's task-local RNG and is safe to call from each of them.
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
    rows = sweep_in_T((16, 32, 64, 128), Ts)
    open(joinpath(DATA, "observables.csv"), "w") do io
        println(io, "L,T,e,m,c,chi")
        writedlm(io, rows, ',')
    end
    println("  → data/observables.csv  ", size(rows, 1), " rows")
end


function make_movie()
    
    # Configurations below, at, and above T_c, for the snapshots and the movie.
    Lmov, spf, nframes = 2^10, 1, 50
    temps = [0.90TC, 1.00TC, 1.10TC]
    prog = Progress(length(temps) * nframes; desc = "movie frames ")
    frames = cat((movie_frames(Lmov, T; nframes, sweeps_per_frame = spf, prog) for T in temps)...; dims = 4)
    serialize(joinpath(DATA, "frames.jls"), (; L = Lmov, T = temps, sweeps_per_frame = spf, frames))
    println("  → data/frames.jls  ", size(frames))
end

function main()
    mkpath(DATA)
    nthreads() == 1 && @warn "single threaded — use `julia -t auto ising.jl`"

    # run_sweep()
    make_movie()
end

# Run the simulation when this file is executed (`julia ising.jl`) or sent to the
# REPL by an editor's run-file shortcut, which `include`s it. test.jl defines
# ISING_LOAD_ONLY first, to borrow the functions without starting a long run.
@isdefined(ISING_LOAD_ONLY) || main()
