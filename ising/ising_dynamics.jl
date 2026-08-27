# 2D Ising model with a choice of dynamics — Metropolis, Wolff cluster, or
# Kawasaki spin-exchange — selectable independently for equilibration and for
# sampling. Combines ising.jl and ising_simple.jl into one script.
#
#     H = -Σ_<ij> s_i s_j ,   s_i = ±1 ,   periodic boundaries,   k_B = J = 1.
#
# Run with all cores:   julia -t auto ising_dynamics.jl
#
# Requires: ProgressMeter  (`using Pkg; Pkg.add("ProgressMeter")`), see README.md
#
# ---------------------------------------------------------------------------
# The three dynamics
# ---------------------------------------------------------------------------
#
# 1. Metropolis (`metropolis!`). A random spin is flipped with probability
#    min(1, exp(-βΔE)). Local, does not conserve magnetisation — the right
#    choice for a movie of a magnet relaxing, but near T_c the correlation
#    time diverges as τ ~ ξ^z, z ≈ 2.17 ("critical slowing down").
#
# 2. Wolff single-cluster (`wolff!`). Grows a cluster of aligned spins from a
#    random seed, bond by bond with probability 1 - exp(-2β), and flips it
#    whole. Non-local; almost free of critical slowing down (z ≈ 0.25) and so
#    the right choice for equilibrium averages near T_c, but a poor movie —
#    cluster flips look like teleporting jumps, not physical relaxation.
#
# 3. Kawasaki spin-exchange (`kawasaki!`). A random nearest-neighbour pair of
#    unlike spins is swapped with probability min(1, exp(-βΔE)). Unlike the
#    other two, a swap conserves total magnetisation exactly, so this is the
#    dynamics of a conserved order parameter (phase separation in a lattice
#    gas / binary alloy), not of a magnet — <|M|> stays pinned near its
#    initial value forever, so `m` and `χ` are not meaningful under Kawasaki
#    sampling. It is, however, the right dynamics for a coarsening movie:
#    domains of like spins grow out of a random start with no field needed.
#
# `equil`/`sample`/`dyn` below take :metropolis, :wolff or :kawasaki, and
# equilibration and sampling may use different ones (e.g. Wolff to reach
# equilibrium fast, Metropolis to sample it, since only the state at the end
# of equilibration is handed to the sampler). A single "sweep" always means
# ~L² spin updates regardless of dynamics, so `nequil`/`nmeas`/`nframes` are
# comparable time steps across all three; see `sweep!` and `advance!`.
#
# Measured per temperature, from E and M of the whole lattice (N = L²):
#
#     e = <E>/N                              energy density
#     m = <|M|>/N                            magnetisation (|·| because the
#                                            Z2 symmetry is unbroken at finite L)
#     c = β² (<E²> - <E>²) / N               specific heat
#     χ = β  (<M²> - <|M|>²) / N             susceptibility
#
# c and χ follow from the fluctuation-dissipation relations. On a finite
# lattice they peak slightly above T_c, and the peaks sharpen and drift
# towards it as L grows. Exact answer: T_c = 2/ln(1+√2) ≈ 2.269 (Onsager).
#
# Output (in data/, shared with ising.jl / ising_simple.jl):
#     observables.csv   L, T, e, m, c, chi — one row per (L, T)
#     frames.jls        serialised (; L, T, sweeps_per_frame, dyn, frames) with
#                       frames::Array{Int8,4} of size (L, L, nframes, nT)

using Base.Threads, DelimitedFiles, Serialization, Printf, ProgressMeter

const TC = 2 / log1p(sqrt(2))            # Onsager critical temperature
const DATA = joinpath(@__DIR__, "data")

# ---------------------------------------------------------------- lattice ---

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

# function lattice(L)
#     s = Matrix{Int8}(undef, L, L)
#     s[:, 1:L÷2] .= 1
#     s[:, L÷2+1:end] .= -1
#     return s
# end

# function lattice(L)
#     s = Matrix{Int8}(undef, L, L)
#     s[:, :] .= 1
#     return s
# end

# ---------------------------------------------------------------- updates ---

"""
    metropolis!(s, β, nsweeps = 1)

`nsweeps` Metropolis sweeps, one sweep being L² attempted single-spin flips.
ΔE = 2·sh, sh = s_i · Σ_nn ∈ {-4,-2,0,2,4}, so the five acceptance ratios are
tabulated once and only uphill moves (sh > 0) consume a random number. Sites
are drawn at random rather than swept in order, which keeps the chain ergodic
(a sequential sweep can get stuck in a limit cycle on small lattices).
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
    kawasaki!(s, β, nsweeps = 1)

`nsweeps` Kawasaki sweeps, one sweep being L² attempted nearest-neighbour spin
exchanges: a random site and a random one of its four neighbours are picked,
and if the two spins differ (equal spins are a zero-ΔE no-op, skipped without
drawing a random number) the pair is swapped with probability min(1,
exp(-βΔE)). Swapping two unlike spins (a, -a) leaves their mutual bond
unchanged, so ΔE only involves each site's other three neighbours; written in
terms of the full 4-neighbour sums it is ΔE = 2s₁(nnsum₁ - nnsum₂) + 4 (the +4
undoes each site double-counting its own future partner).

Conserves total magnetisation exactly — see the module docstring.
"""
function kawasaki!(s::Matrix{Int8}, β::Float64, nsweeps::Int = 1)
    L = size(s, 1)
    @inbounds for _ in 1:(nsweeps * length(s))
        i, j = rand(1:L), rand(1:L)
        d = rand(1:4)
        i2, j2 = d == 1 ? (up(i, L), j) : d == 2 ? (dn(i, L), j) :
                 d == 3 ? (i, up(j, L)) : (i, dn(j, L))
        s1, s2 = s[i, j], s[i2, j2]
        s1 == s2 && continue
        ΔE = 2s1 * (nnsum(s, i, j, L) - nnsum(s, i2, j2, L)) + 4
        if ΔE <= 0 || rand() < exp(-β * ΔE)
            s[i, j], s[i2, j2] = s2, s1
        end
    end
    return s
end

"""
    wolff!(s, β, stack) -> cluster size

Grow one Wolff cluster from a random seed, adding aligned neighbours with
probability 1 - exp(-2β), and flip it. Spins are flipped as they are added,
which simultaneously marks them as visited. `stack` holds linear indices and
is reused between calls.
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
Flip clusters until ~one spin per site has been touched — the Wolff analogue
of one sweep. State-dependent stopping rule: use only to *equilibrate*, never
between measurements (see `calibrate`).
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

How many cluster flips make up ~one sweep, estimated from the mean cluster
size of a short burn-in. Call on an *equilibrated* configuration and keep the
returned count fixed for every measurement, so sampling effort carries no
information about the state being measured.
"""
function calibrate(s, β, stack; nburn::Int = 500)
    tot = 0
    for _ in 1:nburn
        tot += wolff!(s, β, stack)
    end
    return clamp(round(Int, nburn * length(s) / tot), 1, length(s))
end

# ------------------------------------------------------- dynamics switch ---

"""
One sweep (~L² spin updates) of `dyn` ∈ (:metropolis, :kawasaki, :wolff), the
common time unit used for equilibration regardless of which dynamics is used.
"""
function sweep!(dyn::Symbol, s, β, stack)
    dyn === :metropolis ? metropolis!(s, β, 1) :
    dyn === :kawasaki   ? kawasaki!(s, β, 1) :
    dyn === :wolff      ? (wolff_sweep!(s, β, stack); s) :
    error("unknown dynamics $dyn (use :metropolis, :kawasaki or :wolff)")
end

"""`n` further updates of `dyn`: sweeps for :metropolis/:kawasaki, cluster flips for :wolff."""
function advance!(dyn::Symbol, s, β, stack, n)
    dyn === :metropolis ? metropolis!(s, β, n) :
    dyn === :kawasaki   ? kawasaki!(s, β, n) :
    dyn === :wolff      ? wolff!(s, β, stack, n) :
    error("unknown dynamics $dyn (use :metropolis, :kawasaki or :wolff)")
end

"""
Per-measurement work for `dyn`, fixed once before sampling starts (1 sweep for
:metropolis/:kawasaki; a calibrated cluster count for :wolff — see `calibrate`).
"""
samplerate(dyn::Symbol, s, β, stack) = dyn === :wolff ? calibrate(s, β, stack) : 1

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
    simulate(L, T; equil = :wolff, sample = :wolff, nequil = 1_000, nmeas = 10_000) -> (; e, m, c, χ)

Equilibrium averages at temperature `T` on an L×L lattice. `equil` drives
`nequil` warm-up sweeps (discarded), `sample` drives `nmeas` measurements
(one per fixed unit of work, see `samplerate`). Any of :metropolis, :kawasaki,
:wolff may be used for either phase, and the two may differ. Sampling with
`sample = :kawasaki` conserves the magnetisation set by equilibration, so `m`
and `χ` are then not meaningful (see the module docstring).
"""
function simulate(L::Int, T::Float64; 
                   equil::Symbol = :wolff, sample::Symbol = :wolff,
                   nequil::Int = 1_000, nmeas::Int = 10_000)
    β, N = 1 / T, L^2
    s, stack = lattice(L), Int[]
    for _ in 1:nequil
        sweep!(equil, s, β, stack)
    end
    rate = samplerate(sample, s, β, stack)
    e1 = e2 = m1 = m2 = 0.0
    for _ in 1:nmeas
        advance!(sample, s, β, stack, rate)
        E, M = measure(s)
        e1 += E;       e2 += Float64(E)^2
        m1 += abs(M);  m2 += Float64(M)^2
    end
    e1 /= nmeas; e2 /= nmeas; m1 /= nmeas; m2 /= nmeas
    return (e = e1 / N, m = m1 / N, c = β^2 * (e2 - e1^2) / N, χ = β * (m2 - m1^2) / N)
end

"""
    movie_frames(L, T; dyn = :metropolis, equil = :wolff, nframes = 300, sweeps_per_frame = 1, nequil = 200) -> Array{Int8,3}

Equilibrate with `equil` (`nequil` sweeps, discarded), then record `nframes`
snapshots of `dyn` dynamics, `sweeps_per_frame` sweeps (or cluster flips, for
:wolff) apart. :metropolis or :kawasaki give a physically meaningful movie —
:wolff cluster flips are non-local and look unphysical frame to frame.
"""
function movie_frames(L::Int, T::Float64; 
                      dyn::Symbol = :metropolis, equil::Symbol = :wolff, prog = nothing,
                      nframes::Int = 300, sweeps_per_frame::Int = 1, nequil::Int = 200)
    β = 1 / T
    s, stack = lattice(L), Int[]
    for _ in 1:nequil
        sweep!(equil, s, β, stack)
    end
    frames = Array{Int8}(undef, L, L, nframes)
    for f in 1:nframes
        advance!(dyn, s, β, stack, sweeps_per_frame)
        @views frames[:, :, f] .= s
        prog === nothing || next!(prog)
    end
    return frames
end

# ------------------------------------------------------------------- runs ---

"""
Finite-size sweep through the transition. The (L, T) points are independent
chains, so they are simply spread over the available threads.
"""
function sweep_in_T(Ls, Ts; 
                    equil::Symbol = :wolff, sample::Symbol = :wolff,
                    nequil::Int = 1_000, nmeas::Int = 10_000)
    jobs = [(L, T) for L in Ls for T in Ts]
    rows = Matrix{Float64}(undef, length(jobs), 6)
    prog = Progress(length(jobs); desc = "sweep in T   ")
    @threads for k in eachindex(jobs)
        L, T = jobs[k]
        o = simulate(L, T; equil, sample, nequil, nmeas)
        rows[k, :] .= (L, T, o.e, o.m, o.c, o.χ)
        next!(prog)
    end
    return rows
end

function run_sweep(equil::Symbol = :wolff, sample::Symbol = :wolff;
                   Ls = (16, 32, 64, 128), nequil::Int = 1_000, nmeas::Int = 10_000)
    # Denser temperature grid around T_c, where the observables vary fastest.
    Ts = vcat(
        range(1.60, 2.05; length = 10),
        range(2.10, 2.45; length = 29),
        range(2.50, 3.20; length = 10)
    )
    rows = sweep_in_T(Ls, Ts; equil, sample, nequil, nmeas)
    open(joinpath(DATA, "observables.csv"), "w") do io
        println(io, "L,T,e,m,c,chi")
        writedlm(io, rows, ',')
    end
    println("  → data/observables.csv  ", size(rows, 1), " rows")
end

function make_movie(dyn::Symbol = :metropolis; equil::Symbol = :wolff,
                    L::Int = 2^7, sweeps_per_frame::Int = 1, nframes::Int = 300, nequil::Int = 100)
    temps = [0.90TC, 1.00TC, 1.10TC]
    # temps = [1.01TC, 1.02TC, 1.03TC, 1.04TC,]
    prog = Progress(length(temps) * nframes; desc = "movie frames ")
    frames = cat((movie_frames(L, T; dyn, equil, nframes, sweeps_per_frame, nequil, prog) for T in temps)...; dims = 4)
    serialize(joinpath(DATA, "frames.jls"), (; L, T = temps, sweeps_per_frame, dyn, frames))
    println("  → data/frames.jls  ", size(frames))
end

function main()
    mkpath(DATA)
    nthreads() == 1 && @warn "single threaded — use `julia -t auto ising_dynamics.jl`"

    # run_sweep(:wolff, :wolff)
    
    make_movie(:metropolis; L = 2^10, nframes=500, nequil = 1000)
    
    # make_movie(:kawasaki; L = 2^9, nframes=50)
    # make_movie(:kawasaki; equil = :kawasaki, L = 2^7, nequil=1_000_000)
end

# Run the simulation when this file is executed (`julia ising_dynamics.jl`) or sent to
# the REPL by an editor's run-file shortcut, which `include`s it. test.jl-style scripts
# can define ISING_LOAD_ONLY first, to borrow the functions without starting a long run.
@isdefined(ISING_LOAD_ONLY) || main()
