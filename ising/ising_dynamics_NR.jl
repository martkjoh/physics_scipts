# 2D nonreciprocal Ising model. Records the frames of a movie; plot.jl draws
# it. Two dynamics, following two papers that share the same microscopic model:
#
#   glauber!   single spin-flips     — zero conservation laws
#              Avni et al., PRL 134, 117103 (2025), Eq. (1)-(2);
#              = Blom et al., arXiv:2507.01105, Sect. 3
#   kawasaki!  same-species nearest-neighbour exchanges — M^A and M^B each
#              conserved: the *two* conservation laws of
#              Blom et al., arXiv:2507.01105, Sect. 4, Eqs. (34)-(35)
#
# Two spins per site, σ^A and σ^B = ±1, on an L×L lattice with periodic
# boundaries. Local field and energy of spin i on lattice α:
#
#     h_i^α = J Σ_{j nn of i} σ_j^α + K ε^{αβ} σ_i^β ,   ε^{AB} = +1 = -ε^{BA}
#     E_i^α = -σ_i^α h_i^α
#
# Moves are accepted with the Glauber rate ½[1 - tanh(ΔE/2)], energies in units
# of k_B T. Couplings are passed as J̃ = 4J/(k_B T) and K̃ = K/(k_B T), so
# βJ = J̃/4 and βK = K̃.


using Serialization, ProgressMeter, Dates

const DATA = joinpath(@__DIR__, "data")

@inline up(i, L) = ifelse(i == L, 1, i + 1)
@inline dn(i, L) = ifelse(i == 1, L, i - 1)

"""All spins up: s[i, j, 1] = σ^A, s[i, j, 2] = σ^B."""
lattice(L::Int) = ones(Int8, L, L, 2)
random_lattice(L::Int) = rand((Int8(1), Int8(-1)), L, L, 2)

"""Glauber acceptance ½[1 - tanh(βΔE/2)] = 1/(1 + e^{βΔE}) — Avni Eq. (1), Blom Eqs. (5), (35)."""
@inline glauber_accept(βΔE::Float64) = rand() < 1 / (1 + exp(βΔE))

"""
    glauber!(s, J̃, K̃, nsweeps = 1)

Single spin-flip dynamics, zero conservation laws (Avni et al. Eqs. (1)-(2),
Blom et al. Sect. 3). A spin flips with Glauber probability ½[1 - tanh(βΔE/2)],
where

    βΔE_i^α = 2 σ_i^α (βJ Σ_{j nn} σ_j^α + K̃ ε^{αβ} σ_i^β) .

One sweep is 2L² attempts, site and species drawn at random.
"""
function glauber!(s::Array{Int8,3}, J̃::Float64, K̃::Float64, nsweeps::Int = 1)
    L = size(s, 1)
    βJ = J̃ / 4                                  # 2d = 4 in two dimensions
    @inbounds for _ in 1:(nsweeps * length(s))
        i, j, α = rand(1:L), rand(1:L), rand(1:2)
        nn = Int(s[up(i, L), j, α]) + Int(s[dn(i, L), j, α]) +
             Int(s[i, up(j, L), α]) + Int(s[i, dn(j, L), α])
        ε = ifelse(α == 1, 1, -1)                # ε^{AB} = +1, ε^{BA} = -1
        βΔE = 2 * Int(s[i, j, α]) * (βJ * nn + K̃ * ε * Int(s[i, j, 3 - α]))
        glauber_accept(βΔE) && (s[i, j, α] = -s[i, j, α])
    end
    return s
end

"""
    kawasaki!(s, J̃, K̃, nsweeps = 1)

Intralattice spin-exchange dynamics, two conservation laws (Blom et al.
Sect. 4, Eqs. (34)-(35)): two neighbouring same-species spins exchange values,
so M^A and M^B are conserved separately.

A site and one of its four neighbours are drawn at random and their spins
exchanged with Glauber probability ½[1 - tanh(βΔE/2)], where

    βΔE_ij^α = (σ_i^α - σ_j^α)(βh_i^α - βh_j^α) ,
    βh_k^α   = βJ Σ_{l nn of k, l ≠ partner} σ_l^α + K̃ ε^{αβ} σ_k^β .

Sites i and j are the coordinate pairs `(i, j)` and `(i2, j2)` below, each the
other's exchange partner. Equal spins are skipped.

The `l ≠ partner` excludes the other exchanged site from each neighbour sum,
making βΔE the change in the energy of lattice α at fixed σ^β. This is Penrose
(1991) Eq. (8), not the field of Blom Eq. (2) used in their main text; see
"Choice of exchange energy" in README.md.
"""
function kawasaki!(s::Array{Int8,3}, J̃::Float64, K̃::Float64, nsweeps::Int=1)
    L = size(s, 1)
    βJ = J̃ / 4
    @inbounds for _ in 1:(nsweeps * length(s))
        i, j, α = rand(1:L), rand(1:L), rand(1:2)
        dir = rand(1:4)
        i2, j2 = dir == 1 ? (up(i, L), j) : dir == 2 ? (dn(i, L), j) : dir == 3 ? (i, up(j, L)) : (i, dn(j, L))
        s1, s2 = s[i, j, α], s[i2, j2, α]
        s1 == s2 && continue

        # Four neighbours less the partner: each sum contains the other site's
        # spin exactly once, and the subtraction removes it.
        nn1 = Int(s[up(i, L), j, α]) + Int(s[dn(i, L), j, α]) +
              Int(s[i, up(j, L), α]) + Int(s[i, dn(j, L), α]) - Int(s2)
        nn2 = Int(s[up(i2, L), j2, α]) + Int(s[dn(i2, L), j2, α]) +
              Int(s[i2, up(j2, L), α]) + Int(s[i2, dn(j2, L), α]) - Int(s1)
        ε = ifelse(α == 1, 1, -1)
        h1 = βJ * nn1 + K̃ * ε * Int(s[i, j, 3 - α])
        h2 = βJ * nn2 + K̃ * ε * Int(s[i2, j2, 3 - α])
        βΔE = (Int(s1) - Int(s2)) * (h1 - h2)
        if glauber_accept(βΔE)
            s[i, j, α], s[i2, j2, α] = s2, s1
        end
    end
    return s
end

"""
    write_status(dir, i, nframes, t_start)

Progress report for one run, written to `<dir>/status.txt`, so that threaded
runs report separately rather than to a shared progress bar.
"""
function write_status(dir, i, nframes, t_start)
    progress = i / nframes
    format = "dd-mm-yyyy \nHH:MM:SS"
    stat = "Status simulation:\n\n"
    first = "Simulation started:\n" * Dates.format(t_start, format) * "\n\n"
    last = "Last updated:\n" * Dates.format(now(), format) * "\n\n"
    perc = round(progress * 100; digits = 1)
    bars = 25
    filled = round(Int, progress * bars)
    bar = "Progress: $(perc)%\n[" * rpad("|"^filled, bars) * "]\n\n"
    dt = now() - t_start
    run = "Run time:\n" * string(Dates.canonicalize(dt)) * "\n\n"
    tleft = "∞"
    if i > 0
        tleft_f = round(Dates.value(dt) * (1 / progress - 1); digits = 0)
        tleft = string(Dates.canonicalize(Millisecond(tleft_f)))
    end
    i == nframes && (tleft = "0")
    left = "Expected time left:\n" * tleft * "\n"
    write(joinpath(dir, "status.txt"), stat * first * last * bar * run * left)
end

"""
    movie_frames(L, J̃, K̃; nframes, sweeps_per_frame, dynamics! = glauber!, init = random_lattice, prog = nothing, status_dir = nothing) -> Array{Int8,3}

`nframes` snapshots, `sweeps_per_frame` sweeps apart, from `init(L)`. Nothing
is equilibrated first.

Each frame stores the angle variable θ per site as an index 1:4 — 1 = (+,+),
2 = (+,-), 3 = (-,-), 4 = (-,+) — the quadrants of the (σ^A, σ^B) plane in
cyclic order, which is what plot.jl's cyclic colour code expects.

`dynamics!` is `glauber!` or `kawasaki!`; the latter requires
`init = random_lattice`, since it conserves the magnetisation it starts with.
`prog`, if given, is ticked once per frame. `status_dir`, if given, writes a
`write_status` report there every 0.1% of frames instead.
"""
function movie_frames(L::Int, J̃::Float64, K̃::Float64; nframes::Int, sweeps_per_frame::Int,
                      dynamics!::Function = glauber!, init::Function = random_lattice,
                      prog = nothing, status_dir = nothing)
    s = init(L)
    frames = Array{Int8}(undef, L, L, nframes)
    t_start = now()
    report_every = max(1, nframes ÷ 1000)                # ≈ every .1 percent
    for f in 1:nframes
        dynamics!(s, J̃, K̃, sweeps_per_frame)
        @inbounds for j in 1:L, i in 1:L
            a, b = s[i, j, 1] > 0, s[i, j, 2] > 0
            frames[i, j, f] = a ? (b ? 1 : 2) : (b ? 4 : 3)
        end
        prog === nothing || next!(prog)
        if status_dir !== nothing && (f % report_every == 0 || f == nframes)
            write_status(status_dir, f, nframes, t_start)
        end
    end
    return frames
end

"""
    make_movie(points; L, nframes, sweeps_per_frame, dynamics! = glauber!,
               init = lattice, file = "nr_frames.jls")

One panel per (J̃, K̃) in `points`, serialised to `data/<file>` in the
(L, L, nframes, npanels) layout plot.jl reads, with θ indices as values.
"""
function make_movie(points; L::Int, nframes::Int, sweeps_per_frame::Int,
                    dynamics!::Function=glauber!, init::Function=random_lattice, file::String="nr_frames.jls")
    prog = Progress(length(points) * nframes; desc = "movie frames ")
    frames = cat(
        (movie_frames(L, J̃, K̃; nframes, sweeps_per_frame, dynamics!, init, prog) for (J̃, K̃) in points)...; dims = 4
    )
    serialize(joinpath(DATA, file), (; L, points, sweeps_per_frame, frames))
    println("  → data/$file  ", size(frames))
end

"""
    make_movie_runs(points; L, nframes, sweeps_per_frame, dynamics! = kawasaki!,
                    init = random_lattice, dir)

As `make_movie`, but one (J̃, K̃) per thread (`Threads.@threads`; start Julia
with `--threads=auto`), each serialised to its own
`data/<dir>/<n>/frames.jls` for n = 1:length(points).
"""
function make_movie_runs(points; L::Int, nframes::Int, sweeps_per_frame::Int, dir::String,
                         dynamics!::Function=kawasaki!, init::Function=random_lattice)
    outdir = joinpath(DATA, dir)
    mkpath(outdir)
    Threads.@threads for n in eachindex(points)
        J̃, K̃ = points[n]
        rundir = mkpath(joinpath(outdir, string(n)))
        frames = movie_frames(L, J̃, K̃; nframes, sweeps_per_frame, dynamics!, init, status_dir = rundir)
        serialize(joinpath(rundir, "frames.jls"), (; L, J̃, K̃, sweeps_per_frame, frames))
        println("  → data/$dir/$n/frames.jls  ", size(frames))
    end
end

function main()
    mkpath(DATA)

    points           = [
        # (1.5, 0.), (1.5, 0.3), (1.5, 0.6), (1.5, 0.9),
        (2.0, 0.), (2.0, 0.3), (2.0, 0.6), (2.0, 0.9),
        (2.5, 0.), (2.5, 0.3), (2.5, 0.6), (2.5, 0.9),
        (3.0, 0.), (3.0, 0.3), (3.0, 0.6), (3.0, 0.9),
        (3.5, 0.), (3.5, 0.3), (3.5, 0.6), (3.5, 0.9),
    ]

    # zero conservation laws
    L                = 2^11
    nframes          = 100
    sweeps_per_frame = 50

    # make_movie_runs(points; L, nframes, sweeps_per_frame, dir = "nr_glau",
    #                 (dynamics!) = glauber!, init = lattice)

    # two conservation laws
    L                = 2^8
    nframes          = 500
    sweeps_per_frame = 100_000

    make_movie_runs(points; L, nframes, sweeps_per_frame, dir = "nr_kawa2",
                    (dynamics!) = kawasaki!, init = random_lattice)
end

# test_NR.jl defines ISING_LOAD_ONLY to load the functions without running main().
@isdefined(ISING_LOAD_ONLY) || main()
