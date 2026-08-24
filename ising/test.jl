# Correctness tests for ising.jl, against results that are known exactly.
#
#     julia -t auto test.jl
#
# Requires: ProgressMeter (loaded via ising.jl), see README.md
#
# 1. Exact enumeration.  A 4×4 lattice has only 2^16 configurations, so the
#    Boltzmann averages can be summed directly.  Both samplers must reproduce
#    them: this tests detailed balance and the observables at once.
# 2. Onsager's exact energy and magnetisation of the infinite lattice, against
#    the largest simulated L, away from T_c where finite-size effects are small.
# 3. The critical temperature approached by the susceptibility peaks.

using Test, DelimitedFiles, Printf
const ISING_LOAD_ONLY = true            # take ising.jl's functions, not its main()
include(joinpath(@__DIR__, "ising.jl"))

# ------------------------------------------------------- exact references ---

"""Boltzmann averages of a 4×4 lattice by summing all 2^16 configurations."""
function enumerate_4x4(T)
    L = 4; N = L^2; β = 1 / T
    s = Matrix{Int8}(undef, L, L)
    Z = e1 = e2 = m1 = m2 = 0.0
    for bits in 0:(2^N - 1)
        for k in 0:(N - 1)
            s[k + 1] = Int8(2 * ((bits >> k) & 1) - 1)
        end
        E, M = measure(s)
        w = exp(-β * E)
        Z += w
        e1 += w * E;       e2 += w * Float64(E)^2
        m1 += w * abs(M);  m2 += w * Float64(M)^2
    end
    e1 /= Z; e2 /= Z; m1 /= Z; m2 /= Z
    return (e = e1 / N, m = m1 / N,
            c = β^2 * (e2 - e1^2) / N, χ = β * (m2 - m1^2) / N)
end

"""Complete elliptic integral of the first kind, modulus k, by the AGM."""
function ellipk(k)
    a, b = 1.0, sqrt(1 - k^2)
    for _ in 1:50
        a, b = (a + b) / 2, sqrt(a * b)
    end
    return π / (2a)
end

"""Onsager's exact energy per site of the infinite lattice (singular at T_c)."""
function onsager_energy(T)
    β = 1 / T
    k = 2 * sinh(2β) / cosh(2β)^2
    return -coth(2β) * (1 + (2 / π) * (2 * tanh(2β)^2 - 1) * ellipk(k))
end

onsager_m(T) = T < TC ? (1 - sinh(2 / T)^-4)^(1/8) : 0.0

"""Metropolis-only sampler, used to check it against Wolff and exact results."""
function simulate_metropolis(L, T; nequil = 5_000, nmeas = 200_000)
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
    return (e = e1 / N, m = m1 / N,
            c = β^2 * (e2 - e1^2) / N, χ = β * (m2 - m1^2) / N)
end

# ------------------------------------------------------------------ tests ---

@testset "2D Ising" begin

@testset "4×4 exact enumeration" begin
    for T in (1.8, TC, 3.0)
        ex = enumerate_4x4(T)
        wo = simulate(4, T; nequil = 2_000, nmeas = 200_000)
        me = simulate_metropolis(4, T)
        @printf("  T=%.3f  e: exact %+.5f  wolff %+.5f  metro %+.5f\n", T, ex.e, wo.e, me.e)
        @printf("          m: exact %.5f  wolff %.5f  metro %.5f\n", ex.m, wo.m, me.m)
        @printf("          χ: exact %.5f  wolff %.5f  metro %.5f\n", ex.χ, wo.χ, me.χ)
        for (sim, name) in ((wo, "wolff"), (me, "metropolis"))
            @test isapprox(sim.e, ex.e; atol = 5e-3)
            @test isapprox(sim.m, ex.m; atol = 5e-3)
            @test isapprox(sim.c, ex.c; atol = 3e-2)
            @test isapprox(sim.χ, ex.χ; atol = 3e-2)
        end
    end
end

@testset "Wolff cluster construction" begin
    # At β → 0 no bond is ever added: every cluster is a single spin.
    s, stack = lattice(16), Int[]
    @test all(wolff!(s, 1e-12, stack) == 1 for _ in 1:200)
    @test calibrate(s, 1e-12, stack) == length(s)      # ~1 spin per cluster
    # At β → ∞ every aligned neighbour joins: the cluster is a whole domain.
    s = fill(Int8(1), 16, 16)
    @test wolff!(s, 1e3, stack) == 256
    @test all(==(Int8(-1)), s)              # the whole lattice was flipped
    # Magnetisation and energy of a fully aligned lattice: E = -2N, M = N.
    s = fill(Int8(1), 8, 8)
    @test measure(s) == (-2 * 64, 64)
end

@testset "Onsager, infinite lattice" begin
    isfile(joinpath(DATA, "observables.csv")) || (@warn "run ising.jl first"; return)
    raw, _ = readdlm(joinpath(DATA, "observables.csv"), ',', Float64; header = true)
    L = maximum(Int.(raw[:, 1]))
    r = raw[Int.(raw[:, 1]) .== L, :]
    for k in axes(r, 1)
        T = r[k, 2]
        abs(T - TC) < 0.15 && continue      # skip the critical region
        @test isapprox(r[k, 3], onsager_energy(T); atol = 0.02)
        # Only below T_c is <|M|>/N an estimate of the infinite-system m:
        # above T_c the true value is 0 but a finite lattice keeps an
        # O(L^-1) tail, which is tested separately below.
        T < TC && @test isapprox(r[k, 4], onsager_m(T); atol = 0.02)
    end
    dev = maximum(abs(r[k, 3] - onsager_energy(r[k, 2]))
                  for k in axes(r, 1) if abs(r[k, 2] - TC) >= 0.15)
    @printf("  L=%d: max |e - e_Onsager| = %.4f away from T_c\n", L, dev)

    # Above T_c the residual magnetisation must shrink as the lattice grows.
    Ls = sort(unique(Int.(raw[:, 1])))
    hot = 2.9
    ms = map(Ls) do l
        q = raw[Int.(raw[:, 1]) .== l, :]
        q[argmin(abs.(q[:, 2] .- hot)), 4]
    end
    @printf("  <|m|> at T=%.1f: %s  for L = %s\n",
            hot, join((@sprintf("%.4f", x) for x in ms), ", "), join(Ls, ", "))
    @test issorted(ms; rev = true)          # decreasing with L
    @test ms[end] < ms[1] / 2
end

@testset "critical temperature from the susceptibility peaks" begin
    isfile(joinpath(DATA, "observables.csv")) || (@warn "run ising.jl first"; return)
    raw, _ = readdlm(joinpath(DATA, "observables.csv"), ',', Float64; header = true)
    Ls = sort(unique(Int.(raw[:, 1])))
    byL = map(Ls) do L
        r = raw[Int.(raw[:, 1]) .== L, :]
        r[sortperm(r[:, 2]), :]
    end
    # The peak of χ sits above T_c on a finite lattice and approaches it as
    # T_peak - T_c ~ L^(-1/ν) with ν = 1, so it must shrink roughly like 1/L
    # and always stay on the high side.
    peaks = [b[argmax(b[:, 6]), 2] for b in byL]
    @printf("  χ peak: %s   (exact T_c = %.4f)\n",
            join((@sprintf("L=%d → %.3f", L, p) for (L, p) in zip(Ls, peaks)), ", "), TC)
    for (L, p) in zip(Ls, peaks)
        @test p > TC - 0.02                 # never below T_c, up to grid spacing
        @test p - TC < 6 / L                # and within the 1/L envelope
    end
    @test abs(peaks[end] - TC) < abs(peaks[1] - TC)

    # Same statement from the specific heat, whose peak drifts the same way.
    cpeaks = [b[argmax(b[:, 5]), 2] for b in byL]
    @printf("  c peak: %s\n",
            join((@sprintf("L=%d → %.3f", L, p) for (L, p) in zip(Ls, cpeaks)), ", "))
    @test abs(cpeaks[end] - TC) < abs(cpeaks[1] - TC)
end

end
