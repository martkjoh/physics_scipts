# Correctness tests for ising_dynamics_NR.jl.
#
#     julia test_NR.jl        (~15 s, needs no prior simulation run)
#
# At K̃ = 0 the species decouple and each dynamics reduces to the equilibrium
# 2D Ising model, so both are compared against exact enumeration of a 4×4
# lattice: `glauber!` over all 2^16 states, `kawasaki!` over the M = 0 sector.
# The exchange rule is additionally checked by solving πP = π for the full
# transition matrix. At K̃ ≠ 0 the conservation laws are checked, and the three
# phases of Avni et al. Fig. 1 identified by that paper's order parameters.

using Test, Printf, Random, SparseArrays, LinearAlgebra
const ISING_LOAD_ONLY = true            # take the functions, not main()
include(joinpath(@__DIR__, "ising_dynamics_NR.jl"))

# ------------------------------------------------------- exact references ---

"""
Bond energy and magnetisation of one species of a periodic 4×4 lattice, in
units where the bond coupling is βJ = J̃/4.
"""
function measure1(s::AbstractMatrix{Int8})
    L = size(s, 1); E = 0; M = 0
    @inbounds for j in 1:L, i in 1:L
        E -= Int(s[i, j]) * (Int(s[up(i, L), j]) + Int(s[i, up(j, L)]))
        M += Int(s[i, j])
    end
    return E, M
end

"""
Boltzmann averages over the 2^16 configurations of a 4×4 lattice at coupling
βJ = J̃/4, restricted to those with magnetisation `M` when `M !== nothing`
(the sector Kawasaki dynamics cannot leave). Returns ⟨E⟩ and ⟨|M|⟩ per site.
"""
function enumerate_4x4(J̃; M = nothing)
    L = 4; N = L^2; βJ = J̃ / 4
    s = Matrix{Int8}(undef, L, L)
    Z = e1 = m1 = 0.0
    for bits in 0:(2^N - 1)
        for k in 0:(N - 1)
            s[k + 1] = Int8(2 * ((bits >> k) & 1) - 1)
        end
        E, Mc = measure1(s)
        M === nothing || Mc == M || continue
        w = exp(-βJ * E)
        Z += w; e1 += w * E; m1 += w * abs(Mc)
    end
    return (e = e1 / Z / N, m = m1 / Z / N)
end

"""Time average of ⟨E⟩, ⟨|M|⟩ per site of species A under `dynamics!` at K̃ = 0."""
function sample_nr(dynamics!, J̃; L = 4, nequil = 20_000, nmeas = 400_000, init = random_lattice)
    s = init(L); N = L^2
    dynamics!(s, J̃, 0.0, nequil)
    e1 = m1 = 0.0
    for _ in 1:nmeas
        dynamics!(s, J̃, 0.0, 1)
        E, M = measure1(@view s[:, :, 1])
        e1 += E; m1 += abs(M)
    end
    return (e = e1 / nmeas / N, m = m1 / nmeas / N)
end

"""
`kawasaki!` with Blom's main-text ΔE: the local field of their Eq. (2), which
keeps the exchange partner in the neighbour sum. Used only in these tests.
"""
function kawasaki_maintext!(s::Array{Int8,3}, J̃::Float64, K̃::Float64, nsweeps::Int = 1)
    L = size(s, 1); βJ = J̃ / 4
    @inbounds for _ in 1:(nsweeps * length(s))
        i, j, α = rand(1:L), rand(1:L), rand(1:2)
        dir = rand(1:4)
        i2, j2 = dir == 1 ? (up(i, L), j) : dir == 2 ? (dn(i, L), j) :
                 dir == 3 ? (i, up(j, L)) : (i, dn(j, L))
        s1, s2 = s[i, j, α], s[i2, j2, α]
        s1 == s2 && continue
        nn1 = Int(s[up(i, L), j, α]) + Int(s[dn(i, L), j, α]) +
              Int(s[i, up(j, L), α]) + Int(s[i, dn(j, L), α])
        nn2 = Int(s[up(i2, L), j2, α]) + Int(s[dn(i2, L), j2, α]) +
              Int(s[i2, up(j2, L), α]) + Int(s[i2, dn(j2, L), α])
        ε = ifelse(α == 1, 1, -1)
        h1 = βJ * nn1 + K̃ * ε * Int(s[i, j, 3 - α])
        h2 = βJ * nn2 + K̃ * ε * Int(s[i2, j2, 3 - α])
        if glauber_accept((Int(s1) - Int(s2)) * (h1 - h2))
            s[i, j, α], s[i2, j2, α] = s2, s1
        end
    end
    return s
end

"""
    exact_chain(J̃; incl) -> (residual, tv, asym)

Exact stationary state of the single-species Kawasaki chain on the M = 0 sector
of the 4×4 lattice at K̃ = 0, with no sampling: the 12870×12870 transition
matrix is built and π solved from πP = π. `incl` selects whether the exchange
partner stays in each neighbour sum (Blom's main text) or is dropped (Penrose
Eq. (8), as in `kawasaki!`).

Returns the residual ‖πP - π‖_∞, the L1 distance ‖π - Boltzmann‖₁, and the
largest per-edge detailed-balance asymmetry
|π_C P(C→C') - π_C' P(C'→C)| / (π_C P(C→C') + π_C' P(C'→C)),
which is 0 exactly when detailed balance holds and 1 for a one-way edge.
"""
function exact_chain(J̃::Float64; incl::Bool)
    L = 4; N = 16; βJ = J̃ / 4
    nxt(i) = ifelse(i == L, 1, i + 1); prv(i) = ifelse(i == 1, L, i - 1)
    at(i, j) = (j - 1) * L + i
    bonds = vcat([(at(i, j), at(nxt(i), j)) for j in 1:L for i in 1:L],
                 [(at(i, j), at(i, nxt(j))) for j in 1:L for i in 1:L])
    nb = [[at(nxt(i), j), at(prv(i), j), at(i, nxt(j)), at(i, prv(j))]
          for j in 1:L for i in 1:L]
    nb = [nb[at(i, j)] for j in 1:L for i in 1:L]
    spins(b) = [Int(2 * ((b >> k) & 1) - 1) for k in 0:(N - 1)]

    sts = [b for b in 0:(2^N - 1) if count_ones(b) == 8]
    id = Dict(b => k for (k, b) in enumerate(sts)); n = length(sts)

    Ii = Int[]; Jj = Int[]; V = Float64[]; dg = ones(n)
    for (k, b) in enumerate(sts)
        s = spins(b)
        for (a, c) in bonds
            s[a] == s[c] && continue
            na = sum(s[m] for m in nb[a]); nc = sum(s[m] for m in nb[c])
            incl || (na -= s[c]; nc -= s[a])
            p = (1 / (1 + exp((s[a] - s[c]) * βJ * (na - nc)))) / 32
            push!(Ii, k); push!(Jj, id[b ⊻ ((1 << (a - 1)) | (1 << (c - 1)))]); push!(V, p)
            dg[k] -= p
        end
    end
    append!(Ii, 1:n); append!(Jj, 1:n); append!(V, dg)
    P = sparse(Ii, Jj, V, n, n)

    A = SparseMatrixCSC(copy(transpose(P)) - I); A[1, :] .= 1.0; dropzeros!(A)
    rhs = zeros(n); rhs[1] = 1.0
    π = A \ rhs; π ./= sum(π)

    Es = [(-sum(spins(b)[a] * spins(b)[c] for (a, c) in bonds)) for b in sts]
    bz = exp.(-βJ .* Es); bz ./= sum(bz)

    asym = 0.0; rows = rowvals(P); vals = nonzeros(P)
    for c in 1:n, r in nzrange(P, c)
        k = rows[r]; k == c && continue
        f = π[k] * vals[r]; g = π[c] * P[c, k]      # vals[r] = P[k,c]
        f + g > 0 && (asym = max(asym, abs(f - g) / (f + g)))
    end
    (residual = norm(vec(transpose(π) * P) - π, Inf), tv = norm(π - bz, 1), asym = asym)
end

"""Per-species magnetisation of a frame stack's underlying lattice."""
mags(s) = (sum(Int, @view s[:, :, 1]), sum(Int, @view s[:, :, 2]))

"""
Checkerboard L×L lattice, so both species start in the M = 0 sector.
Deterministic, since `kawasaki!` never leaves the sector it starts in.
"""
function half_filled(L::Int)
    s = Array{Int8}(undef, L, L, 2)
    @inbounds for α in 1:2, j in 1:L, i in 1:L
        s[i, j, α] = iseven(i + j) ? Int8(1) : Int8(-1)
    end
    return s
end

# ------------------------------------------------------------------ tests ---

Random.seed!(20250903)

@testset "nonreciprocal Ising" begin

@testset "K̃ = 0: glauber! reduces to the equilibrium Ising model" begin
    # Zero conservation laws, so the whole 2^16 state space is reachable.
    for J̃ in (1.2, 1.763, 2.6)
        ex = enumerate_4x4(J̃)
        si = sample_nr(glauber!, J̃)
        @printf("  J̃=%.3f  e: exact %+.5f  sim %+.5f   |m|: exact %.5f  sim %.5f\n",
                J̃, ex.e, si.e, ex.m, si.m)
        @test isapprox(si.e, ex.e; atol = 8e-3)
        @test isapprox(si.m, ex.m; atol = 8e-3)
    end
end

@testset "K̃ = 0: kawasaki! reduces to the fixed-M Ising model" begin
    # Two conservation laws, so the reachable state space is the M = 0 sector
    # alone and the exact reference is the Boltzmann average over just that.
    for J̃ in (1.2, 2.6)
        ex = enumerate_4x4(J̃; M = 0)
        si = sample_nr(kawasaki!, J̃; init = half_filled, nmeas = 2_000_000)
        @printf("  J̃=%.3f  M=0  e: exact %+.5f  sim %+.5f\n", J̃, ex.e, si.e)
        @test isapprox(si.e, ex.e; atol = 1e-2)
        @test si.m == 0                   # magnetisation frozen by construction
    end
end

@testset "detailed balance, exactly" begin
    # `exact_chain` rebuilds the rule from scratch, so this checks which ΔE is
    # correct; that `kawasaki!` implements it is what the sampling tests show.
    for (incl, name) in ((false, "partner excluded (Penrose Eq. 8)"),
                         (true,  "partner kept (Blom main text)   "))
        r = exact_chain(2.6; incl = incl)
        @printf("  %s  |π-Boltzmann|₁ = %.2e   max edge asymmetry = %.2e\n",
                name, r.tv, r.asym)
        @test r.residual < 1e-10          # π really is the stationary vector
        if incl
            @test r.tv   > 0.5            # stationary state is not Boltzmann
            @test r.asym > 0.5            # and the currents do not vanish
        else
            @test r.tv   < 1e-9           # π is the Boltzmann distribution
            @test r.asym < 1e-9           # detailed balance holds
        end
    end
end

@testset "Blom main-text ΔE, sampled" begin
    # Same J̃, same start, same sampling effort: the only difference is the ΔE.
    J̃ = 2.6
    ex = enumerate_4x4(J̃; M = 0)
    ok = sample_nr(kawasaki!, J̃; init = half_filled, nmeas = 2_000_000)
    bad = sample_nr(kawasaki_maintext!, J̃; init = half_filled, nmeas = 2_000_000)
    @printf("  J̃=%.1f  e: exact %+.5f   footnote-1 %+.5f   main-text %+.5f\n",
            J̃, ex.e, ok.e, bad.e)
    @test isapprox(ok.e, ex.e; atol = 1e-2)
    @test bad.e - ex.e > 0.15             # main-text rule is far too disordered
end

@testset "conservation laws" begin
    s = random_lattice(24)
    M0 = mags(s)
    kawasaki!(s, 3.0, 0.6, 40)
    @test mags(s) == M0                   # two conservation laws (Blom Sect. 4)

    s = lattice(24)
    glauber!(s, 1.6, 0.4, 40)
    @test mags(s) != M0 && mags(s) != (24^2, 24^2)   # zero conservation laws

    # Stronger than the sums above: the spin count per species is invariant.
    s = random_lattice(16)
    c0 = (count(==(Int8(1)), @view s[:, :, 1]), count(==(Int8(1)), @view s[:, :, 2]))
    kawasaki!(s, 2.0, 0.5, 100)
    @test (count(==(Int8(1)), @view s[:, :, 1]), count(==(Int8(1)), @view s[:, :, 2])) == c0
end

@testset "the three phases of Avni et al. Fig. 1" begin
    # Avni et al. Eqs. (4)-(5):
    #
    #   R = √((M_A² + M_B²)/2)           zero only in the disordered phase
    #   𝓛 = ⟨M_B ∂_t M_A - M_A ∂_t M_B⟩   nonzero only in the swap phase
    #
    # L is small deliberately: in 2D the swap phase is a finite-size effect.
    function trace(J̃, K̃; L = 24, nframes = 600, spf = 2)
        s = lattice(L); N = L^2
        a = Float64[]; b = Float64[]
        for _ in 1:nframes
            glauber!(s, J̃, K̃, spf)
            MA, MB = mags(s)
            push!(a, MA / N); push!(b, MB / N)
        end
        return a, b
    end
    R(a, b) = sum(sqrt.((a .^ 2 .+ b .^ 2) ./ 2)) / length(a)
    𝓛(a, b) = sum(b[k] * (a[k + 1] - a[k]) - a[k] * (b[k + 1] - b[k])
                  for k in 1:(length(a) - 1)) / (length(a) - 1)

    cases = (("disorder", 0.6, 0.3), ("swap", 2.2, 0.3), ("static order", 3.2, 0.05))
    out = map(cases) do (name, J̃, K̃)
        a, b = trace(J̃, K̃)
        r, l = R(a, b), 𝓛(a, b)
        @printf("  %-12s J̃=%.1f K̃=%.2f   R=%.3f   𝓛=%+.5f\n", name, J̃, K̃, r, l)
        (r = r, l = l)
    end
    dis, swp, ord = out

    # 𝓛 is compared against the swap value, not zero: on a finite lattice the
    # other phases keep a small residual circulation.
    @test dis.r < 0.15                    # disordered: no order of either kind
    @test swp.r > 0.4                     # swap: ordered ...
    @test swp.l > 5e-3                    # ... and circulating, with a chirality
    @test ord.r > 0.8                     # static order: ordered ...
    @test abs(dis.l) < 0.05 * swp.l       # ... but neither one circulates
    @test abs(ord.l) < 0.05 * swp.l

    # 𝓛 is odd in K̃: reversing its sign must reverse the circulation.
    a, b = trace(2.2, -0.3)
    l_rev = 𝓛(a, b)
    @printf("  swap, K̃<0                  𝓛=%+.5f\n", l_rev)
    @test l_rev < -5e-3
end

end
