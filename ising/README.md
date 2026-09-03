# 2D Ising model — Monte Carlo

Square lattice, `H = -Σ_<ij> s_i s_j`, periodic boundaries, `k_B = J = 1`.
Locates the critical point `T_c = 2/ln(1+√2) ≈ 2.269` from finite lattices.

## What must be installed

Julia (tested on 1.12.6, the juliaup `release` channel). Three packages beyond
what ships with Julia:

```julia
using Pkg; Pkg.add(["ProgressMeter", "CairoMakie", "LaTeXStrings"])
```

| script               | needs                                  |
| -------------------- | -------------------------------------- |
| `ising.jl`           | ProgressMeter                          |
| `ising_simple.jl`    | ProgressMeter                          |
| `ising_dynamics.jl`  | ProgressMeter                          |
| `ising_dynamics_NR.jl` | ProgressMeter                        |
| `test.jl`            | ProgressMeter (it loads `ising.jl`)    |
| `test_NR.jl`         | ProgressMeter (via `ising_dynamics_NR.jl`) |
| `plot.jl`            | CairoMakie, LaTeXStrings, ProgressMeter |

To check what is missing without running anything:

```julia
for p in (:ProgressMeter, :CairoMakie, :LaTeXStrings)
    try; @eval using $p; println(p, " ok"); catch; println(p, " MISSING"); end
end
```


## Files

| file                   | what it does                                                                          |
| ---------------------- | ------------------------------------------------------------------------------------- |
| `ising.jl`             | Wolff cluster sampling for the equilibrium sweep, Metropolis for the movie            |
| `ising_simple.jl`      | the same, with Metropolis only — shorter, but smaller lattices                        |
| `ising_dynamics.jl`    | Metropolis, Wolff, and Kawasaki spin-exchange, switchable per equilibration/sampling   |
| `ising_dynamics_NR.jl` | the *nonreciprocal* two-species model, with zero or two conservation laws (see below) |
| `plot.jl`              | figures and the movie, from whatever is in `data/`                                    |
| `test.jl`              | checks the equilibrium samplers against exactly known results                         |
| `test_NR.jl`           | checks the nonreciprocal dynamics against exact enumeration and the papers' phases    |

The three equilibrium scripts write the same two files, so `plot.jl` and
`test.jl` work with any of them. Running one overwrites another's output.
`ising_dynamics_NR.jl` writes to its own subdirectories of `data/` instead.

## Running

```
julia -t auto ising.jl             # ~4 min on 16 threads → data/
julia -t auto ising_dynamics.jl    # dynamics chosen in its main() → data/
julia -t auto ising_dynamics_NR.jl # dynamics chosen in its main() → data/<dir>/
julia plot.jl                      # ~1.5 min            → fig/
julia -t auto test.jl              # ~7 s
julia test_NR.jl                   # ~15 s, needs no prior run
```

`-t auto` matters: the temperature points are independent chains and are spread
over threads, so a single-threaded run takes roughly 16× longer. The scripts
warn if they are started with one thread.

In VS Code the run-file shortcut `include`s the file into the REPL, which the
scripts detect, so they run the same way as from the command line. `test.jl`
sets `ISING_LOAD_ONLY` before loading `ising.jl` so that it borrows the
functions without starting a multi-minute simulation. One consequence: if you
run `test.jl` and then `ising.jl` in the *same* REPL session, `ising.jl` will
not start, because that flag is still defined — restart the REPL.

## Output

`data/observables.csv` — one row per `(L, T)`: `L, T, e, m, c, chi`.
`data/frames.jls` — serialised `(; L, T, sweeps_per_frame, frames)` with
`frames::Array{Int8,4}` of size `(L, L, nframes, nT)`.

`fig/` — `critical`, `scaling`, `snapshots` (pdf + png) and `ising.mp4`.

Everything in `data/` and `fig/` is regenerable and is covered by the
repository `.gitignore`.

## The nonreciprocal model

`ising_dynamics_NR.jl` puts two spin species `σ^A, σ^B = ±1` on one periodic
`L×L` lattice. A spin on lattice `α` sees the local field

```
h_i^α = J Σ_{j nn of i} σ_j^α + K ε^{αβ} σ_i^β ,    ε^{AB} = +1 = -ε^{BA}
```

and has energy `E_i^α = -σ_i^α h_i^α`. Species `A` aligns with `B` while `B`
anti-aligns with `A`, so the two energies do not come from a common
Hamiltonian and the dynamics has no Gibbs steady state. Couplings are passed as
`J̃ = 4J/k_BT` and `K̃ = K/k_BT`. At `K̃ = 0` the species decouple and each
reduces to the equilibrium 2D Ising model.

Moves are accepted with probability `½[1 - tanh(βΔE/2)] = 1/(1 + e^{βΔE})`.
Two dynamics differ only in the move:

| function    | move                            | conservation laws |
| ----------- | ------------------------------- | ----------------- |
| `glauber!`  | single spin flip                | zero              |
| `kawasaki!` | same-species neighbour exchange | two (`M^A, M^B`)  |

The model is that of Avni et al., PRL **134**, 117103 (2025); the exchange
dynamics and its Cahn-Hilliard limit are Sects. 3-4 of Blom et al.,
arXiv:2507.01105.

### Exchange energy

An exchange swaps the spins on neighbouring sites `i` and `j` of one species.
Both candidate expressions take the form

```
βΔE = (σ_i - σ_j)(βh_i - βh_j)
```

and differ only in whether each neighbour sum includes the other exchanged
site:

```
(A)   βh_k = βJ Σ_{l nn of k}              σ_l + K̃ ε σ_k^β
(B)   βh_k = βJ Σ_{l nn of k, l ≠ partner} σ_l + K̃ ε σ_k^β
```

`kawasaki!` uses (B). Only opposite spins are ever exchanged, so
`(σ_i - σ_j)² = 4` and the two differ by a constant, independent of the
configuration, of `K̃`, of the species and of the direction of the move:

```
ΔE_A = ΔE_B - J̃
```

(B) is the change in the energy of lattice `α` at fixed `σ^β`. An exchange
leaves `σ_i σ_j` unchanged, so the bond between the two sites contributes
nothing, and dropping it from both fields gives the exact energy difference.
(A) instead treats the exchange as two independent flips, each in its own full
field, which counts that bond in both flips.

### Detailed balance

At `K̃ = 0` the dynamics must reproduce the equilibrium Ising model, which
requires detailed balance,

```
π(C) w(C→C') = π(C') w(C'→C)      with     π ∝ e^{-βH}
```

Write `x ≡ βH(C') - βH(C)` for the energy change of the move. Detailed balance
is then the condition

```
w(C→C') / w(C'→C) = e^{-x}
```

Under (B) the fields are unchanged by the exchange — neither site's remaining
neighbours nor the other species move — so the rule returns `x` forward and
`-x` back, and with `w = 1/(1 + e^{βΔE})`

```
w(C→C') / w(C'→C) = (1 + e^{x})⁻¹ / (1 + e^{-x})⁻¹ = (1 + e^{-x})/(1 + e^{x}) = e^{-x}
```

so detailed balance holds identically. Under (A) the constant offset applies in
both directions, giving `x - J̃` forward and `-x - J̃` back:

```
w(C→C') / w(C'→C) = (1 + e^{-x-J̃})/(1 + e^{x-J̃})
```

Setting this equal to `e^{-x}` and clearing denominators gives
`(1 - e^{-J̃})(1 - e^{-x}) = 0`, so it holds only at `J̃ = 0` or `x = 0`.
Detailed balance fails for every other move.

### Numerical check

`test_NR.jl` builds the full 12870×12870 transition matrix of the exchange
chain on the `M = 0` sector of a 4×4 lattice at `K̃ = 0`, `J̃ = 2.6`, and solves
`πP = π` directly. It compares `π` with the Boltzmann distribution and checks
detailed balance edge by edge through the asymmetry
`|f - b| / (f + b)`, where `f = π_C P(C→C')` and `b = π_C' P(C'→C)`: zero under
detailed balance, one for a strictly one-way edge.

| `βΔE` | `‖π - Boltzmann‖₁` | max edge asymmetry | sampled `⟨e⟩` (exact `-0.789`) |
| --- | --- | --- | --- |
| (B), used here | 5e-13 | 5e-13 | -0.794 |
| (A) | 0.80 | 0.80 | -0.430 |

Under (A) the chain carries stationary probability currents and its steady
state is not Boltzmann at any coupling: reproducing its mean energy would need
`J̃_eff = 1.57`, and even there the energy histogram is 0.28 away in total
variation. On a 64×64 lattice (A) fails to coarsen at `J̃ = 2, 3, 4` alike,
while (B) phase-separates. The two checks are logically separate — a chain can
carry currents and still hold a Boltzmann steady state — so both are made.

### Relation to the literature

(B) is the exchange energy of Penrose, *J. Stat. Phys.* **63**, 975 (1991),
his Eq. (8), defined there as `W(s^ab) - W(s)` and required by his
detailed-balance condition. (A) is the form written in the main text of Blom
et al.; their footnote 1 records (B) and cites Penrose for the observation that
it yields a mean-field free energy differing from the spin-flip one. That
difference is Penrose's Eqs. (32)-(33), `-½zεu²` against `-½(z+1)εu²`, the same
shift in the quadratic coefficient that `ΔE_A - ΔE_B` produces. Penrose calls
the mismatch a defect and attributes it to the pair factorisation
`E(s_a s_b) ≈ E(s_a) E(s_b)` used to close the mean-field equations, not to the
choice of exchange energy.

The two papers differ in what they do about it. Penrose keeps the exchange
energy equal to the energy change and accepts the mismatched free energy; Blom
et al. adjust the exchange energy so the two free energies agree, which their
derivation needs in order to share one free energy between the Allen-Cahn and
Cahn-Hilliard cases. For the resulting PDE and its linear stability that choice
is immaterial: both forms give a nonreciprocal Cahn-Hilliard equation of the
same structure, and the shifted coefficient moves phase boundaries in the
`(J, K)` plane without adding or removing types of instability. It matters for
a lattice simulation, where (A) is no longer the energy change and the `K̃ = 0`
limit is no longer the kinetic Ising model.

Two further points. Blom et al. justify detailed balance at `K_a = K_b` by
asserting that the fields interchange under an exchange, hence
`ΔE_ji = -ΔE_ij`; for neighbouring sites the fields of (A) do not interchange,
since after the swap `h_i` contains the moved spin `σ_i` rather than `σ_j`. The
fields that are unchanged, and for which the antisymmetry is exact, are those
of (B). Separately, their PDE and dispersion relations correspond to the free
energy of (A), whereas this simulation corresponds to one with a shifted `J`,
so comparing Monte Carlo results against those figures requires determining the
shift first.
