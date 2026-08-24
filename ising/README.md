# 2D Ising model — Monte Carlo

Square lattice, `H = -Σ_<ij> s_i s_j`, periodic boundaries, `k_B = J = 1`.
Locates the critical point `T_c = 2/ln(1+√2) ≈ 2.269` from finite lattices.

## What must be installed

Julia (tested on 1.12.6, the juliaup `release` channel). Three packages beyond
what ships with Julia:

```julia
using Pkg; Pkg.add(["ProgressMeter", "CairoMakie", "LaTeXStrings"])
```

| script            | needs                                  |
| ----------------- | -------------------------------------- |
| `ising.jl`        | ProgressMeter                          |
| `ising_simple.jl` | ProgressMeter                          |
| `test.jl`         | ProgressMeter (it loads `ising.jl`)    |
| `plot.jl`         | CairoMakie, LaTeXStrings, ProgressMeter |

`Base.Threads`, `DelimitedFiles`, `Printf`, `Random`, `Serialization` and
`Test` ship with Julia — nothing to install for those.

To check what is missing without running anything:

```julia
for p in (:ProgressMeter, :CairoMakie, :LaTeXStrings)
    try; @eval using $p; println(p, " ok"); catch; println(p, " MISSING"); end
end
```

If a package is missing, the error is
`ArgumentError: Package X not found in current path`. Note that `Pkg.add`
installs into the *active* environment: if the REPL says something other than
`(@v1.12) pkg>`, you are in a project environment and need to add the packages
there too, or `activate` the default one first.

## Files

| file              | what it does                                                              |
| ----------------- | ------------------------------------------------------------------------- |
| `ising.jl`        | Wolff cluster sampling for the equilibrium sweep, Metropolis for the movie |
| `ising_simple.jl` | the same, with Metropolis only — shorter, but smaller lattices             |
| `plot.jl`         | figures and the movie, from whatever is in `data/`                         |
| `test.jl`         | checks the samplers against exactly known results                          |

Both simulation scripts write the same two files, so `plot.jl` and `test.jl`
work with either. Running one overwrites the other's output.

## Running

```
julia -t auto ising.jl          # ~4 min on 16 threads → data/
julia plot.jl                   # ~1.5 min            → fig/
julia -t auto test.jl           # ~7 s
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
