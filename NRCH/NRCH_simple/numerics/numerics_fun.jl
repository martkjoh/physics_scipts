using DelimitedFiles
using BenchmarkTools
using Random
import KernelAbstractions.Extras.LoopInfo: @unroll


write_folder = "data/sym/"
param_names = ["u, -r", "a", "D", "phi", "N", "dt"]

function write_file(φt, param, frames, N)
    filename = join(
        param_names[i] * '=' * string(param[i]) * "_"
        for i in range(1, length(param_names))
        )[1:end-1]
    writedlm(write_folder*filename*".txt", reshape(φt, (frames, 2*N)))
end
    
function check(φ, i, frames, pr)
    n = frames//10
    if any(isnan, φ)
        throw(ErrorException("Error, NaN detected" ))
    end

    if (div(i,n)) - div(i-1,n) == 1 && pr
        print("\r"*"|"^div(i,n))
    end
end


function sim(param_num, param_phys, deg=2)
    # parameters

    u, α, D, bφ = param_phys
    M, N, L, frames = param_num
    pr = true

    dx = L / N
    dt = round(.05 * (dx)^4; sigdigits=6)
    skip = div(M, frames)
    rng = MersenneTwister(1234)

    # Finite difference coeficcients, https://en.wikipedia.org/wiki/Finite_difference_coefficient
    coefficients = (
        [1/2] / dx,
        [2/3, -1/12] / dx,
        [3/4, -3/20, 1/60] / dx,
        [-1/280, 4/105, -1/5, 4/5] / dx
    )
    co = coefficients[deg]

    σ = sqrt(2 * D / dt / dx)

    
    # Derivative

    @inline ind(i) = mod(i-1, N)+1

    @inline function ∇(A, i)
        d = 0.
        @inbounds @unroll for j in 1:deg
            d += co[j]*( A[ind(i+j)] - A[ind(i-j)] ) 
        end
        return d
    end

    @inline function ∇²(A, i)
        d = 0.
        @inbounds @unroll for j in 1:deg
            d += co[j]*( ∇(A, i+j) - ∇(A, i-j)  ) 
        end
        return d
    end


    function euler!(φ, μ, δφ, ξ)
        @inbounds for i in 1:N
            @views ruφ² = u * (-1 + (φ[i, 1]^2 + φ[i, 2]^2 ))
            @views μ[i, 1] = ruφ² * φ[i, 1] - ∇²(φ[:, 1], i) + α * φ[i, 2]
            @views μ[i, 2] = ruφ² * φ[i, 2] - ∇²(φ[:, 2], i) - α * φ[i, 1]
        end 
        randn!(rng, ξ)
        ξ .*= σ
        @inbounds for i in 1:N
            @views δφ[i, 1] = ( ∇²(μ[:, 1], i) - ∇(ξ[:, 1], i) ) * dt
            @views δφ[i, 2] = ( ∇²(μ[:, 2], i) - ∇(ξ[:, 2], i) ) * dt
        end
    end
    
    function loop!(φt, φ, μ, δφ, ξ)
        for i in axes(φt, 1)[2:end]
            for _ in 1:skip
                euler!(φ, μ, δφ, ξ)
                φ .+= δφ
            end
            check(φ, i, frames, pr)
            φt[i,:,:] .= φ
        end
        print('\n')
    end

    function run_euler()    
        x = LinRange(0, L-dx, N)
        φ = .3 .* [sin.(-2*pi*x/L) cos.(-2*pi*x/L)]
        φ[:,1] .+= bφ

        φt = zeros(frames, N, 2)
        μ = zeros(N, 2)
        δφ = zeros(N, 2)
        ξ = zeros(N, 2)
        
        φt[1,:,:] .= φ
    
        loop!(φt, φ, μ, δφ, ξ)
        
        param = (u, α, D, bφ, N, dt)
        write_file(φt, param, frames, N)
    end

    run_euler()

end


function run_all()
    M = 1_000_000
    N = 100
    L = 10.
    frame = 1_000

    u = 10.
    D = .001
    bφ = .707
    α = 5.

    param_num = (M, N, L, frame)
    param_phys = (u, α, D, bφ)


    rm(write_folder, recursive=true, force=true)
    mkdir(write_folder[1:end-1])
    @time sim(param_num, param_phys)
end