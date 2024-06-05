using DelimitedFiles
using Base.Threads
using BenchmarkTools

const N = 200
const M = 2_000_000
const L = 10.
const dx = L / N
const dt = .1 * (dx)^4
const frames = 1000
const skip = div(M, frames)
@assert skip*frames == M

print("T = ", M*dt, '\n')
param_names = ["u, -r", "a", "b", "phi1", "phi2"]


@inline ind(i) = mod(i-1, N)+1
@inline ∇²(A, dx, i) =  (A[ind(i+1)] + A[ind(i-1)] - 2*A[i]) / dx^2

function euler!(
    φ::Array{Float64, 2}, 
    μ::Array{Float64, 2},
    δφ::Array{Float64, 2}, 
    param::NTuple{5, Float64}
    )
    
    u, φ1, φ2, a, β = param
    @inbounds for i in axes(φ,1)
        @views μ[i, 1] = u * (-1 + 2*φ[i, 1]^2) * φ[i, 1] + a * φ[i, 2] - ∇²(φ[:, 1], dx, i) + β*randn(Float64)
        @views μ[i, 2] = u * (-1 + 2*φ[i, 2]^2) * φ[i, 2] - a * φ[i, 1] - ∇²(φ[:, 2], dx, i) + β*randn(Float64) 
    end
    @inbounds for i in axes(φ,1)
        @views δφ[i, 1] = ∇²(μ[:, 1], dx, i) * dt
        @views δφ[i, 2] = ∇²(μ[:, 2], dx, i) * dt
    end
end


function check(φ, i)
    n = frames//10
    if any(isnan, φ)
        throw(ErrorException("Error, NaN detected" ))
    end

    if (div(i,n)) - div(i-1,n) == 1
        print("\r"*"|"^div(i,n))
    end
end

function loop(param::NTuple{5, Float64})
    u, a, β, φ1, φ2 = param
    nn = 1
    x = collect(LinRange(0, L-dx, N))
    φ = .1 * [sin.(nn*2*pi*x/L) cos.(nn*2*pi*x/L)]
    # φ = zeros(N, 2)
    φt = zeros(frames, N, 2)
    μ = zeros(N, 2)
    δφ = zeros(N, 2)
    
    φ[:,1] .+= φ1
    φ[:,2] .+= φ2
    φt[1,:,:] .= φ

    for i in axes(φt, 1)[2:end]
        for j in 1:skip
            euler!(φ, μ, δφ, param)
            φ .+= δφ
        end
        check(φ, i)
        φt[i,:,:] .= φ
    end
    
    print('\n')
    return φt
end


function run_euler(param::NTuple{5, Float64})
    φt = loop(param)
    write_file(φt, param)
    return
end

##############
# Utillities #
##############

function write_file(φt, param)
    filename = join(
        param_names[i] * '=' * string(param[i]) * "_"
        for i in range(1, length(param_names))
        )[1:end-1]
    filename = filename * "_N=" * string(N)
    filename = filename * "_dt=" * string(dt * skip)
    writedlm("data/assym/"*filename*".txt", reshape(φt, (frames, 2*N)))
end



# param = (10., 1., 1., -.2, -.1)
# @time run_euler(param);

u, β = 10., 1.
αs = [0, 5., 10, 15]
φs = -[[0, 0], [0.3, 0.3], [.6, .6], [.9, .9], [1.2, 1.2], [0, .3], [0, .6], [0, .9], [0, 1.2]]
αφ = [(α, φ[1], φ[2]) for α in αs for φ in φs]
@time @threads for (α, φ1, φ2) in αφ
    param = (u, α, β, φ1, φ2)
    @time run_euler(param)
end
