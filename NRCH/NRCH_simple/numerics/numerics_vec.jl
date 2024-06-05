using DelimitedFiles
using Random
using BenchmarkTools
using Tullio
using LoopVectorization

const N = 200
const M = 100_000
const L = 10
const dx = L / N
const dt = .1 * (dx)^4
const frames = 100
const skip = div(M, frames)

print("T = ", M*dt, '\n')
param_names = ["u, -r", "bφ", "a", "b"]

i = range(1, N)
DD = zeros((N, N))
for i in range(1, N)
    DD[i, i] = - 2 / dx^2
    DD[i, i%N + 1] = 1 / dx^2
    DD[i%N + 1, i] = 1 / dx^2
end

∇² = DD
∇²dt = DD*dt
eps = [0 .1; -.1 0] 

function euler1!(
    δφ::Array{Float64, 2}, 
    φ::Array{Float64, 2}, 
    temp::Array{Float64, 2}, 
    param::NTuple{4, Float64}
    )
    u, bφ, α, β = param
    randn!(temp)

    @. δφ = - u * φ
    @views @. δφ += u * (φ[:,1]^2 + φ[:,2]^2 ) * φ
    δφ .-= ∇² * φ
    @views @. δφ[:, 1] += α * φ[:, 2]
    @views @. δφ[:, 2] -= α * φ[:, 1]
    @. δφ += β * temp
    δφ .= ∇²dt * δφ
end

# function euler2!(
#     δφ::Array{Float64, 2}, 
#     φ::Array{Float64, 2}, 
#     temp::Array{Float64, 2}, 
#     param::NTuple{4, Float64}
#     )
#     u, bφ, a, β = param
#     randn!(temp)
    
#     @tullio δφ[x, i] = u*(-1 + ( φ[x, j] * φ[x, j] ) )* φ[x, i]
#     @tullio δφ[x, i] += -∇²[x, y] * φ[y, i]
#     @tullio δφ[x, i] += α * eps[i, j] * φ[x, j]
#     @tullio δφ[x, i] += β * temp[x, i]
#     @tullio δφ[x, i] = ∇²[x, y] * δφ[y, i] * dt
# end

function check(φ, i)
    n = frames//10
    if any(isnan, φ)
        throw(ErrorException("Error, NaN detected" ))
    end
    if (div(i,n)) - div(i-1,n) == 1
        print("\r"*"|"^div(i,n))
    end
end

function loop(param)
    φt = zeros(frames, N, 2)
    φ = zeros(N, 2)
    δφ = zeros(N, 2)
    temp = zeros(N, 2)

    φ[:, 1] .= bφ
    φt[1,:,:] .= φ
    
    for i in axes(φt, 1)[2:end]
        for j in 1:skip
            euler1!(δφ, φ, temp, param)
            # euler2!(δφ, φ, temp, param)
            φ .+= δφ
        end
        check(φ, i)
        φt[i,:,:] .= φ
    end
    print('\n')
    return φt
end

function write_file(φt, param)
    filename = join(
        param_names[i] * '=' * string(param[i]) * '_' 
        for i in range(1, length(param_names))
        )[1:end-1]
    writedlm("data/"*filename*".txt", reshape(φt, (frames, 2*N)))
end

function run_euler(param)
    φt = loop(param)
    # write_file(φt, param)
    return φt[end,:,:]
end

α = 0.
bφ = -.8
u = 10.
β = 0.5

param = (u, bφ, α, β)
@time run_euler(param);
