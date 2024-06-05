using Profile
using TimerOutputs

N = 10000
M = 100
a = collect(LinRange(0, 1, N))
b = collect(LinRange(0, 2, N))
dx = 1 / N


i = range(1, N)
D2 = zeros((N, N))
for i in range(1, N)
    D2[i, i] = - 2 / dx^2
    D2[i, i%N + 1] = 1 / dx^2
    D2[i%N + 1, i] = 1 / dx^2
end


function mul(a::Vector{Float64}, b::Vector{Float64}, D2::Matrix{Float64}, M::Int)
    for i in range(1, M)
        b .= a + D2*b
    end
end
@profview mul(a, b, D2, M)
