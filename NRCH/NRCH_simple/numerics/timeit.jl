using TimerOutputs
tmr = TimerOutput()

macro to_string(ex)
    @timeit tmr quote $(string(ex)) end ex
end

function test(n,x)
    @tm y = Vector{A}(undef,n)
    @tm for i in 1:n
        @tm y[i] = A(i*x)
    end
end