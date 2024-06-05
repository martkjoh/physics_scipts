using BenchmarkTools

N = 1000
A = rand(N, N)


function col_slow(A)
    for i in 1:N
        for j in 1:N
            A[j, i] += 1
        end
    end
end

function col(A)
    for i in 1:axes(A, 1)
        for j in 1:size(A, 2)
            A[j, i] += 1
        end
    end
end

function row(A)
    for i in 1:100
        for j in 1:100
            A[i, j] += 1
        end
    end
end


@btime col_slow(A)
@btime col(A)
@btime row(A)


# function alongcolumns(w)
#     for j in axes(w, 2)
#         for i in axes(w, 1)
#             w[i, j] += 1
#         end
#     end
# end

# function alongrows(w)
#     for i in axes(w, 2)
#         for j in axes(w, 1)
#             w[i, j] += 1
#         end
#     end
# end

# w = rand(100, 100);

# @btime alongcolumns($w)
# @btime alongrows($w)