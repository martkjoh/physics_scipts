using Symbolics
using LinearAlgebra

@variables α q r φ;

M = [
    r + q^2 + 3 * φ^2 + 0*im  α;
    -α  r + q^2 + φ^2
];

A = [1 2; 3 4];
# print(eigvecs(A))
# eigvecs(M)
# inv(M)
M
inv(M)

factorize(α^2 + 2 * q * α + q^2)
