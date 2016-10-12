using ESValues
using Base.Test
using Iterators
using StatsBase

include("iterators.jl")

# basic test
srand(1)
P = 5
X = zeros(P, 1)
x = randn(P, 1)
f = x->sum(x, 1)
fnull,φ,φVar = esvalues(x, f, X, nsamples=10)
@test fnull == 0
@test norm(φ .- x) < 1e-5
@test norm(φVar) < 1e-5

# X and x are identical
srand(1)
X = randn(P, 1)
fnull,φ,φVar = esvalues(x, f, X, nsamples=10)
#@test fnull == 0
@test norm(X + φ .- x) < 1e-5
@test norm(φVar) < 1e-5

# non-zero reference distribution
X = ones(P, 1)
fnull,φ,φVar = esvalues(x, f, X, nsamples=10)
@test fnull == 5
@test norm(X + φ .- x) < 1e-5
@test norm(φVar) < 1e-5

X = randn(P, 1)
fnull,φ,φVar = esvalues(x, f, X, nsamples=10)
@test fnull == f(X)[1]
@test norm(X + φ .- x) < 1e-5
@test norm(φVar) < 1e-5

function rawShapley(x, f, X, ind, g=identity)
    M = length(x)
    val = 0.0
    sumw = 0.0
    for s in subsets(setdiff(1:M, ind))
        S = length(s)
        w = factorial(S)*factorial(M - S - 1)/factorial(M)
        tmp = copy(X)
        for i in 1:size(X)[2]
            tmp[s,i] = x[s]
        end
        y1 = g(mean(f(tmp)))
        tmp[ind,:] = x[ind]
        y2 = g(mean(f(tmp)))
        if g != identity
            println(w, " ", y2, " ", y1, " ", s)
        end
        val += w*(y2-y1)
        sumw += w
    end
    @assert abs(sumw - 1.0) < 1e-6
    val
end

# check against brute force computation
X = randn(P, 4)
f = x->sum(x, 1)
fnull,φ,φVar = esvalues(x, f, X, nsamples=10)
for i in 1:length(φ)
    @test abs(φ[i] - rawShapley(x, f, X, i)) < 1e-5
end

# non-linear function
f = x->sum(x, 1).^2
fnull,φ,φVar = esvalues(x, f, X, nsamples=10)
for i in 1:length(φ)
    @test abs(φ[i] - rawShapley(x, f, X, i)) < 1e-5
end

# non-linear function that interestingly is still possible to estimate with only 2P samples
f = x->sum(x.^2, 1).^2
fnull,φ,φVar = esvalues(x, f, X, nsamples=10)
for i in 1:length(φ)
    @test abs(φ[i] - rawShapley(x, f, X, i)) < 1e-5
end

# non-linear logistic function
f = x->logistic(sum(x, 1))
fnull,φ,φVar = esvalues(x, f, X, nsamples=10000)
for i in 1:length(φ)
    @test abs(φ[i] - rawShapley(x, f, X, i)) < 1e-5
end

# test many totally arbitrary functions
function gen_model(M)
    model = Dict()
    for k in subsets(collect(1:M))
        model[k] = randn()
    end
    model
end
P = 10
X = zeros(P, 2)
x = ones(P,1)
for i in 1:10
    model = gen_model(P)
    f = x->[model[find(x[:,i])] for i in 1:size(x)[2]]
    fnull,φ,φVar = esvalues(x, f, X, nsamples=1000000)
    phiRaw = [rawShapley(x, f, X, j) for j in 1:P]
    @test norm(φ .- [rawShapley(x, f, X, j) for j in 1:P]) < 1e-6
end

# non-linear logistic function with logit link
# X = randn(P, 1)
# x = randn(P)
# f = x->logistic(sum(x, 1))
# fnull,φ,φVar = esvalues(x, f, X, :logit, nsamples=1000000)
# for i in 1:length(φ)
#     sv = rawShapley(x, f, X, i, logit)
#     println(sv, " ", x[i])
#     @test abs(φ[i] - rawShapley(x, f, X, i, logit)) < 1e-5
# end

# non-linear logistic function with logit link and random background
P = 2
X = randn(P, 2)
x = randn(P)
f = x->logistic(sum(x, 1))
fnull,φ,φVar = esvalues(x, f, X, :logit, nsamples=1000000)
φRaw = [rawShapley(x, f, X, i, logit) for i in 1:length(φ)]
println(φ)
println(φRaw)
println(φ - φRaw)
@test norm(φ - φRaw) < 1e-5

#     sv = rawShapley(x, f, X, i, logit)
#     println(sv, " ", x[i])
#     @test abs(φ[i] - rawShapley(x, f, X, i, logit)) < 1e-5
# end
