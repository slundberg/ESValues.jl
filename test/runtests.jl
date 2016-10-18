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
@test length(φ) == P
@test length(φVar) == P

# check computation of groups when nothing varies
X = ones(P, 4)
x = ones(P, 1)
f = x->sum(x, 1)
groups = [[1,2],[3],[4],[5]]
fnull,φg,φVar = esvalues(x, f, X, featureGroups=groups, nsamples=8)
@test length(φg) == 4

# check computation of groups when nothing varies
X = ones(P, 4)
x = ones(P, 1)
x[1,1] = 0
f = x->sum(x, 1)
groups = [[1,2],[3],[4],[5]]
fnull,φg,φVar = esvalues(x, f, X, featureGroups=groups, nsamples=8)
@test length(φg) == 4

# make sure things work when only two features vary
x = ones(P, 1)
x[1:2,1] = 0
X = ones(P, 1)
fnull,φ,φVar = esvalues(x, f, X, nsamples=8000)
@test fnull == 5
@test norm(X + φ .- x) < 1e-5
@test norm(φVar) < 1e-5

# X and x are identical
srand(1)
X = randn(P, 1)
fnull,φ,φVar = esvalues(x, f, X, nsamples=10)
@test fnull == sum(X)
@test norm(X + φ .- x) < 1e-5
@test norm(φVar) < 1e-5

# X and x are identical
srand(1)
x = zeros(P,1)
x[1,1] = 1
X = zeros(P,1)
fnull,φ,φVar = esvalues(x, f, X, nsamples=8)
@test fnull == sum(X)
@test norm(X + φ .- x) < 1e-5
@test norm(φVar) < 1e-5

# non-zero reference distribution
X = ones(P, 1)
fnull,φ,φVar = esvalues(x, f, X, nsamples=8000)
@test fnull == 5
@test norm(X + φ .- x) < 1e-5
@test norm(φVar) < 1e-5

X = randn(P, 1)
fnull,φ,φVar = esvalues(x, f, X, nsamples=10)
@test fnull == f(X)[1]
@test norm(X + φ .- x) < 1e-5
@test norm(φVar) < 1e-5

function rawShapley(x, f, X, ind, g=identity; featureGroups=nothing)

    featureGroups != nothing || (featureGroups = Array{Int64,1}[Int64[i] for i in 1:size(X)[1]])
    featureGroups = convert(Array{Array{Int64,1},1}, featureGroups)

    M = length(featureGroups)
    val = 0.0
    sumw = 0.0
    for s in subsets(setdiff(1:M, ind))
        S = length(s)
        w = factorial(S)*factorial(M - S - 1)/factorial(M)
        tmp = copy(X)
        for i in 1:size(X)[2]
            for j in s
                for k in featureGroups[j]
                    tmp[k,i] = x[k]
                end
            end
        end
        y1 = g(mean(f(tmp)))
        for i in 1:size(X)[2]
            for k in featureGroups[ind]
                tmp[k,i] = x[k]
            end
        end
        y2 = g(mean(f(tmp)))
        val += w*(y2-y1)
        sumw += w
    end
    @assert abs(sumw - 1.0) < 1e-6
    val
end

# check brute force computation of groups
X = randn(P, 4)
f = x->sum(x, 1)
groups = [[1,2],[3],[4],[5]]
@test abs(rawShapley(x, f, X, 1) + rawShapley(x, f, X, 2) - rawShapley(x, f, X, 1, featureGroups=groups)) < 1e-5

# check computation of groups
X = randn(P, 4)
f = x->sum(x, 1)
groups = [[1,2],[3],[4],[5]]
fnull,φ,φVar = esvalues(x, f, X, nsamples=10)
fnull,φg,φVar = esvalues(x, f, X, featureGroups=groups, nsamples=8)
@test abs(φ[1] + φ[2] - φg[1]) < 1e-5

# check computation of groups when nothing varies
X = ones(P, 4)
x = ones(P, 1)
f = x->sum(x, 1)
groups = [[1,2],[3],[4],[5]]
fnull,φ,φVar = esvalues(x, f, X, nsamples=10)
fnull,φg,φVar = esvalues(x, f, X, featureGroups=groups, nsamples=8)
@test abs(φ[1] + φ[2] - φg[1]) < 1e-5

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

# non-linear logistic function with groups
f = x->logistic(sum(x, 1))
groups = [[1,2],[3],[4],[5]]
fnull,φ,φVar = esvalues(x, f, X, featureGroups=groups, nsamples=10000)
φRaw = [rawShapley(x, f, X, i, featureGroups=groups) for i in 1:length(φ)]
for i in 1:length(φ)
    @test abs(φ[i] - φRaw[i]) < 1e-5
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
X = randn(P, 1)
x = randn(P)
f = x->logistic(sum(x, 1))
fnull,φ,φVar = esvalues(x, f, X, logit, nsamples=1000000)
for i in 1:length(φ)
    sv = rawShapley(x, f, X, i, logit)
    @test abs(φ[i] - rawShapley(x, f, X, i, logit)) < 1e-5
end
@test sum(abs(φVar)) < 1e-12 # we have exhausted the sample space so there should be no uncertainty

# non-linear logistic function with logit link and random background
P = 2
X = randn(P, 10)
x = randn(P)
f = x->logistic(sum(x, 1))
fnull,φ,φVar = esvalues(x, f, X, logit, nsamples=1000000)
φRaw = [rawShapley(x, f, X, i, logit) for i in 1:length(φ)]
@test norm(φ - φRaw) < 1e-5

# test many totally arbitrary functions with logit link
P = 10
X = zeros(P, 2)
x = ones(P,1)
for i in 1:10
    model = gen_model(P)
    f = x->[logistic(model[find(x[:,i])]) for i in 1:size(x)[2]]
    fnull,φ,φVar = esvalues(x, f, X, logit, nsamples=1000000)
    phiRaw = [rawShapley(x, f, X, j, logit) for j in 1:P]
    @test norm(φ .- phiRaw) < 1e-6
end

# test arbitrary functions with logit link and feature groups
P = 10
X = zeros(P, 2)
x = ones(P,1)
groups = [[1,2],[3],[4],[5,6],[7,8,9,10]]
for i in 1:3
    model = gen_model(P)
    f = x->[logistic(model[find(x[:,i])]) for i in 1:size(x)[2]]
    fnull,φ,φVar = esvalues(x, f, X, logit, nsamples=1000000, featureGroups=groups)
    phiRaw = [rawShapley(x, f, X, j, logit, featureGroups=groups) for j in 1:length(groups)]
    @test norm(φ .- phiRaw) < 1e-6
    @test abs(logistic(logit(fnull)+sum(φ)) - f(x)[1]) < 1e-6
end
