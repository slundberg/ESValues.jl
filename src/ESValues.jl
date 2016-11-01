module ESValues

using GLM
using StatsBase

# package code goes here
include("iterators.jl")

export esvalues, ESValuesEstimator

typealias MaskType Float64

type ESValuesEstimator{T}
    x
    f::Function
    X
    link
    featureGroups::Vector{Vector{Int64}}
    weights
    nsamples::Int64
    varyingInds::Vector{Int64}
    varyingFeatureGroups::Vector{Vector{Int64}}
    data::Matrix{T}
    maskMatrix::Matrix{MaskType}
    kernelWeights::Vector{MaskType}
    y::Vector{Float64}
    ey::Vector{Float64}
    lastMask::Vector{Float64}
    P::Int64
    N::Int64
    M::Int64
    nsamplesAdded::Int64
    nsamplesRun::Int64
    fx::Float64
    fnull::Float64
end

"Designed to determine the ES values (importance) of each feature for f(x)."
function esvalues(x, f::Function, X, link=identity; featureGroups=nothing, weights=nothing, nsamples=0)
    esvalues(ESValuesEstimator(f, X, link; featureGroups=featureGroups, weights=weights, nsamples=nsamples), x)
end

function esvalues(e::ESValuesEstimator, x)

    @assert length(x) == e.P "Provided 'x' length must match the data matrix features count ($(length(x)) != $(e.P))!"
    e.x = x

    # find the feature groups we will test. If a feature does not change from its
    # current value then we know it doesn't impact the model
    e.varyingInds = varying_groups(e.x, e.X, e.featureGroups)
    e.varyingFeatureGroups = e.featureGroups[e.varyingInds]
    e.M = length(e.varyingFeatureGroups)

    # find f(x) and E_x[f(x)]
    e.fx = e.f(x)[1]
    e.fnull = sum(vec(e.f(e.X)) .* e.weights)

    # if no features vary then there no feature has an effect
    if e.M == 0
        return e.fx,zeros(length(e.featureGroups)),zeros(length(e.featureGroups))

    # if only one feature varies then it has all the effect
    elseif e.M == 1
        fx = mean(e.f(x))
        fnull = sum(vec(e.f(e.X)) .* e.weights)
        φ = zeros(length(e.featureGroups))
        φ[e.varyingInds[1]] = e.link(e.fx) - e.link(e.fnull)
        return e.fnull,φ,zeros(length(e.featureGroups))
    end

    # pick a reasonable number of samples if the user didn't specify how many they wanted
    if e.nsamples == 0
        e.nsamples = 2*e.M+1000
    end
    if e.M <= 30 && e.nsamples > 2^e.M-2
        e.nsamples = 2^e.M-2
    end
    @assert e.nsamples >= min(2*e.M, 2^e.M-2) "'nsamples' must be at least 2 times the number of varying feature groups!"

    # add the singleton samples
    allocate!(e)
    for (m,w) in take(drop(eskernelsubsets(collect(1:e.M), ones(e.M)), 2), 2*e.M)
        addsample!(e, x, m, w)
    end
    run!(e)

    # if there might be more samples then enumarate them
    if length(e.y) >= 2*e.M

        # estimate the variance of each ES value estimate
        variances = zeros(e.M)
        for i in 1:2:2*e.M
            variances[div(i+1,2)] = var([e.y[i] - e.fnull, e.fx - e.y[i+1]])
        end

        # now add the rest of the samples giving priority to ES values with high estimated variance
        for (m,w) in take(drop(eskernelsubsets(collect(1:e.M), variances), 2*e.M+2), e.nsamples-(2*e.M))
            addsample!(e, x, m, w)
        end
        run!(e)
    end

    # solve then expand the ES values vector to contain the non-varying features as well
    vφ,vφVar = solve!(e)
    φ = zeros(length(e.featureGroups))
    φ[e.varyingInds] = vφ
    φVar = zeros(length(e.featureGroups))
    φVar[e.varyingInds] = vφVar

    # return the Shapley values along with variances of the estimates
    e.fnull,φ,φVar
end

function ESValuesEstimator{T}(f::Function, X::Matrix{T}, link=identity; featureGroups=nothing, weights=nothing, nsamples=0)
    P,N = size(X)

    # give default values to omitted arguments
    weights != nothing || (weights = ones(N))
    weights ./= sum(weights)
    featureGroups != nothing || (featureGroups = Array{Int64,1}[Int64[i] for i in 1:size(X)[1]])
    featureGroups = convert(Array{Array{Int64,1},1}, featureGroups)
    @assert length(weights) == N "Provided 'weights' must match the number of representative data points (size(X)[2])!"

    ESValuesEstimator(
        zeros(1),
        f,
        X,
        link,
        featureGroups,
        weights,
        nsamples,
        Int64[],
        Vector{Int64}[],
        zeros(T, 1, 1),
        zeros(MaskType, 1, 1),
        zeros(MaskType, 1),
        zeros(Float64, 1),
        zeros(Float64, 1),
        zeros(Float64, 1),
        P,
        N,
        0,
        0,
        0,
        0.0,
        0.0
    )
end

function allocate!{T}(e::ESValuesEstimator{T})
    e.data = zeros(T, e.P, e.nsamples * e.N)
    e.maskMatrix = zeros(MaskType, e.M-1, e.nsamples)
    e.kernelWeights = zeros(MaskType, e.nsamples)
    e.y = zeros(Float64, e.nsamples * e.N)
    e.ey = zeros(Float64, e.nsamples)
    e.lastMask = zeros(Float64, e.nsamples)
end

function addsample!(e::ESValuesEstimator, x, m, w)
    offset = e.nsamplesAdded * e.N
    e.nsamplesAdded += 1
    for i in 1:e.N
        for j in 1:e.M
            for k in e.varyingFeatureGroups[j]
                if m[j] == 1.0
                    e.data[k,offset+i] = x[k]
                else
                    e.data[k,offset+i] = e.X[k,i]
                end
            end
        end
    end
    e.maskMatrix[:,e.nsamplesAdded] = m[1:end-1] - m[end]
    e.lastMask[e.nsamplesAdded] = m[end]
    e.kernelWeights[e.nsamplesAdded] = w
end

function run!(e::ESValuesEstimator)
    e.y[e.nsamplesRun*e.N+1:e.nsamplesAdded*e.N] = e.f(e.data[:,e.nsamplesRun*e.N+1:e.nsamplesAdded*e.N])
    
    # find the expected value of each output
    for i in e.nsamplesRun+1:e.nsamplesAdded
        eyVal = 0.0
        for j in 1:e.N
            eyVal += e.y[(i-1)*e.N + j] * e.weights[j]
        end
        e.ey[i] = eyVal
        e.nsamplesRun += 1
    end
end

function solve!(e::ESValuesEstimator)

    # adjust the y value according to the constraints for the offset and sum
    eyAdj = e.link.(e.ey) .- e.lastMask*(e.link(e.fx) - e.link(e.fnull)) - e.link(e.fnull)

    # solve a weighted least squares equation to estimate φ
    tmp = e.maskMatrix .* e.kernelWeights'
    tmp2 = inv(tmp*e.maskMatrix')
    w = tmp2*(tmp*eyAdj)
    wlast = (e.link(e.fx) - e.link(e.fnull)) - sum(w)
    φ = [w; wlast]

    yHat = e.maskMatrix'w
    φVar = var(yHat .- eyAdj) * diag(tmp2)
    φVar = [φVar; maximum(φVar)] # since the last weight is inferred we use a pessimistic guess of its variance

    # a finite sample adjustment based on how much of the weight is left in the sample space
    fractionWeightLeft = 1 - sum(e.kernelWeights)/sum([(e.M-1)/(s*(e.M-s)) for s in 1:e.M-1])

    φ,φVar*fractionWeightLeft
end

"Identify which feature groups vary."
function varying_groups(x, X, featureGroups)
    varying = zeros(length(featureGroups))
    for (i,inds) in enumerate(featureGroups)
        varying[i] = sum(vec(sum(x[inds] .== X[inds,:],1) .!= length(inds)))
    end
    find(varying)
end

end # module
