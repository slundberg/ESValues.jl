module ESValues

using GLM
using StatsBase

# package code goes here
include("iterators.jl")

export esvalues, ESValuesExplainer

typealias MaskType Float64

type ESValuesExplainer{T}
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
function esvalues(x, f::Function, X, link=:identity; featureGroups=nothing, weights=nothing, nsamples=0)
    esvalues(ESValuesExplainer(f, X, link; featureGroups=featureGroups, weights=weights, nsamples=nsamples), x)
end

function esvalues(e::ESValuesExplainer, x)

    @assert length(x) == e.P "Provided 'x' length must match the data matrix features count ($(length(x)) != $(e.P))!"
    e.x = x

    # find the feature groups we will test. If a feature does not change from its
    # current value then we know it doesn't impact the model
    e.varyingInds = varying_groups(e.x, e.X, e.featureGroups)
    e.varyingFeatureGroups = e.featureGroups[e.varyingInds]
    e.M = length(e.varyingFeatureGroups)

    # if no features vary then there no feature has an effect
    if e.M == 0
        return e.f(x),zeros(e.P),zeros(e.P)
    end

    # pick a reasonable number of samples if the user didn't specify how many they wanted
    if e.nsamples == 0
        e.nsamples = 2*e.M+1000
    end
    if e.nsamples > 2^e.M-2
        e.nsamples = 2^e.M-2
    end
    @assert e.nsamples >= min(2*e.M, 2^e.M-2) "'nsamples' must be at least 2 times the number of varying feature groups!"

    # find f(x) and E_x[f(x)]
    allocate!(e)
    addsample!(e, x, ones(e.M), 1.0)
    addsample!(e, x, zeros(e.M), 1.0)
    run!(e)
    e.fx = e.ey[1]
    e.fnull = e.ey[2]
    reset!(e)
#     display(e.data[:,1:e.N])
#     display(e.data[:,e.N+1:2*e.N])
#     display(e.y)
    # println("fx: ", e.fx)
    # println("fnull: ", e.fnull)
#     return
    # add the singleton samples and then estimate the variance of each ES value estimate
    for (m,w) in take(drop(eskernelsubsets(collect(1:e.M), ones(e.M)), 2), 2*e.M)
        addsample!(e, x, m, w)
    end
    run!(e)
    variances = zeros(e.M)
    for i in 1:2:2*e.M
        variances[div(i+1,2)] = var([e.y[i] - e.fnull, e.fx - e.y[i+1]])
    end
    #println("variances ", variances)

    # now add the rest of the samples giving priority to ES values with high estimated variance
    for (m,w) in take(drop(eskernelsubsets(collect(1:e.M), variances), 2*e.M+2), e.nsamples-(2*e.M))
        #println("sample")
        addsample!(e, x, m, w)
    end
    run!(e)
    φ,φVar = solve!(e)

    # return the Shapley values along with variances of the estimates
    e.fnull,φ,φVar
end

function ESValuesExplainer{T}(f::Function, X::Matrix{T}, link=:identity; featureGroups=nothing, weights=nothing, nsamples=0)
    P,N = size(X)

    # give default values to omitted arguments
    weights != nothing || (weights = ones(N))
    weights ./= sum(weights)
    featureGroups != nothing || (featureGroups = Array{Int64,1}[Int64[i] for i in 1:size(X)[1]])
    featureGroups = convert(Array{Array{Int64,1},1}, featureGroups)
    @assert length(weights) == N "Provided 'weights' must match the number of representative data points (size(X)[2])!"

    ESValuesExplainer(
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

function allocate!{T}(e::ESValuesExplainer{T})
    e.data = zeros(T, e.P, e.nsamples * e.N)
    #println("e.M ", e.M)
    e.maskMatrix = zeros(MaskType, e.M-1, e.nsamples)
    e.kernelWeights = zeros(MaskType, e.nsamples)
    e.y = zeros(Float64, e.nsamples * e.N)
    e.ey = zeros(Float64, e.nsamples)
    e.lastMask = zeros(Float64, e.nsamples)
end

function addsample!(e::ESValuesExplainer, x, m, w)
    offset = e.nsamplesAdded * e.N
    e.nsamplesAdded += 1
    for i in 1:e.N
        e.data[m .== 1,offset+i] = x[m .== 1]
        e.data[m .== 0,offset+i] = e.X[m .== 0,i]
    end
    #println(m[end])
    e.maskMatrix[:,e.nsamplesAdded] = m[1:end-1] - m[end]
    #println(e.fnull)
    # -m[end]*(e.fx - e.fnull) - e.fnull
    #e.ey[e.nsamplesAdded] = 0#-m[end]*(e.fx - e.fnull) - e.fnull
    e.lastMask[e.nsamplesAdded] = m[end]
    e.kernelWeights[e.nsamplesAdded] = w
end

function run!(e::ESValuesExplainer)
    e.y[:] = e.f(e.data)

#     println("run")
#     println(e.ey)

    # find the expected value of each output
    for i in e.nsamplesRun+1:e.nsamplesAdded
        eyVal = 0.0
        for j in 1:e.N
            #println("eyValx ", eyVal)
            eyVal += e.y[(i-1)*e.N + j]
        end
        #println("eyVal ", eyVal, " ", e.N, " $i")
        e.ey[i] = eyVal/e.N
        e.nsamplesRun += 1
    end
end

function reset!(e::ESValuesExplainer)
    e.nsamplesAdded = 0
    e.nsamplesRun = 0
end

function solve!(e::ESValuesExplainer)
    tmp = e.maskMatrix .* e.kernelWeights'
    #println(typeof(tmp))
    #println(size(e.maskMatrix))
    #display(svd(tmp))

    # d = Dict()
    # for i in 1:size(e.maskMatrix)[2]
    #     k = vec(e.maskMatrix[:,i])
    #     d[k] = get(d, k, 0) + 1
    #     if d[k] > 1
    #         println("i $i")
    #     end
    # end
    # mc,mcInd = findmax(values(d))
    # ks = collect(keys(d))
    #println("max count ", mc, ks[mcInd])
    #println(find(map(x->all(x .== ks[mcInd]), ks)))

    # display(e.maskMatrix)
    # display(e.data)
    # println("e.maskMatrix ", e.maskMatrix)
    # println("e.ey ", e.ey)
#     println("e.data ", e.data)
#     println("e.y ", e.y)
    #println("e.fx ", e.fx)
    #println("e.fnull ", e.fnull)

#     println(inv(e.maskMatrix*e.maskMatrix')*e.maskMatrix*e.ey)



    tmp2 = inv(tmp*e.maskMatrix')
    #display(svd(tmp2))
    #println(tmp*e.ey)
    #w = tmp2*(tmp*e.ey)

    # local φ
    # local φVar
    # if e.link == :identity
    #     # println()
    #     # println(size(e.ey))
    #     # println(size(e.lastMask))
    #     # println((e.fx - e.fnull))
    #     # println(e.fnull)
    #     eyAdj = e.ey .- e.lastMask*(e.fx - e.fnull) - e.fnull
    #     m = fit(GeneralizedLinearModel, e.maskMatrix', vec(eyAdj), Normal(), IdentityLink(), wts=e.kernelWeights)
    #     w = coef(m)
    #     wlast = (e.fx - e.fnull) - sum(w)
    #     φ = [w; wlast]
    #     yHat = e.maskMatrix'w
    #     φVar = var(yHat .- eyAdj) * diag(tmp2)
    #
    # elseif e.link == :logit
    #     #z = e.lastMask.*(logit(e.fx) - logit(e.fnull))
    #     println(extrema(e.ey))
    #     println(e.fnull)
    #     println(size((e.ey - e.fnull*e.ey)))
    #     #eyAdj = (e.ey - e.fnull*e.ey)./(-2*e.fnull*e.ey + e.fnull + e.ey)
    #
    #     # we solve for p in the following equation to get the formula below
    #     # logit(p) = logit(y) - logit(fnull) - m*(logit(fx) - logit(fnull))
    #     tmp3 = (e.fx/(1-e.fx)).^e.lastMask
    #     tmp4 = (e.fnull/(1-e.fnull)).^e.lastMask
    #     tmp5 = e.ey .* tmp4
    #     println("size(tmp3) ", size(tmp3))
    #     println("size(tmp4) ", size(tmp4))
    #     println("size(tmp5) ", size(tmp5))
    #     eyAdj = ((e.fnull-1)*tmp5) ./ (e.fnull*(e.ey.*(tmp3 .+ tmp4) .- tmp3) .- tmp5)
    #
    #     println("extrema(eyAdj) ", extrema(eyAdj))
    #     #eyAdj = logistic(eyAdj./(eyAdj .- (eyAdj-1).*exp.(z)))
    #     #println(extrema(eyAdj))
    #     println(e.maskMatrix')
    #     println(vec(eyAdj))
    #     println(vec(e.kernelWeights))
    #     println("e.ey ", e.ey)
    #     println("e.ey ", logit(e.ey))
    #     m = fit(GeneralizedLinearModel, e.maskMatrix', vec(eyAdj), Normal(), LogitLink(), wts=e.kernelWeights)
    #     w = coef(m)
    #     wlast = (logit(e.fx) - logit(e.fnull)) - sum(w)
    #     φ = [w; wlast]
    #     φVar = ones(length(φ)) #var(yHat .- eyAdj) * diag(tmp2)
    # else
    #     error("Unknown link function $(e.link)! Use :identity or :logistic!")
    # end

    # println(size(e.maskMatrix'))
    # println(size(vec(eyAdj)))
    #
    # m = fit(GeneralizedLinearModel, e.maskMatrix', vec(eyAdj), Normal(), linkObj, wts=e.kernelWeights)
    # # println()
    # # println(coef(m))
    # # println(w)
    # w = coef(m)
    #
    #
    local linkf
    if e.link == :identity
        linkf = identity
    elseif e.link == :logit
        linkf = logit
    end

    eyAdj = linkf(e.ey) .- e.lastMask*(linkf(e.fx) - linkf(e.fnull)) - linkf(e.fnull)
    w = tmp2*(tmp*eyAdj)

    wlast = (linkf(e.fx) - linkf(e.fnull)) - sum(w)
    φ = [w; wlast]

    yHat = e.maskMatrix'w
    #println("yHat ", yHat)
    #println(size(yHat), " ", size(e.ey))
    φVar = var(yHat .- eyAdj) * diag(tmp2)

    #println(size(e.maskMatrix))

    φ,φVar
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
