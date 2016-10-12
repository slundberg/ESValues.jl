# our iterator type
type OrderedSubsets{T}
    members::Vector{T}
    valueFunction
    pq::Collections.PriorityQueue
    started::Bool
end

"Enumerate all the subsets ordered by the provided value function."
function orderedsubsets{T}(members::Vector{T}, f, order=:descending)
    orderby = x->-f([x])
    pqOrder = Base.Order.Reverse
    if order == :ascending
        orderby = x->f([x])
        pqOrder = Base.Order.Forward
    end
    OrderedSubsets(
        sort(members, by=orderby),
        f,
        Collections.PriorityQueue(Vector{Int64}, Float64, pqOrder),
        false
    )
end

Base.eltype(it::OrderedSubsets) = Vector{eltype(it.members)}
function Base.length(it::OrderedSubsets)
    length(it.members) < 40 ? 2^length(it.members) : Inf
end

Base.start{T}(it::OrderedSubsets{T}) = nothing

function Base.next{T}(it::OrderedSubsets{T}, state)
    if !it.started
        it.started = true
        Collections.enqueue!(it.pq, [1], it.valueFunction([it.members[1]]))
        return T[],nothing
    end

    # get the next best subset
    nextSubset = Collections.dequeue!(it.pq)

    # move this subset to look for the next "last member"
    if nextSubset[end] < length(it.members)
        nextSubsetInc = copy(nextSubset)
        nextSubsetInc[end] += 1
        Collections.enqueue!(it.pq, nextSubsetInc, it.valueFunction(T[it.members[i] for i in nextSubsetInc]))
    end

    # add the grown version of this subset
    if nextSubset[end] < length(it.members)
        nextSubsetGrown = [nextSubset; nextSubset[end]+1]
        Collections.enqueue!(it.pq, nextSubsetGrown, it.valueFunction(T[it.members[i] for i in nextSubsetGrown]))
    end

    T[it.members[i] for i in nextSubset],nothing
end

Base.done(it::OrderedSubsets, state) = isempty(it.pq) && it.started

# our iterator type
type ESKernelSubsets
    itr
    complement::Bool
    variances::Vector{Float64}
    lastSubset
    M
    numDone
    numTotal
end

function eskernelSubsetWeight(x, M)
    s = length(x)
    if s == M || s == 0
        return 1e12
    end
    (M-1)/(binomial(M,s)*s*(M-s))
end

"Enumerate all the subsets in descending order by the provided value function."
function eskernelsubsets(members, variances)
    M = length(members)

    # ensure all the variances are unique
    variances = variances .+ collect(M:-1:1)/1e5

    function subsetValue(x)

        s = length(x)
        if s == M || s == 0
            return 1e12
        elseif s > M/2
            return -1
        elseif s == 1
            return 1e10 + minimum(variances[x])
        end

        w = (M-1)/(binomial(M,s)*s*(M-s))
        minimum(variances[x])*w
    end

    ESKernelSubsets(
        orderedsubsets(members, subsetValue),
        false,
        variances,
        Int64[],
        M,
        0,
        M < 40 ? 2^M : Inf
    )
end

Base.eltype(it::ESKernelSubsets) = Vector{Float32}
function Base.length(it::ESKernelSubsets)
    it.numTotal
end

function Base.start(it::ESKernelSubsets)
    start(it.itr)
end

function Base.next(it::ESKernelSubsets, state)
    if !it.complement
        subsetElements,tmp = next(it.itr, nothing)

        it.lastSubset = subsetElements
        it.complement = true
        out = zeros(Float32, it.M)
        out[subsetElements] = 1
    else
        out = ones(Float32, it.M)
        out[it.lastSubset] = 0
        it.complement = false
    end
    it.numDone += 1

    return (out,eskernelSubsetWeight(find(out), it.M)),nothing
end

Base.done(it::ESKernelSubsets, state) = it.numDone >= it.numTotal
