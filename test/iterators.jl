

function test_subsets(members, f, itr, order=:descending)
    alreadySeen = Dict()
    lastValue = order == :descending ? Inf : -Inf
    totalCount = 0
    for s in itr
        if !isempty(s)
            if order == :descending
                @test f(s) <= lastValue
            else
                @test f(s) >= lastValue
            end

            lastValue = f(s)
        end

        alreadySeen[s] = get(alreadySeen, s, 0) + 1
        @test alreadySeen[s] == 1
        totalCount += 1
    end
    @test totalCount == 2^length(members)
end

# test orderedsubsets using a minimum function
members = [4.19897,0.0835427,0.0452934,1.77052,0.0945746,4.88804,0.135961]
test_subsets(members, minimum, ESValues.orderedsubsets(members, minimum))


# test orderedsubsets using a value that rejects subsets larger than a given value
P = 7
function subsetValue(x)

    s = length(x)
    if s == 0
        return 1e12
    elseif s > P/2
        return -1
    end

    w = (P-1)/(binomial(P,s)*s*(P-s))
    minimum(x)*w
end
test_subsets(members, subsetValue, ESValues.orderedsubsets(members, subsetValue))


# test orderedsubsets ascending using a maximum function
members = [4.19897,0.0835427,0.0452934,1.77052,0.0945746,4.88804,0.135961]
test_subsets(members, maximum, ESValues.orderedsubsets(members, maximum, :ascending), :ascending)

# test orderedsubsets ascending using a sum function
members = [4.19897,0.0835427,0.0452934,1.77052,0.0945746,4.88804,0.135961]
test_subsets(members, sum, ESValues.orderedsubsets(members, sum, :ascending), :ascending)


# test eskernelsubsets using a value that rejects subsets larger than a given value
members = collect(1:P)
variances = [4.19897,0.0835427,0.0452934,1.77052,0.0945746,4.88804,0.135961]
function subsetValue2(x)
    x = convert(Vector{Int64}, x[1])
    s = length(x)
    if s == P || s == 0
        return 1e12
    elseif s > P/2
        return -1
    elseif s == 1
        return 1e10 + minimum(variances[x])
    end

    w = (P-1)/(binomial(P,s)*s*(P-s))
    minimum(variances[x])*w
end
test_subsets(members, subsetValue2, ESValues.eskernelsubsets(members, variances))
