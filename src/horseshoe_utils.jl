
using StatsBase

function sample_h(ε::T; a=.5, b=1.5, maxattempts = 10^3) where {T<:Real} # todo: include warning for too many attempts
    """
    sample from hL(x) ∝ 1/(1+x)  e^(-ε x)
    """
    λ, ν = zeros(2), zeros(4)
    # C = exp(-ε) * expinti(ε)
    f = x -> x * ε + log(1+x)
    A, J, B = f(a/ε), f(1/ε), f(b/ε)
    λ[1] = (J-A) / ( (1-a) / ε)     # λ[1]= λ₂
    λ[2] = (B-J)/((b-1)/ε)          # λ[2]= λ₃

    ν[1] = log(1+a/ε)
    ν[2] = λ[1]^-1  * exp(-A) * (1- exp(A-J)) 
    ν[3] = λ[2]^-1  * exp(-J) * (1- exp(J-B)) 
    ν[4] = ε^-1 * exp(-B)

    z = 0.0
    attempts = 1
    while(attempts < maxattempts)
        i = sample([1,2,3,4], Weights( ν/sum(ν)))
        if i == 1
            at = 0 
            #while(z<=ηmin && at < 1E2)
            z = (1+a/ε)^rand() - 1 
            #    at+=1 
            #end
            #if at == 1E2
            #    error("too many rejects 2 ", ε )
            #end
            #@show exp( log(1+z) - f(z) )
            if rand() < exp( log(1+z) - f(z) )
                break
            end
        elseif i == 2
            z = rand(Expo(λ[1], a/ε, 1/ε))
            #@show exp( (A+λ[1]*(z-a/ε)) - f(z) )
            if rand() < exp( (A+λ[1]*(z-a/ε)) - f(z) )
                break
            end
        elseif i == 3
            z = rand(Expo(λ[2], 1/ε, b/ε))
            #@show exp( (J + λ[2]*(z-b/ε) ) - f(z) )
            if rand() < exp( (J + λ[2]*(z-b/ε) ) - f(z) )
                break
            end
        elseif i == 4
            z = rand(Expo(ε, b/ε, Inf))
            #@show exp( (B + ε * (z - b/ε)) - f(z) )
            if rand() < exp( (B + ε * (z - b/ε)) - f(z) )
                break
            end
        end
        attempts+=1
    end
    if attempts >= maxattempts
        error("No accepted sample")
    end
    #@show z
    return z
end


struct ExpCauch{T}
    Minv::T # M = exponential decay rate
    Z::T
end

function Distributions.pdf(d::ExpCauch, x::Core.Real)
    if d.Z < 0
        d.Z = -exp(M)*expinti(-M)
    end
    return 1/(1+x) * exp(-d.M*x)/d.Z
end


ExpCauch(M::T) where {T<: Real} = ExpCauch(1.0/M,-1.0)

function Distributions.rand(d::ExpCauch)
    d_upper = Exponential(d.Minv)
    x = 0.0
    while(true)
        x = rand(d_upper)
        if rand() <= 1/(1+x) 
            break
        end
    end
    return x
end

struct Expo{T}
    λ::T
    ν̲::T
    ν̅::T
    H::T
end



Expo(λ::T, ν̲::T, ν̅::T) where {T<: Real} = Expo(λ, ν̲, ν̅, 1 - exp(-λ*(ν̅- ν̲)))

function Distributions.rand(d::Expo)
    u = rand()
    return d.ν̲ - log(1 - u * d.H)/d.λ
end