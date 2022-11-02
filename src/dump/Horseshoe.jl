module Horseshoe

using Distributions
#using StatsBase
#using SpecialFunctions
using LinearAlgebra
using LinearAlgebra: I
using Distributions: rand
using ProgressMeter
using Krylov: lsqr
"""
This module includes sparse prior models with costume build samplers. 
"""

abstract type OutputScheduler end
abstract type Sampler end
abstract type HorseShoe <: Sampler end


include("horseshoe_utils.jl")
include("horseshoe_exact.jl")
include("horseshoe_approx.jl")
include("horseshoe_io.jl")

function init!(sampler::Sampler, param0=nothing)
    if param0 !== nothing
        for p in param0
            sampler.p = params0.p
        end
    end
end

"""
    run!(outp::OutputScheduler, sampler; param0 = nothing)
TBW
"""
function run!(outp::OutputScheduler, sampler::Sampler; param0 = nothing)
    init!(sampler, param0)
    @showprogress for t = 1:(outp.Nsamples * outp.thinning)
        step!(sampler)
        feed!(outp, sampler, t)
    end
    return outp
end

"""
Functions shared among all Horseshoe sampler implementations
"""

function gibbs_σ2!(hs::HorseShoe)
    if hs.sσ2 < 0
        N = size(hs.W, 1)
        hs.σ2 = 1/rand(Gamma((hs.ω + N)/2.0,2.0/(hs.ω + hs.z_Minv_z)));
    else
        a0, b0 = hs.ω, hs.ω
        ssr = hs.z_Minv_z;
        prop_k = exp(rand(Normal(log(1.0/hs.σ2), hs.sσ2)));
        l_prop = (hs.N+a0-2.0)/2.0*log(prop_k)-prop_k*(ssr+b0)/2.0;
        l_curr = (hs.N+a0-2.0)/2.0*log(1.0./hs.σ2)-(1.0/hs.σ2)*(ssr+b0)/2;
        l_ar = (l_prop-l_curr) + (log(prop_k)-log(1.0/hs.σ2));
        l_ar = l_ar;
        acc_s = rand()<exp(l_ar);
        if acc_s
            hs.σ2 = 1/prop_k;
        end
    end
    #rand(Distributions.InverseGamma( .5*(hs.ω + hs.N), .5*(hs.ω + hs.z_Minv_z) ))
end

function log_p_ξ_given_η( ξ::T, logdetM::T, z_Minv_z::T, ω::T, N::Int) where {T <: Real}
    ll = -.5 * logdetM - .5*(N+ω)* log(.5*ω + .5 * z_Minv_z) 
    pr = - log(sqrt(ξ)*(1+ξ))
    return ll + pr
end

function lmh_ratio(y::Array{Float64,2},x::Array{Float64,2},Xi::Float64,cM::LinearAlgebra.Cholesky{Float64,Array{Float64,2}},n::Int64,a0::Float64,b0::Float64)
    ssr = y'*x+b0; ssr = ssr[1];
    try
        ldetM = 2.0*logdet(cM);
        ll = -.5 *ldetM - ((n+a0)/2).*log(ssr);
        lpr = -log(sqrt(Xi).*(1+Xi));
        lr = ll+lpr;
    catch
        lr = -Inf; info("proposal was rejected because I+XDX was not positive-definite");
    end
end 

end # module
