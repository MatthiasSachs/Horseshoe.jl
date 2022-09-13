
mutable struct HorseShoeExact{T} <: HorseShoe
    # Observations  / data
    W::AbstractMatrix{T}    # Design matrix
    z::AbstractVector{T}    # Response vector
    N::Int                  # Number of Observations
    p::Int                  # Number of regression coefficients
    # Parameters
    β::AbstractVector{T}    # Regression coefficients
    # Hyper Parameters (sampled)
    η::AbstractVector{T}     # Local precision parameters
    ξ::T                    # Global precision parameter
    σ2::T                   # Variance of observation error 
    # Hyper Parameters (fixed)
    ω::T                    # hyper-parameters for hyper-prior on σ^2: σ^2 ∼ InvGamma(ω/2,ω/2) 
    s::T                    # Variance of proposal in update of ξ
    # dependent variables 1: depend on η
    DW1::AbstractMatrix{T}  # Square root of WDW, i.e., WDW = DW1' * DW1
    WDW::AbstractMatrix{T}  # Matrix WDW =  W' * Diagonal(1/η) * W
    #M::AbstractMatrix{T}    # Matrix M = I + ξ^-1 * W' * Diagonal(1/η) * W
    # dependent variables 2: depend on η and ξ
    Mchol                   # Cholesky decomposition of M 
    z_Minv_z::T             # value of z' * inv(M) * z
    ηmin::T
    sσ2::T
end

function HorseShoeExact(W,z; β=nothing, η = nothing, ξ=1.0, σ2 = 1.0, ω=1.0, s=1.0, ηmin=0.0, sσ2=-1.0) 
    N, p = size(W)
    @assert length(z) == N
    if β === nothing
        β = ones(p)
    end
    if η === nothing
        η = ones(p)
    end
    DW1 = broadcast(*,1.0./sqrt.(η),transpose(W));
    WDW = DW1'* DW1;
    # M = Symmetric(I +  hs.WDW / hs.ξ)
    Mchol = cholesky(Symmetric(I +  WDW / ξ))
    z_Minv_z = dot(z, Mchol \ z)
    return HorseShoeExact(W, z, N, p, β, η, ξ, σ2, ω, s, DW1, WDW, Mchol, z_Minv_z, ηmin, sσ2) 
end

function update_WDW!(hs::HorseShoeExact)
    hs.DW1 = broadcast(*,1.0./sqrt.(hs.η),transpose(hs.W));
    hs.WDW = hs.DW1'* hs.DW1;
end

function update_M!(hs::HorseShoeExact)
    hs.Mchol = cholesky(Symmetric(I +  hs.WDW / hs.ξ))
    hs.z_Minv_z = dot(hs.z, hs.Mchol \ hs.z)
end

function step!(hs::HorseShoeExact)
    gibbs_ξ!(hs)
    gibbs_η!(hs; ηmin = hs.ηmin )
    gibbs_σ2!(hs)
    gibbs_β!(hs)
end


function gibbs_η!(hs::HorseShoeExact;  ηmin = 0)
    # update η
    for j = 1:hs.p
        ε = hs.β[j]^2 * hs.ξ/(2 * hs.σ2)
        if ε == 0.0
            error("rate ε = $ε is 0")
        else
            z = sample_h(ε)
        end
        hs.η[j]= max(z, ηmin)
    end
    # update dependent variables
    update_WDW!(hs)
    update_M!(hs)
end

function gibbs_ξ!(hs::HorseShoeExact)
    ξ_prop  = exp(rand(Normal(log(hs.ξ),hs.s))); #exp(log(hs.ξ) + hs.s * randn())
    
    #println("ξ_prop: ", ξ_prop)
    #WDW = comp_WDW(hs.W, hs.η) # doesn't need to recomputed if hs.WDW up-to-date
    
    #M = Symmetric(I +  WDW / hs.ξ) # should be up to date? If yes, use hs.M instead 
    #Mchol = cholesky(M)
    #z_Minv_z = dot(hs.z, Mchol \ hs.z)

    #M_prop = Symmetric(I +  hs.WDW / ξ_prop) 
    M_propchol = cholesky(Symmetric(I +  hs.WDW / ξ_prop) ) 
    z_M_propinv_z = dot(hs.z, M_propchol \ hs.z)

    log_p_prop = log_p_ξ_given_η(ξ_prop, logdet(M_propchol), z_M_propinv_z, hs.ω, hs.N)
    log_p_curr = log_p_ξ_given_η( hs.ξ, logdet(hs.Mchol), hs.z_Minv_z, hs.ω, hs.N)
    log_p_acc  = (log_p_prop - log_p_curr ) + (log(ξ_prop) - log(hs.ξ)) 

    if rand() < exp(log_p_acc)
        # update ξ
        hs.ξ = ξ_prop
        # update dependent values
        hs.Mchol, hs.z_Minv_z  = M_propchol, z_M_propinv_z
    end
end

function gibbs_β!(hs::HorseShoeExact)
    u = 1.0./sqrt.(hs.η*hs.ξ) .* randn(size(hs.η))
    v = hs.W * u + randn(hs.N)
    σ = sqrt(hs.σ2)
    #WDW = comp_WDW(hs.W, hs.η) # doesn't need to recomputed if hs.WDW up-to-date
    #M = Symmetric(I +  WDW / hs.ξ) # should be up to date? If yes, use hs.M instead 
    #Mchol = cholesky(M)
    vstar = hs.Mchol \ (hs.z/σ - v)
    hs.β = σ * (u + Diagonal(1.0./(hs.η*hs.ξ)) * transpose(hs.W) * vstar )
end