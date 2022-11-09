

mutable struct HorseShoeApprox{T} <: HorseShoe
    # Observations  / data
    W::AbstractMatrix{T}    # Design matrix
    z::AbstractVector{T}    # Response vector
    N::Int                  # Number of Observations (or mini-batch size if subsampling is employed)
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
    WW::AbstractMatrix{T}   # Matrix W' * W
    #M::AbstractMatrix{T}   # Matrix M = I + ξ^-1 * W' * Diagonal(1/η) * W
    # dependent variables 2: depend on η and ξ
    cM                      # Cholesky decomposition of M 
    cA
    z_Minv_z::T             # value of z' * inv(M) * z
    ηmin::T
    sσ2::T
    woodbury::Bool
    δ::T 
    pδ::Int
    Ws::AbstractMatrix{T}   # Thresholded Design matrix W
    WWs::AbstractMatrix{T}  # Thresholded version of W'* W
    ηs::AbstractVector{T}   # Thresholded Local precision parameters
    βs::AbstractVector{T}   # Thresholded regression coefficients 
    decomp_type_w::Symbol
    decomp_type_nw::Symbol
    use_woodbury::Bool
end
#    val::AbstractArray{<:Union{Missing,Real},3},


function HorseShoeApprox(W,z; β=nothing, η = nothing, ξ=1.0, σ2 = 1.0, ω=1.0, s=1.0, ηmin=0.0, sσ2=-1.0, δ=1E-4, decomp_type_w=:qr, decomp_type_nw=:qr,use_woodbury=false) 
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
    WW =  W' * W
    # M = Symmetric(I +  hs.WDW / hs.ξ)
    # Mchol = cholesky(Symmetric(I +  WDW / ξ))
    #z_Minv_z = dot(z, Mchol \ z)
    cM = nothing
    cA = nothing
    z_Minv_z = -1.0
    pδ = p
    Ws = @view W[:,:]
    WWs = @view WW[:,:]
    ηs = @view η[:]
    βs = @view β[:]
    # hs = HorseShoeApprox(W, z, N, p, β, η, ξ, σ2, ω, s, DW1, WDW, WW, cM, cA, z_Minv_z, ηmin, sσ2, false, δ, pδ, Ws, WWs, ηs, βs) 
    # update_approx!(hs, hs.ξ)
    # update_decomps!(hs)
    return HorseShoeApprox(W, z, N, p, β, η, ξ, σ2, ω, s, DW1, WDW, WW, cM, cA, z_Minv_z, ηmin, sσ2, false, δ, pδ, Ws, WWs, ηs, βs,decomp_type_w,decomp_type_nw,use_woodbury) 
end

function update_approx!(hs::HorseShoeApprox, ξmin::T ) where {T<:Real}
    mask = (1.0./(hs.η*ξmin)) .> hs.δ;
    hs.pδ = sum(mask); # Rank of WDσW'
    hs.ηs = @view hs.η[mask]; 
    hs.Ws = @view hs.W[:,mask];
    hs.WWs = @view hs.WW[mask,mask];
    hs.βs = @view hs.β[mask]
    #hs.woodbury = (hs.pδ < hs.N/2) && hs.use_woodbury
end

function update_decomps!(hs::HorseShoeApprox)
    if hs.woodbury
        hs.cA = woodbury_decomp_update(hs.Ws, hs.ηs, hs.ξ)
        hs.cM = nothing 
    else
        hs.cA = nothing
        hs.cM = regular_decomp_update(hs.Ws, hs.ηs, hs.ξ)
    end  
end

function update_z_Minv_z!(hs::HorseShoeApprox)
    hs.z_Minv_z = dot(hs.z, hs.Minv(hs.z))
end

function Minv(hs::HorseShoeApprox, x::Vector{T}) where {T<:Real}
    """
    Computes the action of z ↦ M^{-1} z. Depending on the rank of Dδ either the Woodbury identity 
    M^{-1} = I - Ws (ξ η + Ws' * Ws)^(-1) Ws'
    or the standard formual 
    """
    if hs.woodbury
        y = hs.cA \ (transpose(hs.Ws)*x)
        return x - hs.Ws * y 
    else
        return hs.cM \ x 
    end
end


function woodbury_decomp_update(Ws::AbstractMatrix{T}, ηs::AbstractVector{T}, ξ::T) where {T<:Real}
    WWs = transpose(Ws) * Ws
    return woodbury_decomp_update(Ws, ηs, ξ, WWs)
end
function woodbury_decomp_update(::AbstractMatrix{T}, ηs::AbstractVector{T}, ξ::T, WWs::AbstractArray) where {T<:Real}
    return cholesky(Diagonal(ξ * ηs) + WWs) 
end

function regular_decomp_update(Ws::AbstractMatrix{T}, ηs::AbstractVector{T}, ξ::T) where {T<:Real}
    D1Ws = Diagonal(1.0./sqrt.(hs.ηs)) * transpose(hs.Ws)
    WDWs = transpose(D1Ws) * D1Ws
    return regular_decomp_update(Ws, ηs, ξ, WDWs)
end

function regular_decomp_update(::AbstractMatrix{T}, ::AbstractVector{T}, ξ::T, WDWs::AbstractArray) where {T<:Real}
    return cholesky(Symmetric(I +  WDWs / ξ) ) 
end


function step!(hs::HorseShoeApprox)
    gibbs_ξ!(hs)
    #update_decomps!(hs)
    gibbs_σ2!(hs)
    gibbs_β!(hs)
    gibbs_η!(hs; ηmin = hs.ηmin)
end

function decomp(A, symb::Symbol)
    return decomp(Symmetric(A), Val(symb) )
end
function decomp(A::Symmetric, ::Val{:qr} )
    return qr(A)
end
function decomp(A::Symmetric, ::Val{:svd} )
    return svd(A)
end
function decomp(A::Symmetric, ::Val{:cholesky} )
    return cholesky(A)
end
function decomp(A::Symmetric, ::Val{:auto} )
    return factorize(A)
end
function Horseshoe.gibbs_ξ!(hs::HorseShoeApprox)
    """
    - No requirement for cA and cM and depedent variables to be up-to-date
    - Update ends with all values cA, cM and z_Minv_z consistent with values of η and ξ
    """
    ξ_prop  = exp(rand(Normal(log(hs.ξ),hs.s))); #exp(log(hs.ξ) + hs.s * randn())
    ξ_min = min(hs.ξ, ξ_prop)
    update_approx!(hs, ξ_min)
    # Todo: 1) precomputed decompositions? 2) Replace Woodbury identity by SVD
    if hs.woodbury
        #WWs = transpose(hs.Ws)*hs.Ws
        # use Sylvester identy to speed up computation of determinant
        WDs  = hs.Ws * Diagonal(1.0./sqrt.(hs.ηs))
        sDWDs = svd(WDs'*WDs); 
        ldetM = sum( log.(1.0.+ (sDWDs.S)/hs.ξ))
        ldetM_prop = sum( log.(1.0.+ (sDWDs.S)/ξ_prop))

        # 
        cA  = decomp(Diagonal(hs.ξ * hs.ηs) + hs.WWs, hs.decomp_type_w)
        z_Minv_z = dot(hs.z, hs.z - hs.Ws * (cA \ (transpose(hs.Ws)*hs.z))) # 2) use SVD result instead
        
    
        cA_prop  = decomp(Diagonal(ξ_prop * hs.ηs) + hs.WWs, hs.decomp_type_w)
        z_Minv_z_prop = dot(hs.z, hs.z - hs.Ws * (cA_prop \ (transpose(hs.Ws)*hs.z)))
        
    else
        D1Ws = Diagonal(1.0./sqrt.(hs.ηs)) * transpose(hs.Ws)
        WDWs = transpose(D1Ws) * D1Ws

        cM = decomp(Symmetric(I +  WDWs / hs.ξ) , hs.decomp_type_nw)
        z_Minv_z = dot(hs.z, cM \ hs.z)
        ldetM = logdet(cM)

        cM_prop = decomp(Symmetric(I +  WDWs / ξ_prop) , hs.decomp_type_nw)
        z_Minv_z_prop = dot(hs.z, cM_prop \ hs.z)
        ldetM_prop = logdet(cM_prop)
    end

    log_p_prop = log_p_ξ_given_η(ξ_prop, ldetM_prop, z_Minv_z_prop, hs.ω, hs.N)
    log_p_curr = log_p_ξ_given_η( hs.ξ, ldetM, z_Minv_z, hs.ω, hs.N)
    log_p_acc  = (log_p_prop - log_p_curr ) + (log(ξ_prop) - log(hs.ξ)) 

    if rand() < exp(log_p_acc)
        # update ξ
        hs.ξ = ξ_prop
        # update dependent values
        hs.cA = (hs.woodbury ? cA_prop : nothing)
        hs.cM = (hs.woodbury ? nothing : cM_prop)
        hs.z_Minv_z = z_Minv_z_prop
    else
        # Remove ? Depends on whether update δ-ξ dependencies.
        hs.cA = (hs.woodbury ? cA : nothing)
        hs.cM = (hs.woodbury ? nothing : cM)
        hs.z_Minv_z = z_Minv_z
    end
end

function gibbs_β!(hs::HorseShoeApprox)
    N = hs.W
    u = 1.0./sqrt.(hs.η*hs.ξ) .* randn(hs.p)
    v = hs.W * u + randn(hs.N)
    σ = sqrt(hs.σ2)
    #WDW = comp_WDW(hs.W, hs.η) # doesn't need to recomputed if hs.WDW up-to-date
    #M = Symmetric(I +  WDW / hs.ξ) # should be up to date? If yes, use hs.M instead 
    #Mchol = cholesky(M)
    vstar = Minv(hs, hs.z/σ - v) 
    hs.β[:] = σ * u
    hs.βs[:] += σ * Diagonal(1.0./(hs.ηs*hs.ξ)) * transpose(hs.Ws) * vstar 
end


function gibbs_η!(hs::HorseShoeApprox;  ηmin = 0)
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
    # Todo: 1) precomputed decompositions? 2) Replace Woodbury identity by SVD
end