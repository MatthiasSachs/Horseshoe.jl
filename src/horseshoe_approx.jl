
struct InMemoryStorage
    W
    z
    msize
    function InMemoryStorage(W::Matrix,z::Vector, msize)
        @assert size(W,2) == length(z)
        return new(W,z, msize) 
    end
end

n_obs(d::InMemoryStorage) = size(W,1)
n_params(d::InMemoryStorage) = size(W,2)

function subsample(d::InMemoryStorage)
    N = size(X,1)
    perm = [i for i in 1:N] 
    batch = Random.shuffle!(perm)[1:msize]
    return X[batch,:], y[batch]
end


mutable struct HorseShoeApprox{T} <: HorseShoe
    # Observations  / data
    database
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
    z_Minv_z::T             # value of z' * inv(M) * z
    ηmin::T
    sσ2::T
    δ::T 
    sδ::Int
    Ws::AbstractMatrix{T}   # Thresholded Design matrix W
    ηs::AbstractVector{T}   # Thresholded Local precision parameters
    βs::AbstractVector{T}   # Thresholded regression coefficients 
    mask
    solver_N::MSolver       # solver used whe N_dominant == true
    solver_p::MSolver       # solver used whe N_dominant == true
    p_factor::Real
    N_dominant::Bool
    params::Dict
    # function HorseShoeApprox(W::AbstractMatrix{T}, z::AbstractVector{T}, N::Int,p::Int, β::AbstractVector{T}, η::AbstractVector{T}, ξ::T, σ2::T,
    #     ω::T, s::T, z_Minv_z::T, ηmin::T, sσ2::T, δ::T, sδ::Int, Ws::AbstractMatrix{T}, ηs::AbstractVector{T}, βs::AbstractVector{T}, mask,
    #     solver_N::MSolver, solver_p::MSolver, N_dominant::Bool,msize::Int
    #     )
    #     hs = new(W, z, N, p, β, η, ξ, σ2,ω, s, z_Minv_z ηmin, sσ2, δ, sδ, Ws, ηs, βs, mask, solver_N, solver_p, N_dominant)
        
    #     update!(hs.solver, )
    # end
end
#    val::AbstractArray{<:Union{Missing,Real},3},


function HorseShoeApprox(;W=nothing, z=nothing, database=nothing, 
    β=nothing, η = nothing, ξ=1.0, σ2 = 1.0, ω=1.0, 
    s=1.0, ηmin=0.0, sσ2=-1.0, δ=1E-4, 
    params=nothing) 
    if (W !== nothing || z !== nothing) && database !==nothing
        @warn "Either provide a database with optional argument database, or an explicit designnmatrix and obervation vector
        by specifying the two optional arguments W and z."
    elseif database !== nothing 
        W, z = subsample(database)
    end
    N, p = size(W)
    @assert length(z) == N
    if β === nothing
    β = ones(p)
    end
    if η === nothing
    η = ones(p)
    end

    z_Minv_z = -1.0
    sδ = p
    Ws = @view W[:,:]
    ηs = @view η[:]
    βs = @view β[:]

    solver_N = generateMsolver(params[:solver_N])
    solver_p = generateMsolver(params[:solver_p])
    # if (:solver_p in keys(params[:solver_p]) == false
    #     params[:solver_p] = Dict(:solver => :Mcholesky, :decomp_type => :auto)     
    # end
    p_factor = (:p_factor in keys(params) ? params[:p_factor] : 2)
    hs = HorseShoeApprox(database, W, z, N, p, β, η, ξ, σ2,ω, s, z_Minv_z, ηmin, sσ2, δ, sδ, Ws, ηs, βs, nothing, solver_N, solver_p, p_factor, false, params)
    return hs
end


function update_approx!(hs::HorseShoeApprox, ξmin::T ) where {T<:Real}
    mask = (1.0./(hs.η*ξmin)) .> hs.δ;
    hs.sδ = sum(mask); # Rank of WDσW'
    hs.ηs = @view hs.η[mask]; 
    hs.Ws = @view hs.W[:,mask];
    hs.βs = @view hs.β[mask]
    hs.mask = mask
    #hs.woodbury = (hs.sδ < hs.N/2) && hs.use_woodbury
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
    if hs.database !== nothing 
        hs.W, hs.z = subsample(hs.database)
    end
    gibbs_ξ!(hs)
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

function gibbs_ξ!(hs::HorseShoeApprox)
    """
    - No requirement for cA and cM and depedent variables to be up-to-date
    - Update ends with all values cA, cM and z_Minv_z consistent with values of η and ξ
    """
    ξ_prop  = exp(rand(Normal(log(hs.ξ),hs.s))); #exp(log(hs.ξ) + hs.s * randn())
    ξ_min = min(hs.ξ, ξ_prop)
    update_approx!(hs, ξ_min)

    N = size(hs.Ws,1)
    p = length(hs.ηs)

    hs.N_dominant = hs.p_factor * p < N 

    if hs.N_dominant
        update!(hs.solver_N, hs, hs.ξ, ξ_prop)
        WDs  = hs.Ws * Diagonal(1.0./sqrt.(hs.ηs))
        sDWDs = svd(WDs'*WDs); 
        ldetM = sum( log.(1.0.+ (sDWDs.S)/hs.ξ))
        ldetM_prop = sum( log.(1.0.+ (sDWDs.S)/ξ_prop))
        z_Minv_z = dot(hs.z, Minv(hs.solver_N, hs.z, false))
        z_Minv_z_prop = dot(hs.z, Minv(hs.solver_N, hs.z, true))
    else
        update!(hs.solver_p, hs, hs.ξ, ξ_prop)
        ldetM = logdet(hs.solver_p, false)
        ldetM_prop = logdet(hs.solver_p, true)
        z_Minv_z = dot(hs.z, Minv(hs.solver_p, hs.z, false))
        z_Minv_z_prop = dot(hs.z, Minv(hs.solver_p, hs.z, true))
    end

    log_p_prop = log_p_ξ_given_η(ξ_prop, ldetM_prop, z_Minv_z_prop, hs.ω, hs.N)
    log_p_curr = log_p_ξ_given_η( hs.ξ, ldetM, z_Minv_z, hs.ω, hs.N)
    log_p_acc  = (log_p_prop - log_p_curr ) + (log(ξ_prop) - log(hs.ξ)) 

    if rand() < exp(log_p_acc)
        hs.ξ = ξ_prop
        hs.N_dominant ? accept!(hs.solver_N) : accept!(hs.solver_p)
        hs.z_Minv_z = z_Minv_z_prop
    else
        # no need to update decompostion if proposal was rejected
        hs.z_Minv_z = z_Minv_z
    end
end


function gibbs_β!(hs::HorseShoeApprox)
    u = 1.0./sqrt.(hs.η*hs.ξ) .* randn(hs.p)
    v = hs.W * u + randn(hs.N)
    σ = sqrt(hs.σ2)
    solver =  ( hs.N_dominant ? hs.solver_N : hs.solver_p)
    vstar =  Minv(solver, hs.z/σ - v)
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