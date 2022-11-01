
mutable struct MWoodbury <: MSolver
    fA
    fA_prop
    Ws
    decomp_type
end
MWoodbury(params::Dict) =  MWoodbury(nothing, nothing, nothing, params)

function MWoodbury(hs::HorseShoeApprox, ξ, ξ_prop, params) 
    decomp_type = params[:decomp_type]
    Ws = hs.Ws
    fA  = decomp(Diagonal(ξ * hs.ηs) + hs.WWs, decomp_type)
    fA_prop  = decomp(Diagonal(ξ_prop * hs.ηs) + hs.WWs, decomp_type)
    return MWoodbury(fA,fA_prop, Ws, decomp_type)
end

function update!(solver::MWoodbury, hs, ξ, ξ_prop)
    solver.Ws = hs.Ws
    solver.fA  = decomp(Diagonal(ξ * hs.ηs) + hs.WWs, solver.decomp_type)
    solver.fA_prop  = decomp(Diagonal(ξ_prop * hs.ηs) + hs.WWs, solver.decomp_type)
end

function accept!(solver::MWoodbury)
    solver.fa = solver.fa_prop
end

function Minv(solver::MWoodbury, x::AbstractVector{T}, prop=false) where {T<:Real}
    fA = (prop ? solver.fA_prop : solver.fA)
    y = fA \ (transpose(solver.Ws)*x)
    return x - solver.Ws * y 
end

mutable struct Mcholesky <: MSolver
    cM
    cM_prop
    decomp_type::Symbol
end

Mcholesky(params_solver::Dict) =  Mcholesky(nothing, nothing, params_solver[:decomp_type])

function Mcholesky(hs::HorseShoeApprox, ξ, ξ_prop, params) 
    decomp_type = params[:decomp_type]
    D1Ws = Diagonal(1.0./sqrt.(hs.ηs)) * transpose(hs.Ws)
    WDWs = transpose(D1Ws) * D1Ws

    cM = decomp(Symmetric(I +  WDWs / ξ), decomp_type)
    cM_prop = decomp(Symmetric(I +  WDWs / ξ_prop), decomp_type)
    return Mcholesky(cM, cM_prop, decomp_type)
end

function update!(solver::Mcholesky, hs, ξ, ξ_prop)
    D1Ws = Diagonal(1.0./sqrt.(hs.ηs)) * transpose(hs.Ws)
    WDWs = transpose(D1Ws) * D1Ws

    solver.cM = decomp(Symmetric(I +  WDWs / ξ) , solver.decomp_type)
    solver.cM_prop = decomp(Symmetric(I +  WDWs / ξ_prop) , solver.decomp_type)
end

function accept!(solver::Mcholesky)
    solver.cM = solver.cM_prop
end

function Minv(solver::Mcholesky, x::AbstractVector{T}, prop=false) where {T<:Real}
    return (prop ? solver.cM_prop \ x  : solver.cM \ x)
end

function logdet(solver::Mcholesky, prop=true)
    return (prop ? logdet(solver.cM_prop) : logdet(solver.cM))
end

mutable struct Mlsqr <: MSolver
    params
    runinfo
    Ws
    Dsqrts
    Dsqrts_prop
end
Mlsqr(params::Dict) = Mlsqr(params, nothing, nothing, nothing, nothing)
#Mlsqr(; args...) = Mlsqr(lsqr_params(;args...), nothing, nothing, nothing, nothing)

function Mlsqr(hs::HorseShoeApprox, ξ, ξ_prop, params) 
    return Mlsqr(params, nothing, 
        hs.Ws, 
        Diagonal(1.0 ./sqrt.(ξ * hs.ηs)),
        Diagonal(1.0 ./sqrt.(ξ_prop * hs.ηs)))
end

function lsqr_params(; axtol=T->10E-16, btol=T->√eps(T), atol=T->√eps(T), rtol=√zero(T),
    etol=T->√zero(T), window=5, conlim=T->10E20/eps(T), itmax=n->2*n, radius=T->zeros(T) )
    params = Dict(
    :axtol => axtol, :btol => btol,
    :atol => atol, :rtol => rtol, :conlim => conlim, :itmax => itmax,
    :etol => etol, :window => window,:radius => radius)
    return params
end

function update!(solver::Mlsqr, hs, ξ, ξ_prop)
    solver.Ws = hs.Ws
    solver.Dsqrts = Diagonal(1.0 ./sqrt.(ξ * hs.ηs))
    solver.Dsqrts_prop = Diagonal(1.0 ./sqrt.(ξ_prop * hs.ηs))
end

function accept!(solver::Mlsqr)
    solver.Dsqrts= solver.Dsqrts_prop
end

function Minv(solver::Mlsqr, x::AbstractVector{T}, prop=false) where {T<:Real}
    Dsqrts = (prop ? solver.Dsqrts_prop : solver.Dsqrts)
    (x_lsqr, stats) = lsqr(solver.Ws * Dsqrts, x,
                        M=I, N=I, sqd=false, λ=one(T),
                        axtol=params[:axtol](T), 
                        btol=params[:btol](T),
                        atol=params[:atol](T), 
                        rtol=params[:rtol](T),
                        etol=params[:etol](T),
                        window=params[:window],
                        itmax=params[:itmax](length(x)), 
                        conlim=params[:conlim](T),
                        radius=params[:radius](T), 
                        verbose=0, 
                        history=true,
                        ldiv=false, 
                        callback=solver->false)
    solver.runinfo[(prop ? :prop : :current)] = stats
    return x - solver.Ws * solver.Dsqrts  * x_lsqr
end

mutable struct Mqr <: MSolver
    qA
    qA_prop
    Ws
    Dsqrts
    Dsqrts_prop
end
Mqr(::Dict) = Mqr(nothing, nothing, nothing, nothing, nothing)
function Mqr(hs::HorseShoeApprox, ξ, ξ_prop, params)
    Ws = hs.Ws

    Dsqrts = Diagonal(1.0 ./sqrt.(ξ * hs.ηs))
    qA = qr( cat(Ws * Dsqrts, Diagonal(ones(ps)), dims=1) )

    Dsqrts_prop = Diagonal(1.0 ./sqrt.(ξ_prop * hs.ηs))
    qA_prop = qr( cat(Ws * Dsqrts_prop, Diagonal(ones(ps)), dims=1) )
    return Mqr(qA, qA_prop, hs.Ws, Dsqrts, Dsqrts_prop)
end

function update!(solver::Mqr, hs::HorseShoeApprox, ξ, ξ_prop)
    ps = length(hs.ηs)
    solver.Ws = hs.Ws

    solver.Dsqrts = Diagonal(1.0 ./sqrt.(ξ * hs.ηs))
    solver.qA = qr( cat(solver.Ws * solver.Dsqrts, Diagonal(ones(ps)), dims=1) )

    solver.Dsqrts_prop = Diagonal(1.0 ./sqrt.(ξ_prop * hs.ηs))
    solver.qA_prop = qr( cat(solver.Ws * solver.Dsqrts_prop, Diagonal(ones(ps)), dims=1) )
end

function accept!(solver::Mqr)
    solver.qA = solver.qA_prop
    solver.Dsqrts= solver.Dsqrts_prop
end

function Minv(solver::Mqr, x::AbstractVector{T}, prop=false) where {T<:Real}
    qA = (prop ? solver.qA_prop : solver.qA)
    Dsqrts = (prop ? solver.Dsqrts_prop : solver.Dsqrts)
    xp = cat(x, zeros(size(solver.Ws,2)), dims=1)
    x_qr = qA \ xp
    return x - solver.Ws * Dsqrts  * x_qr
end