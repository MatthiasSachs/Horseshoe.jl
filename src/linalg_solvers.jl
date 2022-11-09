
module MLinAlg

using LinearAlgebra
using Krylov

export update!, Minv, accept!, lsqr_params, logdet

abstract type MSolver end


mutable struct Woodbury <: MSolver
    fA
    fA_prop
    Ws
    decomp_type
end
Woodbury(params::Dict) =  Woodbury(nothing, nothing, nothing, params)
Woodbury() = Woodbury(nothing, nothing, nothing, :qr)

# function Woodbury(hs::HorseShoe, ξ, ξ_prop, params) 
#     decomp_type = params[:decomp_type]
#     Ws = hs.Ws
#     fA  = decomp(Diagonal(ξ * hs.ηs) + hs.WWs, decomp_type)
#     fA_prop  = decomp(Diagonal(ξ_prop * hs.ηs) + hs.WWs, decomp_type)
#     return Woodbury(fA,fA_prop, Ws, decomp_type)
# end

function update!(solver::Woodbury, hs, ξ, ξ_prop)
    solver.Ws = hs.Ws
    solver.fA  = decomp(Diagonal(ξ * hs.ηs) + solver.WWs, solver.decomp_type)
    solver.fA_prop  = decomp(Diagonal(ξ_prop * hs.ηs) + hs.WWs, solver.decomp_type)
end

function accept!(solver::Woodbury)
    solver.fa = solver.fa_prop
end

function Minv(solver::Woodbury, x::AbstractVector{T}, prop=false) where {T<:Real}
    fA = (prop ? solver.fA_prop : solver.fA)
    y = fA \ (transpose(solver.Ws)*x)
    return x - solver.Ws * y 
end

mutable struct Cholesky <: MSolver
    cM
    cM_prop
    decomp_type::Symbol
end


Cholesky(decomp_type::Symbol) = Cholesky(nothing, nothing, decomp_type)
Cholesky(params_solver::Dict) =  Cholesky(Symbol(params_solver["decomp_type"]))
Cholesky() = Cholesky(:cholesky)
# function Cholesky(hs::HorseShoe, ξ, ξ_prop, params) 
#     decomp_type = params[:decomp_type]
#     D1Ws = Diagonal(1.0./sqrt.(hs.ηs)) * transpose(hs.Ws)
#     WDWs = transpose(D1Ws) * D1Ws

#     cM = decomp(Symmetric(I +  WDWs / ξ), decomp_type)
#     cM_prop = decomp(Symmetric(I +  WDWs / ξ_prop), decomp_type)
#     return Cholesky(cM, cM_prop, decomp_type)
# end

function update!(solver::Cholesky, hs, ξ, ξ_prop)
    D1Ws = Diagonal(1.0./sqrt.(hs.ηs)) * transpose(hs.Ws)
    WDWs = transpose(D1Ws) * D1Ws

    solver.cM = decomp(Symmetric(I +  WDWs / ξ) , solver.decomp_type)
    solver.cM_prop = decomp(Symmetric(I +  WDWs / ξ_prop) , solver.decomp_type)
end

function accept!(solver::Cholesky)
    solver.cM = solver.cM_prop
end

function Minv(solver::Cholesky, x::AbstractVector{T}, prop=false) where {T<:Real}
    return (prop ? solver.cM_prop \ x  : solver.cM \ x)
end

function logdet(solver::Cholesky, prop=true)
    return (prop ? LinearAlgebra.logdet(solver.cM_prop) : LinearAlgebra.logdet(solver.cM))
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

abstract type KrylovSolver end

function update!(solver::KrylovSolver, hs, ξ, ξ_prop)
    solver.Ws = hs.Ws
    solver.Dsqrts = Diagonal(1.0 ./sqrt.(ξ * hs.ηs))
    solver.Dsqrts_prop = Diagonal(1.0 ./sqrt.(ξ_prop * hs.ηs))
end

function accept!(solver::KrylovSolver)
    solver.Dsqrts= solver.Dsqrts_prop
end

function lsqr_params(; axtol=T->10E-16, btol=T->√eps(T), atol=T->√eps(T), rtol=T->√zero(T),
    etol=T->√zero(T), window=5, conlim=T->10E20/eps(T), itmax=n->2*n, radius=0.0)
    params = Dict(
    :axtol => axtol, :btol => btol,
    :atol => atol, :rtol => rtol, :conlim => conlim, :itmax => itmax,
    :etol => etol, :window => window,:radius => radius)
    return params
end

mutable struct LSQR <: KrylovSolver
    params
    runinfo
    Ws
    Dsqrts
    Dsqrts_prop
end
LSQR(params::Dict) = LSQR(params, Dict(), nothing, nothing, nothing)
LSQR() =  LSQR(lsqr_params())
#LSQR(; args...) = LSQR(lsqr_params(;args...), nothing, nothing, nothing, nothing)

# function LSQR(hs::HorseShoe, ξ, ξ_prop, params) 
#     return LSQR(params, nothing, 
#         hs.Ws, 
#         Diagonal(1.0 ./sqrt.(ξ * hs.ηs)),
#         Diagonal(1.0 ./sqrt.(ξ_prop * hs.ηs)))
# end


function Minv(solver::LSQR, x::AbstractVector{T}, prop=false) where {T<:Real}
    params = solver.params
    Dsqrts = (prop ? solver.Dsqrts_prop : solver.Dsqrts)
    (x_lsqr, stats) = Krylov.lsqr(solver.Ws * Dsqrts, x,
                        M=I, N=I, sqd=false, λ=one(T),
                        axtol=params[:axtol](T), 
                        btol=params[:btol](T),
                        atol=params[:atol](T), 
                        rtol=params[:rtol](T),
                        etol=params[:etol](T),
                        window=params[:window],
                        itmax=params[:itmax](length(x)), 
                        conlim=params[:conlim](T),
                        radius=T(params[:radius]), 
                        verbose=0, 
                        history=true,
                        ldiv=false, 
                        callback=solver->false)
    solver.runinfo[(prop ? :prop : :current)] = stats
    return x - solver.Ws * Dsqrts  * x_lsqr
end

mutable struct LSMR <: KrylovSolver
    params
    runinfo
    Ws
    Dsqrts
    Dsqrts_prop
end
LSMR(params::Dict) = LSMR(params, Dict(), nothing, nothing, nothing)
LSMR() =  LSMR(lsmr_params())

function lsmr_params(; axtol=T->10E-16, btol=T->√eps(T), atol=T->√eps(T), rtol=T->√zero(T),
    etol=T->√zero(T), window=5, conlim=T->10E20/eps(T), itmax=n->2*n, radius=0.0)
    params = Dict(
    :axtol => axtol, :btol => btol,
    :atol => atol, :rtol => rtol, :conlim => conlim, :itmax => itmax,
    :etol => etol, :window => window,:radius => radius)
    return params
end

function Minv(solver::LSMR, x::AbstractVector{T}, prop=false) where {T<:Real}
    params = solver.params
    Dsqrts = (prop ? solver.Dsqrts_prop : solver.Dsqrts)
    (x_lsqr, stats) = Krylov.lsmr(solver.Ws * Dsqrts, x,
                        M=I, N=I, sqd=false, λ=one(T),
                        axtol=params[:axtol](T), 
                        btol=params[:btol](T),
                        atol=params[:atol](T), 
                        rtol=params[:rtol](T),
                        etol=params[:etol](T),
                        window=params[:window],
                        itmax=params[:itmax](length(x)), 
                        conlim=params[:conlim](T),
                        radius=T(params[:radius]), 
                        verbose=0, 
                        history=true,
                        ldiv=false, 
                        callback=solver->false)
    solver.runinfo[(prop ? :prop : :current)] = stats
    return x - solver.Ws * Dsqrts  * x_lsqr
end


mutable struct QR <: MSolver
    qA
    qA_prop
    Ws
    Dsqrts
    Dsqrts_prop
end
QR(::Dict) = QR()
QR() = QR(nothing, nothing, nothing, nothing, nothing)
# function QR(hs::HorseShoe, ξ, ξ_prop, params)
#     Ws = hs.Ws

#     Dsqrts = Diagonal(1.0 ./sqrt.(ξ * hs.ηs))
#     qA = qr( cat(Ws * Dsqrts, Diagonal(ones(ps)), dims=1) )

#     Dsqrts_prop = Diagonal(1.0 ./sqrt.(ξ_prop * hs.ηs))
#     qA_prop = qr( cat(Ws * Dsqrts_prop, Diagonal(ones(ps)), dims=1) )
#     return QR(qA, qA_prop, hs.Ws, Dsqrts, Dsqrts_prop)
# end

function update!(solver::QR, hs, ξ, ξ_prop)
    ps = length(hs.ηs)
    solver.Ws = hs.Ws

    solver.Dsqrts = Diagonal(1.0 ./sqrt.(ξ * hs.ηs))
    solver.qA = qr( cat(solver.Ws * solver.Dsqrts, Diagonal(ones(ps)), dims=1) )

    solver.Dsqrts_prop = Diagonal(1.0 ./sqrt.(ξ_prop * hs.ηs))
    solver.qA_prop = qr( cat(solver.Ws * solver.Dsqrts_prop, Diagonal(ones(ps)), dims=1) )
end

function accept!(solver::QR)
    solver.qA = solver.qA_prop
    solver.Dsqrts= solver.Dsqrts_prop
end

function Minv(solver::QR, x::AbstractVector{T}, prop=false) where {T<:Real}
    qA = (prop ? solver.qA_prop : solver.qA)
    Dsqrts = (prop ? solver.Dsqrts_prop : solver.Dsqrts)
    xp = cat(x, zeros(size(solver.Ws,2)), dims=1)
    x_qr = qA \ xp
    return x - solver.Ws * Dsqrts  * x_qr
end

end