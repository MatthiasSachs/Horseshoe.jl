


function create_msolver((params_solver::Dict))
    return Horseshoe.MLinAlg.eval(Symbol(params_solver["type"]))(params_solver["params"]) 
end


function create_HorshoeSampler(params::Dict )
    pkeys = keys(params)
    eta_min = "eta_min" in pkeys ? params["eta_min"] : 1E-8
    delta = "variance_threshold" in pkeys ? params["variance_threshold"] : 1E-3
    s_sigma2 = "s_sigma2" in pkeys ? params["s_sigma2"] : -1.0
    p_factor = "p_factor" in pkeys ? params["p_factor"] : 2
    solver_N = "solver_N" in pkeys ? create_msolver(params["solver_N"]) : create_msolver(Dict("type" => "Cholesky", "decomp_type"=> "auto", "params" =>nothing))
    solver_p = "solver_p" in pkeys ? create_msolver(params["solver_p"]) : create_msolver(Dict("type" => "LSQR", "params"=> Horseshoe.lsqr_params()))
    s_xi = "s_xi" in pkeys ? params["s_xi"] : 1.0
    xi_fixed = "xi_fixed" in pkeys ? params["xi_fixed"] : false
    sigma2_fixed = "sigma2_fixed" in pkeys ? params["sigma2_fixed"] : false
    omega = "omega" in pkeys ? params["omega"] : 1.0
    W = "W" in pkeys ? params["W"] : nothing   
    z = "z" in pkeys ? params["z"] : nothing 
    database = "database" in pkeys ? params["database"] : nothing  
    return HorseShoeApprox(; W=W, z=z, database=database, 
        β=nothing, η = nothing, ξ=1.0, σ2 = 1.0, 
        ξ_fixed=xi_fixed, σ2_fixed=sigma2_fixed,
        sξ=s_xi, sσ2=s_sigma2,
        ω=omega, ηmin=eta_min, 
        δ=delta, 
        p_factor=p_factor, solver_p=solver_p, solver_N=solver_N
    ) 
end




