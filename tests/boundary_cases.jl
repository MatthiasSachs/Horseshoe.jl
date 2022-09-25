
using Random

function synth_data(n,p, σ_true, p_sparsity; rhoX = 0.0, rnd_seed=571)

    Random.seed!(rnd_seed)
    nz = Int(ceil(p * p_sparsity))
    β_true = randn(p)*(σ_true/10);
    if nz > 1
        β_true[1:nz] = 2.0.^(-range(-2,.25,nz));
    else
        β_true[1] = 2.0
    end
    if rhoX !== 0.0
        X = rand(Normal(0.,1.),n,p);
        for j=2:p
            X[:,j] = rhoX.*X[:,j-1]+X[:,j];
        end
        X = broadcast(*,X,1.0./std(X;dims=1));
    else
        X = rand(Normal(),n,p);
    end

    y = X*β_true+σ_true.*rand(Normal(0.,1.),n);

    return X, y
end

collect(range(-2,.25, 2))

σ_true = 2.0;
β_true = randn(p)*(σ_true/10);
β_true[1:23] = 2.0.^(-range(-2,.25,23));
ξ_true = 1.0;
rhoX = 0.0
p_sparsity = 0.1
N_samples = 1000
# n = 2
# p = 100
# X, y = synth_data(n,p, σ_true, p_sparsity; rhoX = rhoX, rnd_seed=571)

# hs = HorseShoeApprox(X,y; 
#     β=2.0*randn(p), 
#     η = ones(p), 
#     ξ=ξ_true^(-2),
#     σ2 = σ_true^2, 
#     ω=1., 
#     s=.5, 
#     δ=0.001,
#     decomp_type_w = :qr,
#     decomp_type_nw = :cholesky) 

# outp = VarOutput(hs, [:β, :η, :ξ, :σ2,:sδ, :woodbury], N_samples; thinning = 1, nchains=1)
# run!(outp, hs; param0 = nothing)

for (n,p) in [(1,100),(100,1), (1,1)]
    @show (n,p)
    X, y = synth_data(n,p, σ_true, p_sparsity; rhoX = rhoX, rnd_seed=571)
    hs = HorseShoeApprox(X,y; 
                β=2.0*randn(p), 
                η = ones(p), 
                ξ=ξ_true^(-2),
                σ2 = σ_true^2, 
                ω=1., 
                s=.5, 
                δ=0.001,
                decomp_type_w = :qr,
                decomp_type_nw = :cholesky) 

    
    outp = VarOutput(hs, [:β, :η, :ξ, :σ2,:sδ, :woodbury], N_samples; thinning = 1, nchains=1)
    run!(outp, hs; param0 = nothing)
end
                
                