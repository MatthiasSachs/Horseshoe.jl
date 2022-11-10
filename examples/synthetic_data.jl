

using Horseshoe
using Horseshoe: HorseShoeApprox, VarOutput, run!, get_var
using Distributions
using Random
using Horseshoe.MLinAlg
using PyPlot

path2plots = "./plots"
rnd_seed = 571;
Random.seed!(rnd_seed)

p = 1000; n = 100;

σ_true = 2.0;
β_true = ones(p)*0.01;
β_true[1:23] = 2.0.^(-range(-2,.25,23));
ξ_true = 1.0;
corX = false

if corX
    X = rand(Normal(0.,1.),n,p);
    for j=2:p
        X[:,j] = rhoX.*X[:,j-1]+X[:,j];
    end
    X = broadcast(*,X,1.0./std(X,1));
else
    X = rand(Normal(),n,p);
end

y = X*β_true+σ_true.*rand(Normal(0.,1.),n);

hs = HorseShoeApprox(;W=X,z=y, 
                β=2.0*randn(p), 
                η = ones(size(β_true)), 
                ξ=ξ_true^(-2),
                σ2 = σ_true^2, 
                ω=1., 
                sξ=.5, 
                δ=0.0,
                solver_p=Horseshoe.MLinAlg.Cholesky()) 

N_samples = 10000
outp = VarOutput(hs, [:β, :η, :ξ, :σ2,:pδ, :N_dominant], N_samples; thinning = 1, nchains=1)
run!(outp, hs; param0 = nothing)

#%%

# for (i,s) in enumerate(outp.params_list)
#     Plots.plot()
#     Plots.plot!(get_var(outp,s)[1:end-1,:]; title = string(s))
#     if s in [:η, :σ2, :ξ]
#       Plots.yaxis!(:log)
#     end
#     Plots.savefig(joinpath(string(path2plots, "/regression_$s.pdf")))
# end


#%%
β_true= β_true

β_traj = get_var(outp, :β; kchain=1)
xi_mean = [mean(β_traj[:,i]) for i in 1:p]
xi_sd = [sqrt.(mean(β_traj[:,i].^2) - xi_mean[i]^2) for i in 1:p]
ci = zeros(p,2)
ci[:,1] = xi_mean-3*xi_sd
ci[:,2] = xi_mean+3*xi_sd;

figfactor = 3
fig = figure(figsize=(figfactor*12,figfactor*3))
for i = 1:p
    PyPlot.plot([i,i], ci[i,:], "b-", lw=0.5)
end
PyPlot.plot(1:p, xi_mean, "ro", markersize=2, label="posterior mean")
PyPlot.plot(1:p, β_true, "gx", markersize=2, label="true parameter")
PyPlot.grid(true)
display(gcf())
#ylim([-20,20])
xlabel("Component index")
ylabel("Value")
title("Confidence intervals for signals")
legend(ncol=2,bbox_to_anchor=(0.65, 1))
PyPlot.savefig(joinpath("", "./plots/regression_1.pdf"), bbox_inches="tight")
#%%
using MCMCChains
chn = Chains(outp, params_list = [:β])

