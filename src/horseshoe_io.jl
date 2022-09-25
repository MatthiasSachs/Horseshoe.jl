
using MCMCChains


"""
Outpuscheduler:
"""

mutable struct VarOutput <: OutputScheduler
    vals::Array{Float64,3}
    params_list::Array{Symbol}
    params_range::Dict{Symbol,UnitRange{Int64}}
    Nsamples::Int
    thinning::Int    
end

function VarOutput(model, params_list::Vector{Symbol}, Nsamples::Int; thinning = 1, nchains=1)
    first = 1
    last = -1
    params_range = Dict{Symbol,UnitRange{Int64}}([])
    for s in params_list
        last = first + length(getfield(model,s)) - 1
        params_range[s] = first:last
        first = last + 1
    end
    vals = zeros(Nsamples, last, nchains)
    #Chains(vals, params_list, params_range, Nsamples, thinning)
    return VarOutput(vals, params_list, params_range,Nsamples, thinning)
end

function feed!(outp::VarOutput, sampler,  t::Int;kchain=1)
    if t % outp.thinning == 0
        for s in outp.params_list
            outp.vals[t ÷ outp.thinning, outp.params_range[s], kchain] .= getfield(sampler,s)
        end
    end
end

function get_var(outp::VarOutput, s::Symbol; kchain=1)
    @assert s in outp.params_list
    return outp.vals[:, outp.params_range[s],kchain]
end

function Base.getindex(outp::VarOutput, s::Symbol)
    #; kchain=1
    @assert s in outp.params_list
    return outp.vals[:, outp.params_range[s],:]
end

"""
Interface to MCMCChains
"""
function MCMCChains.Chains(outp::OutputScheduler; params_list=nothing)
    

    if params_list === nothing
        params_list = outp.params_list
        export_range = outp.params_range
    else
        export_range = Dict{Symbol,UnitRange{Int64}}([])
        first, last = 1, -1
        for s in params_list
            @assert s in outp.params_list
            srange = outp.params_range[s]
            last = first + length(srange) - 1
            export_range[s] = first:last
            first = last + 1
        end 
    end
    nparams = sum(length(srange) for srange in values(export_range))
    
    Nsamples, nchains = size(outp.vals,1), size(outp.vals,3) 
    
    # generate symbols of variable names for output
    names = Symbol[]
    for s in params_list
        append!(names, Symbol.(s, 1:length(export_range[s])))
    end

    vals = zeros(Nsamples, nparams, nchains)
    for s in params_list
        vals[:,export_range[s],:] = outp.vals[:,outp.params_range[s],:] 
    end

    return Chains(vals, names)

end

