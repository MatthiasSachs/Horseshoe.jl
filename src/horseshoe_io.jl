
using MCMCChains

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

