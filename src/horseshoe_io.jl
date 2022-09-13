
using MCMCChains

function MCMCChains.Chains(outp::OutputScheduler; params_list=nothing)
    if params_list === nothing
        params_list = outp.params_list
    end

    first = 1
    last = -1
    export_range = Dict{Symbol,NamedTuple{(:first, :last), Tuple{Int64, Int64}}}([])
    export_len = Dict{Symbol,Int}([])
    len = 0
    for s in params_list
        @assert s in outp.params_list
        g = outp.params_range[s]
        export_len[s] = (g.last - g.first) + 1
        len += export_len[s]
        last = first + export_len[s]  - 1
        export_range[s] = (first=first,last=last)
        first = last + 1
    end 
    Nsamples, nparams, nchains = size(outp.vals,1), last, size(outp.vals,3) 
    
    names = Symbol[]
    for s in params_list
        append!(names, Symbol.(s, 1:export_len[s]))
    end
    vals = zeros(Nsamples, nparams, nchains)
    for s in params_list
        ex_first, ex_last = export_range[s].first, export_range[s].last
        first, last = outp.params_range[s].first, outp.params_range[s].last
        vals[:,ex_first:ex_last,:] = outp.vals[:,first:last,:] 
    end
    return Chains(vals, names)
    # if section_dict !== nothing
    #     set_section(chn,section_dict)
    # end
end

