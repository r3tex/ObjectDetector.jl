module YOLO
export get_input_size

import ..to, ..AbstractModel, ..get_input_size, ..wrap_model, ..uses_gpu, ..get_cfg

models_dir() = joinpath(@__DIR__, "models")

import Flux
import Flux: gpu, cpu, σ
using LazyArtifacts
using TimerOutputs
using AllocArrays: AllocArray, BumperAllocator
using UnsafeArrays: UnsafeArray

include("nms.jl")

#########################################################
##### FUNCTIONS FOR PARSING CONFIG AND WEIGHT FILES #####
#########################################################
"""
    cfgparse(val::AbstractString)

Convert config String values into native Julia types
not type safe, but not performance critical
"""
function cfgparse(val::AbstractString)
    if all(isletter, val)
        return val::AbstractString
    else
        return occursin('.', val) ? parse(Float64, val) : parse(Int64, val)
    end
end

"""
    cfgsplit(dat::String)

Split config String into a key and value part
split value into array if necessary
"""
function cfgsplit(dat::String)
    name, values = split(dat, '=')
    values = split(values, ',')
    k = Symbol(strip(name))
    v = length(values) == 1 ? cfgparse(values[1]) : cfgparse.(values)
    return (k => v)::Pair{Symbol,<:Any}
end

"""
    cfgread(file::String)

Read config file and return an array of settings
"""
function cfgread(file::String)
    data = reverse(filter(d -> length(d) > 0 && d[1] != '#', readlines(file)))
    out = Array{Pair{Symbol, Dict{Symbol, Any}}, 1}(undef, 0)
    settings = Dict{Symbol, Any}()
    for row in data
        if row[1] == '['
            push!(out, Symbol(row[2:end-1]) => settings)
            settings = Dict{Symbol, Any}()
        else
            push!(settings, cfgsplit(row))
        end
    end
    return reverse(out)::Array{Pair{Symbol, Dict{Symbol, Any}}, 1}
end

"""
    readweights(bytes::IOBuffer, kern::Int, ch::Int, fl::Int, bn::Bool)

Read the YOLO binary weights
"""
function readweights(bytes::Union{IOBuffer,Nothing}, kern::Int, ch::Int, fl::Int, bn::Bool; old_darknet::Bool=false)
    function read_array(io::IOBuffer, n::Int)
        expected = n * sizeof(Float32)
        data = read(io, expected)
        if length(data) < expected && eof(io)
            error("Unexpectedly ran out of data in file. Check weights file corresponds to cfg file. Expected $expected bytes, got $(length(data)).")
        end
        Vector(reinterpret(Float32, data))
    end
    dummy = isnothing(bytes)
    if bn
        if old_darknet
            # PJReddie Darknet: scales, biases, means, vars
            bw = dummy ? ones(Float32, fl) : read_array(bytes, fl)  # weights (scale)
            bb = dummy ? ones(Float32, fl) : read_array(bytes, fl)  # bias
        else
            # AlexeyAB fork: biases, scales, means, vars
            bb = dummy ? ones(Float32, fl) : read_array(bytes, fl)  # bias
            bw = dummy ? ones(Float32, fl) : read_array(bytes, fl)  # weights (scale)
        end
        bm = dummy ? ones(Float32, fl) : read_array(bytes, fl)  # mean
        bv = dummy ? ones(Float32, fl) : read_array(bytes, fl)  # variance
        cb = zeros(Float32, fl)  # conv bias (zero when BN is used)
        cw = dummy ? ones(Float32, kern, kern, ch, fl) : reshape(reinterpret(Float32, read_array(bytes, kern*kern*ch*fl)), kern, kern, ch, fl)
        cw = Float32.(flip(cw))
        if any(<(0), bv)
            @warn "Clipping negative BN variances. This could indicate an issue with the weights file/cfgfile" count_negative = count(<(0), bv)  minval = minimum(bv)
            bv = clamp.(bv, 0.0f0, Inf32)
        end
        return cw, cb, bb, bw, bm, bv
    else
        cb = dummy ? ones(Float32, fl) : read_array(bytes, fl)
        cw = dummy ? ones(Float32, kern, kern, ch, fl) : reshape(reinterpret(Float32, read_array(bytes, kern*kern*ch*fl)), kern, kern, ch, fl)
        cw = Float32.(flip(cw))
        return cw, cb, 0.0, 0.0, 0.0, 0.0
    end
end

########################################################
##### FUNCTIONS NEEDED FOR THE YOLO CONSTRUCTOR ########
########################################################

# These match https://github.com/AlexeyAB/darknet/blob/9ade741db91fd3d796d2abb0c9889b10943ea28a/src/activation_kernels.cu#L28

# Matches: lhtan_activate_kernel
lhtan(x) = x < 0f0 ? 0.001f0 * x :
           x > 1f0 ? 0.001f0 * (x - 1f0) + 1f0 :
                     x

# Matches: lhtan_gradient_kernel
lhtan_grad(x) = (x > 0f0 && x < 1f0) ? 1f0 : 0.001f0

# Matches: hardtan_activate_kernel
hardtan(x) = clamp(x, -1f0, 1f0)

# Matches: linear_activate_kernel
linear(x) = x

# Matches: logistic_activate_kernel
logistic(x) = 1f0 / (1f0 + exp(-x))

# Matches: loggy_activate_kernel
loggy(x) = 2f0 / (1f0 + exp(-x)) - 1f0

# Matches: relu_activate_kernel
relu(x) = x > 0 ? x : zero(x)

# Matches: relu6_activate_kernel
relu6(x) = clamp(x, 0f0, 6f0)

# Matches: elu_activate_kernel
elu(x) = x >= 0 ? x : exp(x) - 1f0

# Matches: selu_activate_kernel
selu(x) = x >= 0 ? 1.0507f0 * x : 1.0507f0 * 1.6732f0 * (exp(x) - 1f0)

# Matches: relie_activate_kernel
relie(x) = x > 0 ? x : 0.01f0 * x

# Matches: ramp_activate_kernel
ramp(x) = x > 0 ? x : 0.1f0 * x

# Matches: leaky_activate_kernel
leaky(x) = x > 0 ? x : 0.1f0 * x

# Matches: tanh_activate_kernel
tanh_activate(x) = 2f0 / (1f0 + exp(-2f0 * x)) - 1f0

# Matches: gelu_activate_kernel (approximation using tanh)
gelu(x) = 0.5f0 * x * (1f0 + tanh(0.797885f0 * x + 0.035677f0 * x^3))

# Matches: softplus_kernel
softplus(x; threshold=20f0) =
    x > threshold ? x :
    x < -threshold ? exp(x) :
    log1p(exp(x))

# Matches: plse_activate_kernel
plse(x) = x < -4f0 ? 0.01f0 * (x + 4f0) :
          x >  4f0 ? 0.01f0 * (x - 4f0) + 1f0 :
          0.125f0 * x + 0.5f0

# Matches: stair_activate_kernel
stair(x) = begin
    n = floor(Int, x)
    iseven(n) ? floor(x / 2f0) : (x - n) + floor(x / 2f0)
end

# Matches: mish_yashas2 (apparently this works better on gpu?)
mish_fast(x) = begin
    e = exp(x)
    n = e * e + 2 * e
    x <= -0.6 ? x * n / (n + 2) : x - 2 * x / (n + 2)
end

# Classical definition
# mish(x) = x * tanh(softplus(x))

# Based on activate_array_swish_kernel
swish(x) = x * σ(x)


# Use this dict to translate the config activation names to function names
const ACT = Dict(
    "linear"    => linear,
    "logistic"  => logistic,
    "loggy"     => loggy,
    "relu"      => relu,
    "relu6"     => relu6,
    "elu"       => elu,
    "selu"      => selu,
    "relie"     => relie,
    "ramp"      => ramp,
    "leaky"     => leaky,
    "tanh"      => tanh_activate,
    "gelu"      => gelu,
    "softplus"  => softplus,
    "mish"      => mish_fast,
    "swish"     => swish,
    "plse"      => plse,
    "stair"     => stair,
    "lhtan"     => lhtan,
    "hardtan"   => hardtan,
)

"""
    prettyprint(str, col)

Provide an array of strings and an array of colors
so the constructor can print what it's doing as it generates the model.
"""
prettyprint(str, col) = for (s, c) in zip(str, col) printstyled(s, color=c) end

"""
    flip(x)
Flip weights to make crosscorelation kernels work using convolution filters
This is only run once when weights are loaded
"""
flip(x) = @view x[end:-1:1, end:-1:1, :, :]

"""
    upsample(a, stride)

Optimized upsampling without indexing for better GPU performance
"""
function upsample(a::AbstractArray, stride)
    m1, n1, o1, p1 = size(a)
    ar = reshape(a, (1, m1, 1, n1, o1, p1))
    b = similar(a, stride, 1, stride, 1, 1, 1)
    b .= 1f0
    return reshape(ar .* b, (m1 * stride, n1 * stride, o1, p1))
end

"""
    reorg(a, stride)

Reshapes feature map - decreases size and increases number of channels, without
changing elements. stride=2 mean that width and height will be decreased by 2
times, and number of channels will be increased by 2x2 = 4 times, so the total
number of element will still the same: width_old*height_old*channels_old = width_new*height_new*channels_new
"""
function reorg(a, stride)
    w, h, c = size(a)
    return reshape(a, (w // stride, h // stride, c*(stride^2)))
end

"""
    overridecfg!(cfgvec::Vector{Pair{Symbol,Dict{Symbol,Any}}}, cfgchanges::Vector{Tuple{Symbol,Int,Symbol,Any}})

Override settings from the YOLO .cfg file, before loading the model.
`cfgchanges` takes the form of a vector of (layer symbol, ith instance of given layer, field symbol, value)
i.e.
`overridecfg!(cfgvec, [(:net, 1, :height, 512), (:net, 1, :width, 512)])`
"""
function overridecfg!(cfgvec::Vector{Pair{Symbol,Dict{Symbol,T}}},
                        cfgchanges::Vector{Tuple{Symbol,Int,Symbol,U}};
                        silent::Bool = false) where {T,U}
    layers = map(x->first(x), cfgchanges)
    for cfgchange in cfgchanges
        layer_idxs = findall(layers .== cfgchange[1])
        length(layer_idxs) < cfgchange[2] && error("Number of $(cfgchange[1]) layers found ($(length(layer_idxs))) less than desired ($(cfgchange[2])).")
        layer_idx = layer_idxs[cfgchange[2]]
        !haskey(last(cfgvec[layer_idx]), cfgchange[3]) && error("Key not found in selected layer dict")
        cfgvec[layer_idx][2][cfgchange[3]] = cfgchange[4]
    end
end

"""
    assertdimconform(cfgvec::Vector{Pair{Symbol,Dict{Symbol,Any}}})

Assert that height and width conform to the model capabilities.
"""
function assertdimconform(cfgvec::Vector{Pair{Symbol,Dict{Symbol,T}}}) where {T}
    width = cfgvec[1][2][:width]
    height = cfgvec[1][2][:height]
    firstconvfilters = cfgvec[2][2][:filters]

    @assert (mod(width, firstconvfilters) == 0) "Model width $width not compatible with first conv size of filters=$firstconvfilters. Width should be an integer multiple of $firstconvfilters"
    @assert (mod(height, firstconvfilters) == 0) "Model height $height not compatible with first conv size of filters=$firstconvfilters. Height should be an integer multiple of $firstconvfilters"
    return true
end

_broadcast(act) = x -> act.(x)
_upsample(stride) = x -> upsample(x, stride)
_reorg(stride) = x -> reorg(x, stride)
_route(val) = x -> val
_add(val, act) = x -> act.(x + val)
_cat(arrays::AbstractArray...) = x -> cat(arrays...; dims=3)

flux_maxpool = true
@static if flux_maxpool
    ## Flux maxpool approach
    function _maxpool(siz, stride)
        # For a 2x2 pool, use explicit padding to preserve dimensions.
        pad = siz == 2 && stride == 1 ? (0, 1, 0, 1) : div(siz - 1, 2)
        return x -> maxpool(x; siz, stride, pad)
    end
    function maxpool(x; siz, stride, pad)
        return Flux.maxpool(x, Flux.PoolDims(x, (siz, siz); stride = (stride, stride), padding = pad))
    end
else
    ## Direct copy of darknet maxpool approach
    function _maxpool(siz, stride)
        pad = if siz == 2 && stride == 1
            # For a 2×2 pool with stride=1, pad asymmetrically so that
            # for an odd input (e.g. 13) the effective input becomes 14,
            # producing an output of 13.
            1
        elseif siz == 2 && stride == 2
            0
        else
            div(siz, 2)
        end
        return x -> darknet_maxpool_layer(x, siz, (stride, stride), pad)
    end
    function maxpool(x::AbstractArray{Float32,4},
        siz::Int,
        stride::Tuple{Int,Int},
        pad::Int;
        return_indexes::Bool=false)
        # x: input array with dimensions (H, W, C, N)
        # siz: pooling window size (e.g., 2)
        # stride: (stride_y, stride_x)
        # pad: total padding (as in Darknet, where often for 2×2, stride=1, pad is set so that the
        #      effective input is increased asymetrically)
        # return_indexes: if true, also return the indexes of the max values.
        H, W, C, N = size(x)
        stride_y, stride_x = stride
        out_h = div(H + pad - siz, stride_y) + 1
        out_w = div(W + pad - siz, stride_x) + 1

        # Allocate output; note we set the pool default to -Inf
        y = fill(-Inf32, out_h, out_w, C, N)
        idx = return_indexes ? similar(y, Int) : nothing

        # Compute offsets as in Darknet:
        #   h_offset = -l.pad/2,  w_offset = -l.pad/2.
        h_offset = -div(pad, 2)
        w_offset = -div(pad, 2)

        # Loop over batch, channel, and output spatial locations.
        # In Darknet, the loops are ordered as: batch, channel, out_h, out_w
        for b in 1:N
            for k in 1:C
                for i in 1:out_h
                    for j in 1:out_w
                        max_val = -Inf32
                        max_index = -1  # default (could be left as -1 if no valid element is found)
                        # Loop over the pooling window:
                        for n in 0:(siz-1)
                            for m in 0:(siz-1)
                                # Compute current position, adjusting for 1-indexed Julia arrays:
                                cur_h = h_offset + (i - 1) * stride_y + n + 1
                                cur_w = w_offset + (j - 1) * stride_x + m + 1
                                if cur_h >= 1 && cur_h <= H && cur_w >= 1 && cur_w <= W
                                    val = x[cur_h, cur_w, k, b]
                                    if val > max_val
                                        max_val = val
                                        # Save linear index (or you could choose to store a CartesianIndex)
                                        max_index = LinearIndices(x)[CartesianIndex(cur_h, cur_w, k, b)]
                                    end
                                end
                            end
                        end
                        y[i, j, k, b] = max_val
                        if return_indexes
                            idx[i, j, k, b] = max_index
                        end
                    end
                end
            end
        end
        return return_indexes ? (y, idx) : y
    end
end

########################################################
##### THE YOLO OBJECT AND CONSTRUCTOR ##################
########################################################
mutable struct Yolo <: AbstractModel
    cfg::Dict{Symbol, Any}                   # This holds all settings for the model
    chain::Flux.Chain                        # This holds chains of weights and functions
    W::Dict{Int64, AbstractArray}            # This holds arrays that the model writes to
    out::Array{Dict{Symbol, Any}, 1}         # This holds values and arrays needed for inference
    uses_gpu::Bool                           # Whether the gpu was requested to be used

    # for ConstructionBase
    Yolo(cfg::Dict{Symbol, Any} , chain::Flux.Chain, W::Dict{Int64}, out::Array{Dict{Symbol, Any}, 1}, uses_gpu::Bool) = new(cfg, chain, W, out, uses_gpu)

    # The constructor takes the official YOLO config files and weight files
    Yolo(cfgfile::String, weightfile::Union{Nothing,String}, batchsize::Int = 1; silent::Bool = false, cfgchanges=nothing, use_gpu::Bool=true, disallow_bumper::Bool = false, allocator=nothing) = begin
        # load dummy weights (avoids download for precompilation)
        dummy = isnothing(weightfile)

        # make our own shorthand `gpu` function that can be switched on or off
        if use_gpu && !(gpu([0f0]) isa Vector{Float32})
            uses_gpu = true
        else
            uses_gpu = false
        end
        maybe_gpu(x) = uses_gpu ? gpu(x) : x

        # read the config file and return [:layername => Dict(:setting => value), ...]
        # the first 'layer' is not a real layer, and has overarching YOLO settings
        cfgvec = cfgread(cfgfile)

        # make and requested changes before loading
        cfgchanges !== nothing && overridecfg!(cfgvec, cfgchanges, silent=silent)

        # check that chosen width and height of model conform with first conv layer
        assertdimconform(cfgvec)

        cfg = cfgvec[1][2]
        cfg[:cfgname] = basename(cfgfile)
        cfg[:laststage] = any(cfg -> first(cfg) === :region, cfgvec) ? :region : :yolo
        weightbytes = if dummy
            nothing # readweights knows to make up dummy weights if this is nothing
        else
            IOBuffer(read(weightfile)) # read weights file sequentially like byte stream
        end
        # these settings are populated as the network is constructed below
        # some settings are re-read later for the last part of construction
        maj, min, subv = if dummy
            ones(Int32, 3)
        else
            reinterpret(Int32, read(weightbytes, 4*3))
        end
        cfg[:darknetversion] = VersionNumber(maj, min, subv)
        old_darknet = cfg[:darknetversion] < v"0.2.0"
        # In AlexeyAB, seen was split into seen and seen_images for more training info tracking
        seen, seen_images = if old_darknet
            reinterpret(Int32, read(weightbytes, 4*1)), 0
        else
            reinterpret(Int32, read(weightbytes, 4*2))
        end
        cfg[:batchsize] = batchsize
        cfg[:output] = []

        # PART 1 - THE LAYERS
        #####################
        cfg_idx = 0
        ch = [cfg[:channels]] # this keeps track of channels per layer for creating convolutions
        first_chan_removed = false
        fn = Array{Any, 1}(nothing, 0) # this keeps the 'function' generated by every layer
        acts = Dict{Int, String}()
        for (blocktype, block) in cfgvec[2:end]
            cfg_idx += 1
            if blocktype === :convolutional
                stack   = []
                kern    = block[:size]
                filters = block[:filters]
                pad     = Bool(block[:pad]) ? div(kern-1, 2) : 0
                stride  = block[:stride]
                act     = ACT[block[:activation]]
                acts[cfg_idx] = block[:activation]
                bn      = haskey(block, :batch_normalize)
                cw, cb, bb, bw, bm, bv = try
                    readweights(weightbytes, kern, ch[end], filters, bn; old_darknet)
                catch
                    !silent && println()
                    @error "Error reading weights for layer $cfg_idx of type $blocktype. Check the weights file." kern ch[end] filters pad stride act bn
                    rethrow()
                end
                push!(stack, maybe_gpu(Flux.Conv(cw, cb; stride = stride, pad = pad, dilation = 1)))
                # push!(stack, x -> begin
                #     _out = maybe_gpu(Flux.Conv(cw, cb; stride=stride, pad=pad, dilation=1))(x)
                #     @info "Layer conv $(size(x)) => $(size(_out))"
                #     return _out
                # end)
                bn && push!(stack, maybe_gpu(Flux.BatchNorm(identity, bb, bw, bm, bv, 1f-5, 0.1f0, true, true, nothing, length(bb))))
                push!(stack, _broadcast(act))
                push!(fn, Flux.Chain(stack...))
                push!(ch, filters)
                !silent && prettyprint(["($cfg_idx) ","conv($kern,$(ch[end-1])->$(ch[end]))"," => "],[:blue,:white,:green])
                if !first_chan_removed
                    deleteat!(ch, 1) # remove first channel after use
                    first_chan_removed = true
                end
            elseif blocktype === :upsample
                stride = block[:stride]
                push!(fn, _upsample(stride)) # upsample using Kronecker tensor product
                push!(ch, ch[end])
                !silent && prettyprint(["($cfg_idx) ","upsample($stride)"," => "],[:blue,:magenta,:green])
            elseif blocktype === :reorg
                stride = block[:stride]
                push!(fn, _reorg(stride)) # reorg (reshape to (w/stride, h/stride, c*stride^2))
                push!(ch, ch[end])
                !silent && prettyprint(["($cfg_idx) ","reorg($stride)"," => "],[:blue,:magenta,:green])
            elseif blocktype === :maxpool
                siz = block[:size]
                stride = block[:stride]
                push!(fn, _maxpool(siz, stride))
                # push!(fn, x -> begin
                #     _out = _maxpool(siz, stride)(x)
                #     @info "Layer maxpool(siz=$siz, stride=$stride) $(size(x)) => $(size(_out))"
                #     return _out
                # end)
                push!(ch, ch[end])
                !silent && prettyprint(["($cfg_idx) ","maxpool($siz,$stride)"," => "],[:blue,:magenta,:green])
            # for these layers don't push a function to fn, just note the skip-type and where to skip from
            elseif blocktype === :route
                layers = block[:layers]
                # negative values are relative to the current layer. -1 means the layer before
                # positive values are absolute layer numbers but need to be corrected to one-based indexing
                indices = map(l -> l < 0 ? (cfg_idx + l) : l + 1, layers)

                if haskey(block, :groups) && haskey(block, :group_id)
                    @assert length(indices) == 1 "Grouped route only makes sense with a single input layer"
                    channels = ch[indices[1]] ÷ block[:groups]
                else
                    channels = sum(ch[i] for i in indices)
                end
                push!(ch, channels)

                # Store metadata in case we need it later during `_route`
                if haskey(block, :groups)
                    push!(fn, (indices, :route, block))
                    if !silent
                        desc = "route($(indices[1]), groups=$(block[:groups]), group_id=$(block[:group_id]))"
                        prettyprint(["\n($(length(fn))) ",desc," => "],[:blue,:cyan,:green])
                    end
                    continue
                elseif length(indices) > 1
                    push!(fn, (indices, :cat))
                else
                    push!(fn, (indices[1], :route))
                end
                !silent && prettyprint(["\n($(length(fn))) ","route($(join(indices, ",")))"," => "],[:blue,:cyan,:green])
            elseif blocktype === :shortcut
                idx = block[:from] + cfg_idx
                act = haskey(block, :activation) ? ACT[block[:activation]] : linear
                push!(fn, (idx, :add, act))
                push!(ch, ch[end])
                !silent && prettyprint(["\n($cfg_idx) ","shortcut($idx,$cfg_idx)"," => "],[:blue,:cyan,:green])
            elseif blocktype in (:yolo, :region)
                get!(block, :iou_loss, "mse")
                get!(block, :nms_kind, :default)
                get!(block, :beta_nms, 0.6)
                get!(block, :scale_x_y, 1.0f0)
                get!(block, :new_coords, 0)
                block[:final_conv_activation] = get(acts, cfg_idx-1, "linear")
                push!(fn, nothing) # not a real layer. used for bounding boxes etc...
                push!(ch, ch[end])
                push!(cfg[:output], block)
                !silent && prettyprint(["($cfg_idx) ","$blocktype"," || "],[:blue,:yellow,:green])
            else
                error("Unknown layer type $blocktype")
            end
        end

        # Sanity check that all weights were used
        if weightbytes !== nothing && !eof(weightbytes)
            fsize = filesize(weightfile)
            read_bytes = position(weightbytes)
            error("Not all weights were read. Check that the weights file matches the cfg file. Read $(read_bytes) bytes. Filesize $(fsize) bytes.")
        end

        # PART 2 - THE SKIPS
        ####################
        # Create test batch. Note that darknet is row-major, so width-first
        test_batches = [maybe_gpu(rand(Float32, cfg[:width], cfg[:height], cfg[:channels], batchsize))]
        # find all skip-layers and all YOLO layers
        skip_idxs = Int[]
        for (i, f) in enumerate(fn)
            if f isa Tuple
                if f[1] isa AbstractVector
                    append!(skip_idxs, f[1])
                else
                    push!(skip_idxs, f[1])
                end
            elseif f === nothing
                push!(skip_idxs, i - 1)
            end
        end
        needout = sort(unique(vcat(0, skip_idxs)))

        chainstack = Flux.Chain[] # layers that just feed forward can be grouped together in chains
        layer2out = Dict() # this dict translates layer numbers to chain numbers
        W = Dict{Int64, typeof(test_batches[1])}() # this holds temporary outputs for use by skip-layers and YOLO output
        out = Array{Dict{Symbol, Any}, 1}(undef, 0) # store values needed for interpreting YOLO output
        !silent && println("\n\nGenerating chains and outputs: ")
        for i in eachindex(needout)[2:end]
            !silent && print(rpad("$(i-1))", 4))
            fst, lst = needout[i-1]+1, needout[i] # these layers feed forward to an output
            if fn[fst] === nothing # check if sequence of layers begin with YOLO output
                push!(out, Dict(:idx => layer2out[fst-1]))
                fst += 1
            end
            # generate the functions used by the skip-layers and reference the temporary outputs
            range = fst:lst
            !silent && print(rpad(string(range), 8))
            for j in range
                if typeof(fn[j]) <: Tuple
                    skip_type = fn[j][2]
                    if skip_type === :route
                        arrayidx = layer2out[fn[j][1]]
                        if length(fn[j]) == 3 && haskey(fn[j][3], :groups)
                            groups = fn[j][3][:groups]
                            group_id = fn[j][3][:group_id] # zero-based (this is usually 1, so the 2nd group)
                            group_size = size(W[arrayidx], 3) ÷ groups
                            group_start = group_id * group_size + 1
                            group_end = (group_id+1) * group_size
                            fn[j] = _route(W[arrayidx][:, :, group_start:group_end, :])
                        else
                            fn[j] = _route(W[arrayidx])
                        end
                    elseif skip_type === :add
                        arrayidx = layer2out[fn[j][1]]
                        act = fn[j][3]
                        fn[j] = _add(W[arrayidx], act)
                    elseif skip_type === :cat
                        indices = fn[j][1]
                        fn[j] = _cat([W[layer2out[ind]] for ind in indices]...)
                    else
                        error("Unknown skip layer $skip_type")
                    end
                end
            end
            push!(chainstack, Flux.Chain(fn[range]...)) # add sequence of functions to a chain

            outvar = chainstack[end](test_batches[end]) # run the new chain on the previous output
            !silent && println("Output: ", size(outvar))
            push!(test_batches, outvar)
            push!(W, i-1 => copy(outvar)) # generate a temporary array for the output of the chain
            push!(layer2out, [l => i-1 for l in range]...)
        end
        !silent && println()
        matrix_sizes_x = [size(v, 1) for (k,v) in W]
        matrix_sizes_y = [size(v, 2) for (k,v) in W]
        cfg[:gridsize] = (minimum(matrix_sizes_x), minimum(matrix_sizes_y)) # the gridsize is determined by the smallest matrix
        cfg[:layer2out] = layer2out
        push!(out, Dict(:idx => length(W)))

        # PART 3 - THE OUTPUTS
        ######################
        @views for i in eachindex(out)
            # we pre-process some linear matrix transformations and store the values for each YOLO output
            w, h, f, b = size(W[out[i][:idx]]) # width, height, filters, batchsize
            strideh = cfg[:height] ÷ h # stride height for this particular output
            stridew = cfg[:width] ÷ w # stride width
            if haskey(cfg[:output][i], :mask)
                anchormask = cfg[:output][i][:mask] .+ 1 # check which anchors are necessary from the config
            else
                # Use all anchors (length(anchors) / 2 because each anchor has 2 vals)
                anchormask = 1:round(Int, length(cfg[:output][i][:anchors])/2)
            end
            anchorvals = reshape(cfg[:output][i][:anchors], 2, :)[:, anchormask] ./ [stridew, strideh]
            # attributes are (normed and centered) - x, y, w, h, confidence, [number of classes]...
            attributes = 5 + cfg[:output][i][:classes]

            # precalculate the offset of prediction from cell-relative to (last) layer-relative
            offset = maybe_gpu(similar(test_batches[1], Float32, w, h, 2, length(anchormask), b))
            for j in 0:w-1, k in 0:h-1
                offset[j+1, k+1, 1, :, :] .= j
                offset[j+1, k+1, 2, :, :] .= k
            end

            # precalculate the scale factor from layer-relative to image-relative
            scale = maybe_gpu(similar(test_batches[1], Float32, w, h, 2, length(anchormask), b))
            scale[:, :, 1, :, :] .= stridew
            scale[:, :, 2, :, :] .= strideh

            # precalculate the anchor shapes to scale up the detection boxes
            anchor = maybe_gpu(similar(test_batches[1], Float32, w, h, 2, length(anchormask), b))
            for j in eachindex(anchormask)
                anchor[:, :, 1, j, :] .= anchorvals[1, j] * stridew
                anchor[:, :, 2, j, :] .= anchorvals[2, j] * strideh
            end

            out[i][:size] = (w, h, attributes, length(anchormask), b)
            out[i][:offset] = offset
            out[i][:scale] = scale
            out[i][:anchor] = anchor
            get!(cfg[:output][i], :truth_thresh,  get(cfg[:output][i], :thresh, 0.0)) # for object being detected (at all). Called thresh in v2
            get!(cfg[:output][i], :ignore_thresh, 1.0) # for ignoring detections of same object (overlapping)
            out[i][:truth_thresh] = cfg[:output][i][:truth_thresh]
            out[i][:ignore_thresh] = cfg[:output][i][:ignore_thresh]
            out[i][:scale_x_y] = cfg[:output][i][:scale_x_y]
            out[i][:nms_kind] = Symbol(cfg[:output][i][:nms_kind])
            out[i][:beta_nms] = Float32(cfg[:output][i][:beta_nms])
            out[i][:final_conv_activation] = cfg[:output][i][:final_conv_activation]
        end

        yolomod = new(cfg, Flux.Chain(chainstack), W, out, uses_gpu)
        if uses_gpu || disallow_bumper
            return yolomod
        else
            kw = allocator === nothing ? NamedTuple() : (; allocator=allocator)
            return wrap_model(yolomod; kw...)
        end
    end
end

uses_gpu(y::Yolo) = y.uses_gpu
get_cfg(y::Yolo) = y.cfg

# make yolo `Adapt`-able
Flux.@layer :ignore Yolo

"""
    get_input_size(model::Yolo)

Returns model size tuple in (width, height, channels, batchsize) order (row-major)
"""
get_input_size(model::Yolo) = (get_cfg(model)[:width], get_cfg(model)[:height], get_cfg(model)[:channels], get_cfg(model)[:batchsize])

function Base.show(io::IO, yolo::Yolo)
    detect_thresh = yolo.cfg[:output][1][:truth_thresh]
    overlap_thresh = yolo.cfg[:output][1][:ignore_thresh]
    ln1 = "$(yolo.cfg[:cfgname]). Trained with DarkNet $(yolo.cfg[:darknetversion])\n"
    ln2 = "WxH: $(yolo.cfg[:width])x$(yolo.cfg[:height])   channels: $(yolo.cfg[:channels])   batchsize: $(yolo.cfg[:batchsize])\n"
    ln3 = "gridsize: $(yolo.cfg[:gridsize][1])x$(yolo.cfg[:gridsize][2])   classes: $(yolo.cfg[:output][1][:classes])   thresholds: Detect $detect_thresh. Overlap $overlap_thresh"
    print(io, ln1 * ln2 * ln3)
end

########################################################
##### FUNCTIONS FOR INFERENCE ##########################
########################################################
"""
    clipdetect!(input::AbstractArray, conf)

Sets all values under a given threshold to zero
"""
function clipdetect!(input::AbstractArray, conf)
    @inbounds for i in axes(input, 2)
        # class_confidence is at end-2. Must be computed before this in findmax!
        if input[end-2, i] < conf
            input[end-2, i] = 0f0
        end
    end
end

"""
    findmax!(input::Array{T}) where {T}

Findmax, get the class with highest confidence and class number out.
"""
function findmax!(input::AbstractArray{T}) where {T}
    @inbounds for i in axes(input, 2)
        input[end-2, i], input[end-1, i] = findmax(@view input[6:end-3, i])
    end
end


"""
    keepdetections(arr::AbstractArray)

Reduces the size of array and only keeps detections over threshold
"""
function keepdetections(arr::AbstractArray)
    return arr[:, arr[end-2, :] .> 0]
end

function extend_for_attributes(weights::AbstractArray, w, h, bo, ba)
    x = similar(weights, Float32, w, h, 4, bo, ba)
    x .= 0f0
    return cat(weights, x, dims = 3)
end

check_w_type(arr::AllocArray) = check_w_type(arr.arr)
check_w_type(::UnsafeArray) = throw(ArgumentError("Internal error: UnsafeArray has leaked into internal buffer"))
check_w_type(::Any) = nothing

"""
    (yolo::Yolo)(img::AbstractArray;  detect_thresh=nothing, overlap_thresh=nothing)

Simply pass a batch of images to the yolo object to do inference.

detect_thresh: Optionally override the minimum allowable detection confidence
overlap_thresh: Optionally override the maximum allowable overlap (IoU)
show_timing::Bool=false: Show timing information for each layer
conf_fix::Bool=true: Apply fix to the confidence score calculation. Without this the class scores are not multiplied by the box confidence score, as they should be.
"""
function (yolo::Yolo)(img::T; detect_thresh=nothing, overlap_thresh=nothing, show_timing=false, conf_fix=true) where {T <: AbstractArray}
    if show_timing
        enable_timer!(to)
        reset_timer!(to)
    else
        disable_timer!(to)
    end
    @timeit to "yolo" begin
        @assert ndims(img) == 4 # width, height, channels, batchsize
        yolo.W[0] = img

        # FORWARD PASS
        ##############
        @timeit to "forward pass" for i in eachindex(yolo.chain) # each chain writes to a predefined output
            @timeit to "layer $i" begin
                f = yolo.chain[i]
                out = f(yolo.W[i-1]::T)
                check_w_type(yolo.W[i])
                yolo.W[i] .= out
            end
        end

        # PROCESSING EACH YOLO OUTPUT
        #############################
        @timeit to "post-processing" begin
            outweights = Any[]
            outnr = 0
            @timeit to "processing outputs" @views for out in yolo.out
                outnr += 1
                w, h, a, bo, ba = out[:size]
                weights = reshape(yolo.W[out[:idx]], w, h, a, bo, ba)
                sxy = out[:scale_x_y]
                # adjust the predicted box coordinates into pixel values
                if yolo.cfg[:output][outnr][:new_coords] == 1
                    if haskey(yolo.cfg[:output][outnr], :max_delta)
                        delta = Float32(yolo.cfg[:output][outnr][:max_delta])
                        clamp!(weights[:, :, 1:2, :, :], -delta, delta)
                    end
                    weights[:, :, 1:2, :, :] = (weights[:, :, 1:2, :, :] .* sxy .- (sxy - 1)/2 .+ out[:offset]) .* out[:scale]
                    weights[:, :, 3:4, :, :] = (weights[:, :, 3:4, :, :] .* sxy).^2 .* out[:anchor]
                else
                    # Classic behavior
                    weights[:, :, 1:2, :, :] = (σ.(weights[:, :, 1:2, :, :]) .* sxy .- (sxy - 1)/2 .+ out[:offset]) .* out[:scale]
                    weights[:, :, 3:4, :, :] = exp.(weights[:, :, 3:4, :, :]) .* out[:anchor]
                end

                # Apply sigmoid to objectness (5) and class scores (6:a) ONLY if the
                # preceding conv layer activation was NOT logistic (e.g., it was linear)
                if out[:final_conv_activation] != "logistic"
                    weights[:, :, 5:end, :, :] = σ.(weights[:, :, 5:end, :, :])
                end

                if conf_fix
                    # post-sigmoid class confidence scores should be multiplied by the post-sigmoid box confidence score
                    # see https://github.com/openvinotoolkit/open_model_zoo/blob/master/models/public/yolo-v3-tiny-tf/README.md#original-model-1
                    weights[:, :, 6:end, :, :] = weights[:, :, 6:end, :, :] .* weights[:, :, 5:5, :, :]
                end

                # Convert to image width & height scale (0.0-1.0)
                weights[:, :, 1, :, :] = weights[:, :, 1, :, :] ./ size(img, 1) #x
                weights[:, :, 2, :, :] = weights[:, :, 2, :, :] ./ size(img, 2) #y
                if yolo.cfg[:laststage] === :region # indicates yolov2
                    cellsize_x, cellsize_y = (yolo.cfg[:width], yolo.cfg[:height]) ./ yolo.cfg[:gridsize]
                    weights[:, :, 3, :, :] = (weights[:, :, 3, :, :] ./ size(img, 1)) * cellsize_x #w
                    weights[:, :, 4, :, :] = (weights[:, :, 4, :, :] ./ size(img, 2)) * cellsize_y #h
                else
                    weights[:, :, 3, :, :] = (weights[:, :, 3, :, :] ./ size(img, 1)) #w
                    weights[:, :, 4, :, :] = (weights[:, :, 4, :, :] ./ size(img, 2)) #h
                end

                weights[:, :, 1, :, :] = weights[:, :, 1, :, :] .- (weights[:, :, 3, :, :] .* 0.5) #x1
                weights[:, :, 2, :, :] = weights[:, :, 2, :, :] .- (weights[:, :, 4, :, :] .* 0.5) #y1
                weights[:, :, 3, :, :] = weights[:, :, 1, :, :] .+ weights[:, :, 3, :, :] #x2
                weights[:, :, 4, :, :] = weights[:, :, 2, :, :] .+ weights[:, :, 4, :, :] #y2

                # add additional attributes for post-inference analysis: confidence, classnr, outnr, batchnr
                weights = extend_for_attributes(weights, w, h, bo, ba)

                weights[:, :, a+3, outnr, :] .= outnr # write output number to attribute a+3
                for batch in 1:ba weights[:, :, a+4, :, batch] .= batch end # write batchnumber to attribute a+4
                weights = permutedims(weights, [3, 1, 2, 4, 5]) # place attributes first
                weights = reshape(weights, a+4, :) # reshape to attr, data

                detect_thresh = Float32(@something detect_thresh out[:truth_thresh])
                findmax!(weights)
                clipdetect!(weights, detect_thresh) # set all detections below conf-thresh to zero
                push!(outweights, weights)
            end

            # PROCESSING ALL PREDICTIONS
            ############################
            @timeit to "filter detections" batchout = cpu(keepdetections(cat(outweights..., dims=2)))

            if size(batchout, 2) < 2
                ret = batchout # empty or singular output doesn't need further filtering
            else
                batchsize = yolo.cfg[:batchsize]
                @timeit to "nms" begin
                    overlap_thresh = Float32(@something overlap_thresh yolo.out[1][:ignore_thresh])
                    nms_kind = yolo.out[1][:nms_kind]
                    beta_nms = yolo.out[1][:beta_nms]
                    ret = perform_detection_nms(batchout, overlap_thresh, batchsize; kind=nms_kind, beta=beta_nms)
                end
            end
        end
    end
    if show_timing
        show(to, sortby=:firstexec)
        println()
    end
    return ret
end

include(joinpath(@__DIR__, "pretrained.jl"))

end #module
