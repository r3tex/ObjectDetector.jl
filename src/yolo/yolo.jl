module YOLO
export get_input_size

import ..to, ..AbstractModel, ..get_input_size, ..wrap_model, ..uses_gpu, ..get_cfg
#import ..getArtifact #disabled due to https://github.com/JuliaLang/Pkg.jl/issues/1579

models_dir() = joinpath(@__DIR__, "models")

import Flux
import Flux: gpu, σ
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
function readweights(bytes::Union{IOBuffer,Nothing}, kern::Int, ch::Int, fl::Int, bn::Bool)
    dummy = isnothing(bytes)
    if bn
        bb = dummy ? ones(Float32, fl) : reinterpret(Float32, read(bytes, fl*4))
        bw = dummy ? ones(Float32, fl) : reinterpret(Float32, read(bytes, fl*4))
        bm = dummy ? ones(Float32, fl) : reinterpret(Float32, read(bytes, fl*4))
        bv = dummy ? ones(Float32, fl) : reinterpret(Float32, read(bytes, fl*4))
        cb = zeros(Float32, fl)
        cw = dummy ? ones(Float32, kern, kern, ch, fl) : reshape(reinterpret(Float32, read(bytes, kern*kern*ch*fl*4)), kern, kern, ch, fl)
        cw = Float32.(flip(cw))
        return cw, cb, bb, bw, bm, bv
    else
        cb = dummy ? ones(Float32, fl) : reinterpret(Float32, read(bytes, fl*4))
        cw = dummy ? ones(Float32, kern, kern, ch, fl) : reshape(reinterpret(Float32, read(bytes, kern*kern*ch*fl*4)), kern, kern, ch, fl)
        cw = Float32.(flip(cw))
        return cw, cb, 0.0, 0.0, 0.0, 0.0
    end
end

########################################################
##### FUNCTIONS NEEDED FOR THE YOLO CONSTRUCTOR ########
########################################################
"""
    leaky(x, a = oftype(x/1, 0.1))

YOLO wants a leakyrelu with a fixed leakyness of 0.1 so we define our own
"""
leaky(x, a = oftype(x/1, 0.1)) = max(a*x, x/1)

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

# Use this dict to translate the config activation names to function names
const ACT = Dict(
    "leaky" => leaky,
    "linear" => identity
)

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

function maxpool(x; siz, stride)
    return Flux.maxpool(x, Flux.PoolDims(x, (siz, siz); stride = (stride, stride), padding = (0,2-stride,0,2-stride)))
end

_broadcast(act) = x -> act.(x)
_upsample(stride) = x -> upsample(x, stride)
_reorg(stride) = x -> reorg(x, stride)
_route(val) = x -> val
_add(val) = x -> x + val
_cat(val) = x -> cat(x, val, dims = 3)
_maxpool(siz, stride) = x -> maxpool(x; siz, stride)

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
        cfgchanges != nothing && overridecfg!(cfgvec, cfgchanges, silent=silent)

        # check that chosen width and height of model conform with first conv layer
        assertdimconform(cfgvec)

        cfg = cfgvec[1][2]
        yoloversion = any(first.(cfgvec) .== :region) ? 2 : 3 #v2 calls the last stage "region", v3 uses "yolo"
        cfg[:yoloversion] = yoloversion
        weightbytes = if dummy
            nothing # readweights knows to make up dummy weights if this is nothing
        else
            IOBuffer(read(weightfile)) # read weights file sequentially like byte stream
        end
        # these settings are populated as the network is constructed below
        # some settings are re-read later for the last part of construction
        maj, min, subv, im1, im2 = if dummy
            ones(Int32, 5)
        else
            reinterpret(Int32, read(weightbytes, 4*5))
        end
        cfg[:darknetversion] = VersionNumber("$maj.$min.$subv")
        cfg[:batchsize] = batchsize
        cfg[:output] = []

        # PART 1 - THE LAYERS
        #####################
        ch = [cfg[:channels]] # this keeps track of channels per layer for creating convolutions
        fn = Array{Any, 1}(nothing, 0) # this keeps the 'function' generated by every layer
        for (blocktype, block) in cfgvec[2:end]
            if blocktype == :convolutional
                stack   = []
                kern    = block[:size]
                filters = block[:filters]
                pad     = Bool(block[:pad]) ? div(kern-1, 2) : 0
                stride  = block[:stride]
                act     = ACT[block[:activation]]
                bn      = haskey(block, :batch_normalize)
                cw, cb, bb, bw, bm, bv = readweights(weightbytes, kern, ch[end], filters, bn)
                push!(stack, maybe_gpu(Flux.Conv(cw, cb; stride = stride, pad = pad, dilation = 1)))
                bn && push!(stack, maybe_gpu(Flux.BatchNorm(identity, bb, bw, bm, bv, 1f-5, 0.1f0, true, true, nothing, length(bb))))
                push!(stack, _broadcast(act))
                push!(fn, Flux.Chain(stack...))
                push!(ch, filters)
                !silent && prettyprint(["($(length(fn))) ","conv($kern,$(ch[end-1])->$(ch[end]))"," => "],[:blue,:white,:green])
                ch = ch[1] == cfg[:channels] ? ch[2:end] : ch # remove first channel after use
            elseif blocktype == :upsample
                stride = block[:stride]
                push!(fn, _upsample(stride)) # upsample using Kronecker tensor product
                push!(ch, ch[end])
                !silent && prettyprint(["($(length(fn))) ","upsample($stride)"," => "],[:blue,:magenta,:green])
            elseif blocktype == :reorg
                stride = block[:stride]
                push!(fn, _reorg(stride)) # reorg (reshape to (w/stride, h/stride, c*stride^2))
                push!(ch, ch[end])
                !silent && prettyprint(["($(length(fn))) ","reorg($stride)"," => "],[:blue,:magenta,:green])
            elseif blocktype == :maxpool
                siz = block[:size]
                stride = block[:stride]
                push!(fn, _maxpool(siz, stride))
                push!(ch, ch[end])
                !silent && prettyprint(["($(length(fn))) ","maxpool($siz,$stride)"," => "],[:blue,:magenta,:green])
            # for these layers don't push a function to fn, just note the skip-type and where to skip from
            elseif blocktype == :route
                idx1 = length(fn) + block[:layers][1] + 1
                if length(block[:layers]) > 1
                    if block[:layers][2] > 0
                        idx2 = block[:layers][2] + 1
                    else
                        idx2 = length(fn) + block[:layers][2] + 1 # Handle -ve route selections
                    end
                    push!(ch, ch[idx1] + ch[idx2])
                    push!(fn, (idx2, :cat)) # cat two layers along the channel dim
                else
                    idx2 = ""
                    push!(ch, ch[idx1])
                    push!(fn, (idx1, :route)) # pull a whole layer from a few steps back
                end
                !silent && prettyprint(["\n($(length(fn))) ","route($idx1,$idx2)"," => "],[:blue,:cyan,:green])
            elseif blocktype == :shortcut
                act = ACT[block[:activation]]
                idx = block[:from] + length(fn)+1
                push!(fn, (idx, :add)) # take two layers with equal num of channels and adds their values
                push!(ch, ch[end])
                !silent && prettyprint(["\n($(length(fn))) ","shortcut($idx,$(length(fn)-1))"," => "],[:blue,:cyan,:green])
            elseif blocktype == :yolo
                push!(fn, nothing) # not a real layer. used for bounding boxes etc...
                push!(ch, ch[end])
                push!(cfg[:output], block)
                !silent && prettyprint(["($(length(fn))) ","YOLO"," || "],[:blue,:yellow,:green])
            elseif blocktype == :region
                push!(fn, nothing) # not a real layer. used for bounding boxes etc...
                push!(ch, ch[end])
                push!(cfg[:output], block)
                !silent && prettyprint(["($(length(fn))) ","region"," || "],[:blue,:yellow,:green])
            end
        end

        # PART 2 - THE SKIPS
        ####################
        # Create test batch. Note that darknet is row-major, so width-first
        test_batches = [maybe_gpu(rand(Float32, cfg[:width], cfg[:height], cfg[:channels], batchsize))]
        # find all skip-layers and all YOLO layers
        needout = sort(vcat(0, [l[1] for l in filter(f -> typeof(f) <: Tuple, fn)], findall(x -> x == nothing, fn) .- 1))
        chainstack = Flux.Chain[] # layers that just feed forward can be grouped together in chains
        layer2out = Dict() # this dict translates layer numbers to chain numbers
        W = Dict{Int64, typeof(test_batches[1])}() # this holds temporary outputs for use by skip-layers and YOLO output
        out = Array{Dict{Symbol, Any}, 1}(undef, 0) # store values needed for interpreting YOLO output
        !silent && println("\n\nGenerating chains and outputs: ")
        for i in 2:length(needout)
            !silent && print("$(i-1) ")
            fst, lst = needout[i-1]+1, needout[i] # these layers feed forward to an output
            if typeof(fn[fst]) == Nothing # check if sequence of layers begin with YOLO output
                push!(out, Dict(:idx => layer2out[fst-1]))
                fst += 1
            end
            # generate the functions used by the skip-layers and reference the temporary outputs
            for j in fst:lst
                if typeof(fn[j]) <: Tuple
                    arrayidx = layer2out[fn[j][1]]
                    skip_type = fn[j][2]
                    if skip_type == :route
                        fn[j] = _route(identity(W[arrayidx]))
                    elseif skip_type == :add
                        fn[j] =  _add(W[arrayidx])
                    elseif skip_type == :cat
                        fn[j] = _cat(W[arrayidx])
                    else
                        error("Unknown skip layer $skip_type")
                    end
                end
            end
            push!(chainstack, Flux.Chain(fn[fst:lst]...)) # add sequence of functions to a chain
            push!(test_batches, chainstack[end](test_batches[end])) # run the previous test image
            push!(W, i-1 => copy(test_batches[end])) # generate a temporary array for the output of the chain
            push!(layer2out, [l => i-1 for l in fst:lst]...)
        end
        !silent && print("\n\n")
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
                anchormask = 1:round(Int, length(cfg[:output][i][:anchors])/2)
            end
            anchorvals = reshape(cfg[:output][i][:anchors], 2, :)[:, anchormask] ./ [stridew, strideh]
            # attributes are (normed and centered) - x, y, w, h, confidence, [number of classes]...
            attributes = 5 + cfg[:output][i][:classes]

            # precalculate the offset of prediction from cell-relative to (last) layer-relative
            offset = maybe_gpu(reshape(zeros(Float32, w*h*2*length(anchormask)*b), w, h, 2, length(anchormask), b))
            @views for i in 0:w-1, j in 0:h-1
                offset[i+1, j+1, 1, :, :] = offset[i+1, j+1, 1, :, :] .+ i
                offset[i+1, j+1, 2, :, :] = offset[i+1, j+1, 2, :, :] .+ j
            end

            # precalculate the scale factor from layer-relative to image-relative
            scale = maybe_gpu(reshape(similar(out, Float32, w*h*2*length(anchormask)*b), w, h, 2, length(anchormask), b))
            scale .= 1.0f0

            @views for i in 0:w-1, j in 0:h-1
                scale[i+1, j+1, 1, :, :] = scale[i+1, j+1, 1, :, :] .* stridew
                scale[i+1, j+1, 2, :, :] = scale[i+1, j+1, 2, :, :] .* strideh
            end

            # precalculate the anchor shapes to scale up the detection boxes
            x = similar(out, Float32, w*h*2*length(anchormask)*b)
            x .= 1.0f0
            anchor = maybe_gpu(reshape(x, w, h, 2, length(anchormask), b))

            for i in 1:length(anchormask)
                anchor[:, :, 1, i, :] .= anchorvals[1, i] * stridew
                anchor[:, :, 2, i, :] .= anchorvals[2, i] * strideh
            end

            out[i][:size] = (w, h, attributes, length(anchormask), b)
            out[i][:offset] = offset
            out[i][:scale] = scale
            out[i][:anchor] = anchor
            out[i][:truth] = get(cfg[:output][i], :truth_thresh, get(cfg[:output][i], :thresh, 0.0)) # for object being detected (at all). Called thresh in v2
            out[i][:ignore] = get(cfg[:output][i], :ignore_thresh, 0.3) # for ignoring detections of same object (overlapping)
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
    detect_thresh = get(yolo.cfg[:output][1], :truth_thresh, get(yolo.cfg[:output][1], :thresh, 0.0))
    overlap_thresh = get(yolo.cfg[:output][1], :ignore_thresh, 0.0)
    ln1 = "YOLO v$(yolo.cfg[:yoloversion]). Trained with DarkNet $(yolo.cfg[:darknetversion])\n"
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
   rows, cols = size(input)
   for i in 1:cols
       input[5, i] = ifelse(input[5, i] > conf, input[5, i], Float32(0))
   end
end

"""
    findmax!(input::Array{T}, idst::Int, idend::Int) where {T}

Findmax, get the class with highest confidence and class number out.
"""
function findmax!(input::AbstractArray{T}, idst::Int, idend::Int) where {T}
    for i in 1:size(input, 2)
        input[end-2, i], input[end-1, i] = findmax(@view input[idst:idend, i])
    end
end


"""
    keepdetections(arr::AbstractArray)

Reduces the size of array and only keeps detections over threshold
"""
function keepdetections(arr::AbstractArray)
    return arr[:, arr[5, :] .> 0]
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
    (yolo::Yolo)(img::AbstractArray;  detectThresh=nothing, overlapThresh=yolo.out[1][:ignore])

Simply pass a batch of images to the yolo object to do inference.

detectThresh: Optionally override the minimum allowable detection confidence
overalThresh: Optionally override the maximum allowable overlap (IoU)
show_timing::Bool=false: Show timing information for each layer
conf_fix::Bool=true: Apply fix to the confidence score calculation. Without this the class scores are not multiplied by the box confidence score, as they should be.
"""
function (yolo::Yolo)(img::T; detectThresh=nothing, overlapThresh=yolo.out[1][:ignore], show_timing=false, conf_fix=true) where {T <: AbstractArray}
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
                # adjust the predicted box coordinates into pixel values
                weights[:, :, 1:2, :, :] = (σ.(weights[:, :, 1:2, :, :]) + out[:offset]) .* out[:scale]
                weights[:, :, 5:end, :, :] = σ.(weights[:, :, 5:end, :, :])
                if conf_fix
                    # post-sigmoid class confidence scores should be multiplied by the post-sigmoid box confidence score
                    # see https://github.com/openvinotoolkit/open_model_zoo/blob/master/models/public/yolo-v3-tiny-tf/README.md#original-model-1
                    weights[:, :, 6:end, :, :] = weights[:, :, 6:end, :, :] .* weights[:, :, 5:5, :, :]
                end
                weights[:, :, 3:4, :, :] = exp.(weights[:, :, 3:4, :, :]) .* out[:anchor]

                cellsize_x, cellsize_y = (yolo.cfg[:width], yolo.cfg[:height]) ./ yolo.cfg[:gridsize]

                # Convert to image width & height scale (0.0-1.0)
                weights[:, :, 1, :, :] = weights[:, :, 1, :, :] ./ size(img, 1) #x
                weights[:, :, 2, :, :] = weights[:, :, 2, :, :] ./ size(img, 2) #y
                if yolo.cfg[:yoloversion] == 2
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

                thresh = detectThresh === nothing ? Float32(out[:truth]) : Float32(detectThresh)
                clipdetect!(weights, thresh) # set all detections below conf-thresh to zero
                findmax!(weights, 6, a)
                push!(outweights, weights)
            end

            # PROCESSING ALL PREDICTIONS
            ############################
            @timeit to "filter detections" batchout = Flux.cpu(keepdetections(cat(outweights..., dims=2)))

            if size(batchout, 2) < 2
                if show_timing
                    show(to, sortby=:firstexec)
                    println()
                end
                ret = batchout # empty or singular output doesn't need further filtering
            else
                batchsize = yolo.cfg[:batchsize]
                @timeit to "nms" begin
                    ret = perform_detection_nms(batchout, overlapThresh, batchsize)
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

include(joinpath(@__DIR__,"pretrained.jl"))

end #module
