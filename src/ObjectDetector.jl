module ObjectDetector
export Yolo

#= ############################################

THIS IS Robert Luciani's IMPLEMENTATION OF YOLO

############################################ =#

using Flux
in(:CuArrays, names(Main, imported = true)) && (using CuArrays; CuArrays.allowscalar(false))

#########################################################
##### FUNCTIONS FOR PARSING CONFIG AND WEIGHT FILES #####
#########################################################

# convert config String values into native Julia types
# not type safe, but not performance critical
function cfgparse(val::AbstractString)
    if all(isletter, val)
        return val::AbstractString
    else
        return out = occursin('.', val) ? parse(Float64, val) : parse(Int64, val)
    end
end

# split config String into a key and value part
# split value into array if necessary
function cfgsplit(dat::String)
    name, values = split(dat, '=')
    values = split(values, ',')
    k = Symbol(strip(name))
    v = length(values) == 1 ? cfgparse(values[1]) : [cfgparse(v) for v in values]
    return k::Symbol => v::Any
end

# read config file and return an array of settings
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

# Read the YOLO binary weights
function readweights(bytes::IOBuffer, kern::Int, ch::Int, fl::Int, bn::Bool)
    if bn
        bb = reinterpret(Float32, read(bytes, fl*4))
        bw = reinterpret(Float32, read(bytes, fl*4))
        bm = reinterpret(Float32, read(bytes, fl*4))
        bv = reinterpret(Float32, read(bytes, fl*4))
        cb = zeros(Float32, fl)
        cw = reshape(reinterpret(Float32, read(bytes, kern*kern*ch*fl*4)), kern, kern, ch, fl)
        cw = Float32.(flip(cw))
        return cw, cb, bb, bw, bm, bv
    else
        cb = reinterpret(Float32, read(bytes, fl*4))
        cw = reshape(reinterpret(Float32, read(bytes, kern*kern*ch*fl*4)), kern, kern, ch, fl)
        cw = Float32.(flip(cw))
        return cw, cb, 0.0, 0.0, 0.0, 0.0
    end
end

########################################################
##### FUNCTIONS NEEDED FOR THE YOLO CONSTRUCTOR ########
########################################################

# Use different generators depending on presence of GPU
onegen = in(:CuArrays, names(Main, imported = true)) ? CuArrays.ones : ones
zerogen = in(:CuArrays, names(Main, imported = true)) ? CuArrays.zeros : zeros

# YOLO wants a leakyrelu with a fixed leakyness of 0.1 so we define our own
leaky(x, a = oftype(x/1, 0.1)) = max(a*x, x/1)

# Provide an array of strings and an array of colors
# so the constructor can print what it's doing as it generates the model.
prettyprint(str, col) = for (s, c) in zip(str, col) printstyled(s, color=c) end

# Flip weights to make crosscorelation kernels work using convolution filters
# This is only run once when wheights are loaded
flip(x) = x[end:-1:1, end:-1:1, :, :]

# We need a max-pool with a fixed stride of 1
function maxpools1(x, kernel = 2)
    x = cat(x, x[:, end:end, :, :], dims = 2)
    x = cat(x, x[end:end, :, :, :], dims = 1)
    return maxpool(x, (kernel, kernel), stride = 1)
end

# Optimized upsampling without indexing for better GPU performance
function upsample(a, stride)
    m1, n1, o1, p1 = size(a)
    ar = reshape(a, (1, m1, 1, n1, o1, p1))
    b = onegen(stride, 1, stride, 1, 1, 1)
    return reshape(ar .* b, (m1 * stride, n1 * stride, o1, p1))
end

# Use this dict to translate the config activation names to function names
const ACT = Dict(
    "leaky" => leaky,
    "linear" => identity
)

########################################################
##### THE YOLO OBJECT AND CONSTRUCTOR ##################
########################################################

mutable struct Yolo
    cfg::Dict{Symbol, Any}                   # This holds all settings for the model
    chain::Array{Any, 1}                     # This holds chains of weights and functions
    W::Dict{Int64, T} where T <: DenseArray  # This holds arrays that the model writes to
    out::Array{Dict{Symbol, Any}, 1}         # This holds values and arrays needed for inference

    # The constructor takes the official YOLO config files and weight files
    Yolo(cfgfile::String, weightfile::String, batchsize::Int = 1) = begin
        # read the config file and return [:layername => Dict(:setting => value), ...]
        # the first 'layer' is not a real layer, and has overarching YOLO settings
        cfgvec = cfgread(cfgfile)
        cfg = cfgvec[1][2]
        weightbytes = IOBuffer(read(weightfile)) # read weights file sequentially like byte stream
        # these settings are populated as the network is constructed below
        # some settings are re-read later for the last part of construction
        maj, min, subv, im1, im2 = reinterpret(Int32, read(weightbytes, 4*5))
        cfg[:version] = VersionNumber("$maj.$min.$subv")
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
                push!(stack, gpu(Conv(cw, cb; stride = stride, pad = pad, dilation = 1)))
                bn && push!(stack, gpu(BatchNorm(identity, Flux.param(bb), Flux.param(bw), bm, bv, 1e-5, 0.1, false)))
                push!(stack, x -> act.(x))
                push!(fn, Chain(stack...))
                push!(ch, filters)
                prettyprint(["($(length(fn))) ","conv($kern,$(ch[end-1])->$(ch[end]))"," => "],[:blue,:white,:green])
                ch = ch[1] == cfg[:channels] ? ch[2:end] : ch # remove first channel after use
            elseif blocktype == :upsample
                stride = block[:stride]
                push!(fn, x -> upsample(x, stride)) # upsample using Kronecker tensor product
                push!(ch, ch[end])
                prettyprint(["($(length(fn))) ","upsample($stride)"," => "],[:blue,:magenta,:green])
            elseif blocktype == :maxpool
                siz = block[:size]
                stride = block[:stride] # use our custom stride function if size is 1
                stride == 1 ? push!(fn, x -> maxpools1(x, siz)) : push!(fn, x -> maxpool(x, (siz, siz), stride = (stride, stride)))
                push!(ch, ch[end])
                prettyprint(["($(length(fn))) ","maxpool($siz,$stride)"," => "],[:blue,:magenta,:green])
            # for these layers don't push a function to fn, just note the skip-type and where to skip from
            elseif blocktype == :route
                idx1 = block[:layers][1] + length(fn)+1
                idx2 = length(block[:layers]) > 1 ? block[:layers][2]+1 : ""
                if idx2 == ""
                    push!(ch, ch[idx1])
                    push!(fn, (idx1, :route)) # pull a whole layer from a few steps back
                else
                    push!(ch, ch[idx1] + ch[idx2])
                    push!(fn, (idx2, :cat)) # cat two layers along the channel dim
                end
                prettyprint(["\n($(length(fn))) ","route($idx1,$idx2)"," => "],[:blue,:cyan,:green])
            elseif blocktype == :shortcut
                act = ACT[block[:activation]]
                idx = block[:from] + length(fn)+1
                push!(fn, (idx, :add)) # take two layers with equal num of channels and adds their values
                push!(ch, ch[end])
                prettyprint(["\n($(length(fn))) ","shortcut($idx,$(length(fn)-1))"," => "],[:blue,:cyan,:green])
            elseif blocktype == :yolo
                push!(fn, nothing) # not a real layer. used for bounding boxes etc...
                push!(ch, ch[end])
                push!(cfg[:output], block)
                prettyprint(["($(length(fn))) ","YOLO"," || "],[:blue,:yellow,:green])
            end
        end

        # PART 2 - THE SKIPS
        ####################
        testimg = gpu(rand(Float32, cfg[:width], cfg[:height], cfg[:channels], batchsize))
        # find all skip-layers and all YOLO layers
        needout = sort(vcat(0, [l[1] for l in filter(f -> typeof(f) <: Tuple, fn)], findall(x -> x == nothing, fn) .- 1))
        chainstack = [] # layers that just feed forward can be grouped together in chains
        layer2out = Dict() # this dict translates layer numbers to chain numbers
        W = Dict{Int64, typeof(testimg)}() # this holds temporary outputs for use by skip-layers and YOLO output
        out = Array{Dict{Symbol, Any}, 1}(undef, 0) # store values needed for interpreting YOLO output
        println("\n\nGenerating chains and outputs: ")
        for i in 2:length(needout)
            print("$(i-1) ")
            fst, lst = needout[i-1]+1, needout[i] # these layers feed forward to an output
            if typeof(fn[fst]) == Nothing # check if sequence of layers begin with YOLO output
                push!(out, Dict(:idx => layer2out[fst-1]))
                fst += 1
            end
            # generate the functions used by the skip-layers and reference the temporary outputs
            for j in fst:lst
                if typeof(fn[j]) <: Tuple
                    arrayidx = layer2out[fn[j][1]]
                    if fn[j][2] == :route
                        fn[j] = x -> identity(W[arrayidx])
                    elseif fn[j][2] == :add
                        fn[j] = x -> x + W[arrayidx]
                    elseif fn[j][2] == :cat
                        fn[j] = x -> cat(x, W[arrayidx], dims = 3)
                    end
                end
            end
            push!(chainstack, Chain(fn[fst:lst]...)) # add sequence of functions to a chain
            push!(W, i-1 => chainstack[end](testimg).data) # generate a temporary array for the output of the chain
            testimg = chainstack[end](testimg).data
            push!(layer2out, [l => i-1 for l in fst:lst]...)
        end
        print("\n\n")
        cfg[:gridsize] = minimum([size(v, 1) for (k,v) in W]) # the gridsize is determined by the smallest matrix
        cfg[:layer2out] = layer2out
        push!(out, Dict(:idx => length(W)))

        # PART 3 - THE OUTPUTS
        ######################
        for i in eachindex(out)
            # we pre-process some linear matrix transformations and store the values for each YOLO output
            w, h, f, b = size(W[out[i][:idx]]) # width, height, filters, batchsize
            strideh = cfg[:height] ÷ h # stride height for this particular output
            stridew = cfg[:width] ÷ w # stride width
            anchormask = cfg[:output][i][:mask] .+ 1 # check which anchors are necessary from the config
            anchorvals = reshape(cfg[:output][i][:anchors], 2, :)[:, anchormask] ./ [stridew, strideh]
            # attributes are (normed and centered) - x, y, w, h, confidence, [number of classes]...
            attributes = 5 + cfg[:output][i][:classes]

            # precalculate the offset of prediction from cell-relative to (last) layer-relative
            offset = reshape(zerogen(w*h*2*length(anchormask)*b), w, h, 2, length(anchormask), b)
            for i in 0:w-1, j in 0:h-1
                offset[i+1, j+1, 1, :, :] = offset[i+1, j+1, 1, :, :] .+ i
                offset[i+1, j+1, 2, :, :] = offset[i+1, j+1, 2, :, :] .+ j
            end

            # precalculate the scale factor from layer-relative to image-relative
            scale = reshape(onegen(w*h*2*length(anchormask)*b), w, h, 2, length(anchormask), b)
            for i in 0:w-1, j in 0:h-1
                scale[i+1, j+1, 1, :, :] = scale[i+1, j+1, 1, :, :] .* stridew
                scale[i+1, j+1, 2, :, :] = scale[i+1, j+1, 2, :, :] .* strideh
            end

            # precalculate the anchor shapes to scale up the detection boxes
            anchor = reshape(onegen(w*h*2*length(anchormask)*b), w, h, 2, length(anchormask), b)
            for i in 1:length(anchormask)
                anchor[:, :, 1, i, :] = anchorvals[1, i] * stridew
                anchor[:, :, 2, i, :] = anchorvals[2, i] * strideh
            end

            out[i][:size] = (w, h, attributes, length(anchormask), b)
            out[i][:offset] = offset
            out[i][:scale] = scale
            out[i][:anchor] = anchor
            out[i][:truth] = cfg[:output][i][:truth_thresh] # for object being detected (at all)
            out[i][:ignore] = cfg[:output][i][:ignore_thresh] # for ignoring detections of same object (overlapping)
        end

        return new(cfg, chainstack, W, out)
    end
end

function Base.show(io::IO, yolo::Yolo)
    ln1 = "DarkNet $(yolo.cfg[:version])\n"
    ln2 = "WxH: $(yolo.cfg[:width])x$(yolo.cfg[:height])   channels: $(yolo.cfg[:channels])   batchsize: $(yolo.cfg[:batchsize])\n"
    ln3 = "gridsize: $(yolo.cfg[:gridsize])   classes: $(yolo.cfg[:output][1][:classes])   thresholds: $(get.(yolo.cfg[:output], :ignore_thresh, -1))"
    print(io, ln1 * ln2 * ln3)
end

########################################################
##### FUNCTIONS FOR INFERENCE ##########################
########################################################

include("gpukern.jl")

# Sets all values under a given threshold to zero
function clipdetect!(input::Array, conf)
   rows, cols = size(input)
   for i in 1:cols
       input[5, i] = ifelse(input[5, i] > conf, input[5, i], Float32(0))
   end
end

function clipdetect!(input::CuArray, conf)
    rows, cols = size(input)
    @cuda blocks=cols threads=1024 kern_clipdetect(input, conf)
end

# findmax, get the class with highest confidence and class number out.
function findmax!(input::CuArray, idst::Int, idend::Int)
    rows, cols = size(input)
    @cuda blocks=cols threads=rows kern_findmax!(input, idst, idend)
end

function findmax!(input::Array{T}, idst::Int, idend::Int) where {T}
    for i in 1:size(input, 2)
        input[end-2, i], input[end-1, i] = findmax(input[idst:idend, i])
    end
end

# Reduces the size of array and only keeps detections over threshold
function keepdetections(arr::Array)
    return arr[:, arr[5, :] .> 0]
end

function keepdetections(input::CuArray) # THREADS:BLOCKS CAN BE OPTIMIZED WITH BETTER KERNEL
    rows, cols = size(input)
    bools = CuArrays.zeros(Int32, cols)
    @cuda blocks=cols threads=rows kern_genbools(input, bools)
    idxs = cumsum(bools)
    n = count(bools)
    output = CuArray{Float32, 2}(undef, rows, n)
    @cuda blocks=cols threads=rows kern_keepdetections(input, output, bools, idxs)
    return output
end

# Bounding Box Intersection Over Union - removes overlapping boxes for same object
function bboxiou(box1, box2)
    b1x1, b1y1, b1x2, b1y2 = box1
    b2x1, b2y1, b2x2, b2y2 = box2[1, :], box2[2, :], box2[3, :], box2[4, :]
    rectx1 = max.(b1x1, b2x1)
    recty1 = max.(b1y1, b2y1)
    rectx2 = min.(b1x2, b2x2)
    recty2 = min.(b1y2, b2y2)
    z = zeros(length(rectx2))
    interarea = max.(rectx2 .- rectx1 .+ 1, z) .* max.(recty2 .- recty1 .+ 1, z)
    b1area = (b1x2 - b1x1 + 1) * (b1y2 - b1y1 + 1)
    b2area = (b2x2 .- b2x1 .+ 1) .* (b2y2 .- b2y1 .+ 1)
    iou = interarea ./ (b1area .+ b2area .- interarea)
    return iou
end


# Simply pass a batch of images to the Yolo object to do inference.
function (yolo::Yolo)(img::DenseArray)
    @assert ndims(img) == 4 # width, height, channels, batchsize
    yolo.W[0] = gpu(img)

    # FORWARD PASS
    ##############
    for i in eachindex(yolo.chain) # each chain writes to a predefined output
        yolo.W[i] = yolo.chain[i](yolo.W[i-1]).data
    end

    # PROCESSING EACH YOLO OUTPUT
    #############################
    outweights = []
    outnr = 0
    for out in yolo.out
        outnr += 1
        w, h, a, bo, ba = out[:size]
        weights = reshape(yolo.W[out[:idx]], w, h, a, bo, ba)
        # adjust the predicted box coordinates into pixel values
        weights[:, :, 1:2, :, :] = (σ.(weights[:, :, 1:2, :, :]) + out[:offset]) .* out[:scale]
        weights[:, :, 5:end, :, :] = σ.(weights[:, :, 5:end, :, :])
        weights[:, :, 3:4, :, :] = exp.(weights[:, :, 3:4, :, :]) .* out[:anchor]
        weights[:, :, 1, :, :] = weights[:, :, 1, :, :] .- weights[:, :, 3, :, :] .* 0.5
        weights[:, :, 2, :, :] = weights[:, :, 2, :, :] .- weights[:, :, 4, :, :] .* 0.5
        weights[:, :, 3, :, :] = weights[:, :, 3, :, :] .+ weights[:, :, 1, :, :]
        weights[:, :, 4, :, :] = weights[:, :, 4, :, :] .+ weights[:, :, 2, :, :]

        # add additional attributes for post-inference analysis: confidence, classnr, outnr, batchnr
        weights = cat(weights, zerogen(w, h, 4, bo, ba), dims = 3)
        weights[:, :, a+3, outnr, :] = outnr # write output number to attribute a+3
        for batch in 1:ba weights[:, :, a+4, :, batch] = batch end # write batchnumber to attribute a+4
        weights = permutedims(weights, [3, 1, 2, 4, 5]) # place attributes first
        weights = reshape(weights, a+4, :) # reshape to attr, data
        clipdetect!(weights, Float32(out[:truth])) # set all detections below conf-thresh to zero
        findmax!(weights, 6, a)
        push!(outweights, weights)
    end

    # PROCESSING ALL PREDICTIONS
    ############################

    batchout = cpu(keepdetections(cat(outweights..., dims=2)))
    size(batchout, 1) == 0 && return genzeros(1, 1)

    classes = unique(batchout[end-1, :])
    output = Array{Array{Float32, 2},1}(undef, 0)
    for c in classes
        detection = sortslices(batchout[:, batchout[end-1, :] .== c], dims = 2, by = x -> x[5], rev = true)
        for l in 1:size(detection, 2)
            iou = bboxiou(detection[1:4, l], detection[1:4, l+1:end])
            ds = findall(v -> v > yolo.out[1][:ignore], iou)
            detection = detection[:, setdiff(1:size(detection, 2), ds .+ l)]
            l >= size(detection, 2) && break
        end
        push!(output, detection)
    end
    return hcat(output...)
end

end
