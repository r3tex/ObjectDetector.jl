# Train a YOLO detector on COCO.
#
#   julia --project -t auto train.jl --data /path/to/coco --epochs 5
#
# `--data` is a directory holding COCO's `val2017/` (or `train2017/`) images and
# `annotations/instances_<split>.json`, laid out as the official zips extract.
#
# By default this fine-tunes the pretrained COCO weights on all 80 classes, which
# is the cheapest way to see the loss move. `--classes person,car` cuts the model
# down to a subset, and `--from` chooses what the weights start as: the pretrained
# detector, a backbone from the ImageNet example, or nothing at all.

const BACKEND = let
    i = findfirst(==("--backend"), ARGS)
    requested = i === nothing ? "auto" : ARGS[i + 1]
    requested == "auto" ? (Sys.isapple() && Sys.ARCH === :aarch64 ? "metal" : "cpu") : requested
end

using Printf, Random
using Flux, FileIO, ImageIO
using ObjectDetector, ObjectDetector.YOLO

Sys.isapple() && using AppleAccelerate
BACKEND == "metal" && using Metal

include("coco.jl")

const DEFAULTS = Dict{Symbol, Any}(
    :data => joinpath(@__DIR__, "coco"),
    :split => "val2017",
    :out => joinpath(@__DIR__, "runs"),
    :cfg => "v3_tiny_COCO",
    :from => "pretrained",
    :weights => "",
    :classes => "",
    :epochs => 5,
    :batchsize => 8,
    :lr => 1.0f-4,
    :res => 416,
    :limit => 0,
    :warmup => 0,
)

parsevalue(::Int, s) = parse(Int, s)
parsevalue(::Float32, s) = parse(Float32, s)
parsevalue(::AbstractString, s) = String(s)

function parseargs(argv)
    opts = copy(DEFAULTS)
    i = firstindex(argv)
    while i <= lastindex(argv)
        arg = argv[i]
        startswith(arg, "--") || error("unexpected argument $arg")
        key = Symbol(arg[3:end])
        haskey(opts, key) || key === :backend || error("unknown option $arg")
        if key === :backend
            i += 2
            continue
        end
        i < lastindex(argv) || error("$arg needs a value")
        opts[key] = parsevalue(opts[key], argv[i + 1])
        i += 2
    end
    return NamedTuple(opts)
end

"""
    build_model(opts, nclasses)

The detector to train. Changing the class count means changing both the `[yolo]`
layers and the convolution feeding each one, whose filter count is
`anchors_per_head * (5 + classes)`.
"""
function build_model(opts, nclasses)
    cfgfile, weightfile = YOLO.YOLO_MODELS[opts.cfg]()
    # Concretely typed: `overridecfg!` dispatches on the element type, and a
    # Vector{Any} of the same tuples is a MethodError.
    changes = Tuple{Symbol, Int, Symbol, Int}[
        (:net, 1, :width, opts.res), (:net, 1, :height, opts.res)]
    if nclasses != 80
        for (i, (blockidx, nanchors)) in enumerate(head_conv_blocks(cfgfile))
            push!(changes, (:yolo, i, :classes, nclasses))
            push!(changes, (:convolutional, blockidx, :filters, nanchors * (5 + nclasses)))
        end
    end
    if opts.from == "pretrained"
        # Reuse everything the pretrained detector learned, except the heads when
        # the class count changed.
        stop = nclasses == 80 ? nothing : 15
        return YOLO.Yolo(cfgfile, weightfile, 1; silent = true, cfgchanges = changes,
            weights_stop_layer = stop, trainable_batchnorm = stop !== nothing)
    elseif opts.from == "backbone"
        isfile(opts.weights) || error("--from backbone needs --weights <file>")
        return YOLO.Yolo(cfgfile, opts.weights, 1; silent = true, cfgchanges = changes,
            weights_stop_layer = 15, trainable_batchnorm = true)
    elseif opts.from == "scratch"
        return YOLO.Yolo(cfgfile, nothing, 1; silent = true, cfgchanges = changes,
            weights_stop_layer = 0, trainable_batchnorm = true)
    end
    error("unknown --from $(opts.from); expected pretrained, backbone or scratch")
end

# For each [yolo], the convolution feeding it and how many anchors it uses. The
# convolution is counted in convolutional blocks, which is what cfgchanges
# indexes, and its filter count is anchors * (5 + classes).
function head_conv_blocks(cfgfile)
    nconv = 0
    out = Tuple{Int, Int}[]
    for (blocktype, block) in YOLO.cfgread(cfgfile)[2:end]
        blocktype === :convolutional && (nconv += 1)
        blocktype === :yolo || continue
        nanchors = haskey(block, :mask) ? length(block[:mask]) : length(block[:anchors]) ÷ 2
        push!(out, (nconv, nanchors))
    end
    return out
end

function main(opts)
    imagedir = joinpath(opts.data, opts.split)
    annfile = joinpath(opts.data, "annotations", "instances_$(opts.split).json")
    isdir(imagedir) || error("no images at $imagedir")
    isfile(annfile) || error("no annotations at $annfile")

    classes = isempty(opts.classes) ? nothing : String.(split(opts.classes, ','))
    data = coco_samples(imagedir, annfile; classes, limit = opts.limit)
    isempty(data) && error("no usable images; check --classes")
    nclasses = classes === nothing ? 80 : length(classes)

    model = build_model(opts, nclasses)
    @printf("%s %s, %d classes | %d images, %d boxes | from %s\n",
        opts.cfg, opts.split, nclasses, length(data), nboxes(data), opts.from)
    @printf("backend %s, %d threads, %dx%d, batch %d\n",
        BACKEND, Threads.nthreads(), opts.res, opts.res, opts.batchsize)

    mkpath(opts.out)
    t0 = time()
    res = train!(model, data;
        epochs = opts.epochs, batchsize = opts.batchsize, lr = opts.lr,
        image_loader = FileIO.load, warmup_batches = opts.warmup,
        flip_augment = true, checkpoint_dir = opts.out, silent = false)
    @printf("%d epochs in %.1f min | loss %.3f -> %.3f\n",
        opts.epochs, (time() - t0) / 60, first(res.losses), last(res.losses))

    weightfile = joinpath(opts.out, "coco-finetuned.weights")
    save_weights(model, weightfile)
    println("wrote ", weightfile)
    return model, res
end

if abspath(PROGRAM_FILE) == @__FILE__
    main(parseargs(ARGS))
end
