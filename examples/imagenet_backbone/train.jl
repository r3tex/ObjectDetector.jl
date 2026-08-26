# Pre-train a YOLO backbone on ImageNet classification, then hand it to
# ObjectDetector as darknet weights.
#
# ImageNet has no bounding boxes, so this trains the convolutional trunk only,
# which is the role darknet's `yolov3-tiny.conv.15` plays: an initialisation for
# detection training, which then happens with `train!` on a dataset that does
# have boxes.
#
#   julia --project -t auto train.jl --data /path/to/imagenette2-320 --epochs 5
#
# The default reads any tree laid out as `<split>/<wnid>/*.JPEG`, which covers
# both Imagenette and the full ILSVRC-2012 train/val directories. Pass `--full`
# to go through `ImageNet(split; dir)` instead, which also reads the devkit
# metadata and checks the official file counts.

# The backend decides which packages get loaded, so it is read straight out of
# ARGS here rather than through the option parser below.
const BACKEND = let
    i = findfirst(==("--backend"), ARGS)
    requested = i === nothing ? "auto" : ARGS[i + 1]
    # Metal is roughly 2.5x the CPU on this model; `bench.jl` prints the
    # comparison. Pass `--backend cpu` to force Apple's Accelerate BLAS instead.
    requested == "auto" ? (Sys.isapple() && Sys.ARCH === :aarch64 ? "metal" : "cpu") : requested
end

using Printf, Statistics, Random
using Flux, JLD2
using ObjectDetector, ObjectDetector.YOLO

if Sys.isapple()
    # Puts Apple's Accelerate BLAS behind the im2col GEMM that NNlib's CPU
    # convolutions are built on.
    using AppleAccelerate
end
if BACKEND == "metal"
    using Metal
end

const USE_METAL = BACKEND == "metal" && Metal.functional()
BACKEND == "metal" && !USE_METAL && @warn "Metal is not functional here, falling back to the CPU."

include("backbone.jl")
include("data.jl")

todevice(x) = USE_METAL ? Flux.gpu(x) : Flux.cpu(x)
sync() = USE_METAL ? Metal.synchronize() : nothing

#####
##### Options
#####

const DEFAULTS = Dict{Symbol, Any}(
    :data => joinpath(@__DIR__, "imagenette2-320"),
    :out => joinpath(@__DIR__, "runs"),
    :cfg => joinpath(YOLO.models_dir(), "yolov3-tiny.cfg"),
    :backend => USE_METAL ? "metal" : "cpu",
    :epochs => 5,
    :batchsize => 64,
    :lr => 3.0f-4,
    :res => 224,
    :stop_layer => 15,
    :full => false,
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
        haskey(opts, key) || error("unknown option $arg")
        if opts[key] isa Bool
            opts[key] = true
            i += 1
        else
            i < lastindex(argv) || error("$arg needs a value")
            opts[key] = parsevalue(opts[key], argv[i + 1])
            i += 2
        end
    end
    return NamedTuple(opts)
end

#####
##### Train / evaluate
#####

function evaluate(model, ds, ncls, batchsize)
    correct = total = 0
    loss = 0.0
    for (X, Y) in batches(ds, ncls; batchsize, shuffle = false)
        ŷ = Flux.cpu(model(todevice(X)))
        loss += Flux.logitcrossentropy(ŷ, Y) * size(X, 4)
        correct += count(Flux.onecold(ŷ) .== Flux.onecold(Y))
        total += size(X, 4)
    end
    return loss / total, correct / total
end

function main(opts)
    # ImageNet's convention is to open the image a little larger than the crop,
    # so the crop has room to move.
    open_size = round(Int, opts.res * 256 / 224)
    load = opts.full ? imagenet : imagenet_subset
    trainset = load(opts.data, :train;
        transform = RandomCropNormalize(;
            output_size = (opts.res, opts.res), open_size = (open_size, open_size)))
    valset = load(opts.data, :val;
        transform = CenterCropNormalize(;
            output_size = (opts.res, opts.res), open_size = (open_size, open_size)))
    ncls = nclasses(trainset)

    # A randomly initialised detector. Its trunk is what gets trained; the
    # detection heads stay random and are trained later by `train!` on boxes.
    yolo = YOLO.Yolo(opts.cfg, nothing, 1; silent = true, use_gpu = false,
        weights_stop_layer = 0, trainable_batchnorm = true)
    model = todevice(classifier(yolo; nclasses = ncls, stop_layer = opts.stop_layer))
    # `Flux.update!` updates in place, which is what the package's own `train!`
    # does. On CPU the trunk's arrays are shared with `yolo`, so this trains the
    # detector's trunk directly.
    state = Flux.setup(Flux.AdamW(opts.lr), model)

    @printf("backend %s, %d threads | %d classes, %d train / %d val images\n",
        opts.backend, Threads.nthreads(), ncls, length(trainset), length(valset))
    @printf("%s: cfg blocks 1-%d, %d parameters at %d x %d\n",
        basename(opts.cfg), opts.stop_layer, sum(length, Flux.trainables(model)), opts.res, opts.res)

    mkpath(opts.out)
    nb = nbatches(trainset, opts.batchsize)
    for epoch in 1:opts.epochs
        Flux.trainmode!(model)
        running = 0.0
        seen = 0
        t0 = time()
        for (i, (X, Y)) in enumerate(batches(trainset, ncls; batchsize = opts.batchsize))
            x, y = todevice(X), todevice(Y)
            loss, grads = Flux.withgradient(m -> Flux.logitcrossentropy(m(x), y), model)
            Flux.update!(state, model, grads[1])
            running += loss * size(X, 4)
            seen += size(X, 4)
            if i % 10 == 0 || i == nb
                # GPU work is queued asynchronously, so the rate is only real
                # once the queue has drained.
                sync()
                @printf("\r  epoch %d  %4d/%-4d  loss %.4f  %5.1f img/s",
                    epoch, i, nb, running / seen, seen / (time() - t0))
            end
        end
        sync()
        traintime = time() - t0
        print("\n")

        Flux.testmode!(model)
        vloss, vacc = evaluate(model, valset, ncls, opts.batchsize)
        @printf("  epoch %d done in %.1fs | train loss %.4f | val loss %.4f | val acc %.3f\n",
            epoch, traintime, running / seen, vloss, vacc)

        jldsave(joinpath(opts.out, "checkpoint.jld2");
            state = Flux.state(Flux.cpu(model)), epoch, opts = Dict(pairs(opts)))
    end

    # Put the trained trunk back into the detector and write darknet weights,
    # which reload with the same cfg and can be fine-tuned with `train!`.
    ncopied = copy_backbone!(yolo, Flux.cpu(model)[:backbone])
    weightfile = joinpath(opts.out, "yolov3-tiny-imagenet.weights")
    save_weights(yolo, weightfile)
    @printf("%d conv blocks in the detector's trunk; wrote %s\n", ncopied, weightfile)
    return model
end

if abspath(PROGRAM_FILE) == @__FILE__
    main(parseargs(ARGS))
end
