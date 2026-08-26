# Throughput of one training step (forward + backward + optimiser update) for the
# YOLO backbone classifier, across backends and batch sizes.
#
#   julia --project -t auto bench.jl metal 32,64,128 224
#
# Run one configuration per process: Metal grows its buffer pool over the first
# few steps, so whichever configuration is measured first in a shared process
# pays for that and looks far slower than it is.
using Flux, Optimisers, Printf
using ObjectDetector, ObjectDetector.YOLO
include("backbone.jl")

backend = get(ARGS, 1, "cpu")
sizes = parse.(Int, split(get(ARGS, 2, "64"), ','))
res = parse(Int, get(ARGS, 3, "224"))
cfgfile = get(ARGS, 4, joinpath(YOLO.models_dir(), "yolov3-tiny.cfg"))

sync = () -> nothing
todevice = identity
if backend == "metal"
    using Metal
    sync = Metal.synchronize
    todevice = Flux.gpu
elseif backend == "accelerate"
    using AppleAccelerate
end

for bs in sizes
    yolo = YOLO.Yolo(cfgfile, nothing, 1; silent = true, use_gpu = false,
        weights_stop_layer = 0, trainable_batchnorm = true)
    m = todevice(classifier(yolo; nclasses = 10, nconv = 9))
    x = todevice(randn(Float32, res, res, 3, bs))
    y = todevice(Flux.onehotbatch(rand(1:10, bs), 1:10))
    st = Optimisers.setup(Optimisers.AdamW(1.0f-3), m)
    # Written out rather than wrapped in a closure: closing over `m` and `st`,
    # which are reassigned every step, boxes them and makes the whole forward
    # pass dynamically dispatched, which shows up as a slower step.
    for _ in 1:4
        _, g = Flux.withgradient(mm -> Flux.logitcrossentropy(mm(x), y), m)
        st, m = Optimisers.update!(st, m, g[1])
    end
    sync()
    n = 8
    t = time()
    for _ in 1:n
        _, g = Flux.withgradient(mm -> Flux.logitcrossentropy(mm(x), y), m)
        st, m = Optimisers.update!(st, m, g[1])
    end
    sync()
    dt = (time() - t) / n
    @printf("%-11s res=%d bs=%-4d threads=%-2d  %6.3f s/step  %6.1f img/s\n",
        backend, res, bs, Threads.nthreads(), dt, bs / dt)
end
