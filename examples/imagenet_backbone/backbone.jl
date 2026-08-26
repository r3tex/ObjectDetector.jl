# Turn the convolutional trunk of an ObjectDetector YOLO model into a Flux
# image classifier, and put the trained weights back afterwards.
#
# The layers themselves come from the package: `YOLO.Yolo` parses the `.cfg`,
# builds the convolutions and batch-norms and initialises them, so the trunk
# trained here is the same object the detector runs, not a re-implementation.

using Flux
using ObjectDetector
using ObjectDetector: YOLO

"""
    yolo_model(m) -> YOLO.Yolo

Unwrap the allocator wrapper that `YOLO.Yolo` returns on CPU.
"""
yolo_model(m::YOLO.Yolo) = m
yolo_model(m::ObjectDetector.AllocWrappedModel) = m.model

"""
    PureActivation(act)

Non-mutating stand-in for `YOLO.BroadcastActivation`, which writes its result
back into its input. That is what the inference path wants and what Zygote
cannot differentiate, so the trunk is rebuilt with this in its place. The
package does the same thing internally in `YOLO._pure`.
"""
struct PureActivation{F}
    act::F
end
(l::PureActivation)(x) = l.act.(x)
Base.show(io::IO, l::PureActivation) = print(io, "PureActivation(", l.act, ")")

pure(l::YOLO.BroadcastActivation) = PureActivation(l.act)
pure(l) = l

"""
    backbone_layers(yolo; nconv = 9) -> Vector

The flattened layers of the first `nconv` `[convolutional]` blocks, including
the batch-norms, activations and max-pools between them.

`nconv = 9` is darknet's `yolov3-tiny.conv.15` split, which is what the
package's own fine-tuning recipe restores with `weights_stop_layer = 15`.
`nconv = 7` stops at the 1024-channel trunk instead, the part both detection
branches share.
"""
function backbone_layers(yolo; nconv::Int = 9)
    flat = Any[]
    YOLO._flatten_layers!(flat, yolo_model(yolo).chain)
    layers = Any[]
    seen = 0
    for l in flat
        l isa Flux.Conv && (seen += 1)
        seen > nconv && break
        push!(layers, l)
    end
    seen > nconv || error("model has only $seen convolutional layers, need more than $nconv")
    return layers
end

"""
    classifier(yolo; nclasses, nconv = 9)

Trunk plus a global-average-pool classification head, which is how darknet
pre-trains a backbone on ImageNet. Returns a named `Chain` so the trunk can be
recovered with `model[:backbone]`.

The convolutions and batch-norms are the *same objects* the detector holds, but
`Optimisers.update!` rebuilds them into fresh arrays, so training does not
write through to `yolo`. Use [`copy_backbone!`](@ref) to put the result back.
"""
function classifier(yolo; nclasses::Int, nconv::Int = 9)
    layers = map(pure, backbone_layers(yolo; nconv))
    channels = size(last(filter(l -> l isa Flux.Conv, layers)).weight, 4)
    backbone = Chain(layers...)
    head = Chain(GlobalMeanPool(), Flux.flatten, Dense(channels => nclasses))
    return Chain(; backbone, classifier = head)
end

"""
    copy_backbone!(yolo, trained)

Copy the trained trunk parameters back into `yolo`, in place, so the model can
be written out with `save_weights` and reloaded as a detector. `trained` is the
backbone `Chain` from [`classifier`](@ref); its convolutions and batch-norms are
matched in order against the head of the model's own flattened layer list.
"""
function copy_backbone!(yolo, trained)
    dstlayers = Any[]
    YOLO._flatten_layers!(dstlayers, yolo_model(yolo).chain)
    srclayers = Any[]
    YOLO._flatten_layers!(srclayers, trained)
    # Match on the layers that hold parameters rather than by position, so this
    # keeps working if the trunk is rebuilt with a different layer count.
    hasparams(l) = l isa Flux.Conv || l isa Flux.BatchNorm
    ncopied = 0
    for (dst, src) in zip(filter(hasparams, dstlayers), filter(hasparams, srclayers))
        if dst isa Flux.Conv
            src isa Flux.Conv || error("layer mismatch: $(typeof(dst)) vs $(typeof(src))")
            copyto!(dst.weight, src.weight)
            dst.bias isa AbstractArray && copyto!(dst.bias, src.bias)
            ncopied += 1
        else
            src isa Flux.BatchNorm || error("layer mismatch: $(typeof(dst)) vs $(typeof(src))")
            copyto!(dst.β, src.β); copyto!(dst.γ, src.γ)
            copyto!(dst.μ, src.μ); copyto!(dst.σ², src.σ²)
        end
    end
    return ncopied
end
