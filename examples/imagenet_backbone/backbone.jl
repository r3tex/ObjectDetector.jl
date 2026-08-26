# Turn the convolutional trunk of an ObjectDetector YOLO model into a Flux image
# classifier, and get the trained weights back into the model afterwards.
#
# The layers come from the package: `YOLO.Yolo` parses the `.cfg`, builds the
# convolutions and batch-norms and initialises them, so the trunk trained here is
# the same object the detector runs, not a re-implementation.

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
    PureActivation(layer)

Non-mutating stand-in for `YOLO.BroadcastActivation`, which writes its result
back into its own input. That is what the inference path wants and what Zygote
cannot differentiate through.

The substitution itself is the package's: this defers to `YOLO._pure`, which is
what the detector's own training path uses for the same reason. `outs` is empty
because the trunk contains no skip layers - `backbone_layers` stops before the
first route.

Only the activation is wrapped, never a `Conv` or `BatchNorm`: this type is not
a Functors node, so anything inside it would be invisible to `Flux.setup`.
"""
struct PureActivation{L}
    layer::L
end
(p::PureActivation)(x) = YOLO._pure(p.layer, (), x)
Base.show(io::IO, p::PureActivation) = print(io, "PureActivation(", p.layer.act, ")")

pure(l::YOLO.BroadcastActivation) = PureActivation(l)
pure(l) = l

"""
    backbone_layers(yolo; stop_layer = 15) -> Vector

The flattened layers of the first `stop_layer` cfg blocks, including the
batch-norms, activations and max-pools between them.

`stop_layer` counts cfg blocks, the same unit `YOLO.Yolo`'s `weights_stop_layer`
uses, so the default of 15 is darknet's `yolov3-tiny.conv.15` split: layers 0-14,
nine convolutions. Pass 13 to stop at the 1024-channel trunk instead, the part
both detection branches share.
"""
function backbone_layers(yolo; stop_layer::Int = 15)
    model = yolo_model(yolo)
    blocks = get(model.cfg, :layerblocks, nothing)
    blocks === nothing && error("model does not carry its cfg layer blocks; \
                                 build it with a current version of ObjectDetector")
    1 <= stop_layer <= length(blocks) ||
        error("stop_layer=$stop_layer is outside the model's $(length(blocks)) cfg blocks")
    nconv = count(b -> first(b) === :convolutional, blocks[1:stop_layer])

    flat = Any[]
    YOLO._flatten_layers!(flat, model.chain)
    layers = Any[]
    seen = 0
    for l in flat
        l isa Flux.Conv && (seen += 1)
        seen > nconv && break
        push!(layers, l)
    end
    return layers
end

"""
    classifier(yolo; nclasses, stop_layer = 15)

Trunk plus a global-average-pool classification head, which is how darknet
pre-trains a backbone on ImageNet. Returns a named `Chain` so the trunk can be
recovered with `model[:backbone]`.

The convolutions and batch-norms are the *same objects* the detector holds, so
training this with `Flux.update!`, which updates in place, trains `yolo` too.
"""
function classifier(yolo; nclasses::Int, stop_layer::Int = 15)
    layers = map(pure, backbone_layers(yolo; stop_layer))
    channels = size(last(filter(l -> l isa Flux.Conv, layers)).weight, 4)
    backbone = Chain(layers...)
    head = Chain(GlobalMeanPool(), Flux.flatten, Dense(channels => nclasses))
    return Chain(; backbone, classifier = head)
end

"""
    copy_backbone!(yolo, trained) -> Int

Copy the trained trunk parameters into `yolo` and return the number of
convolutions copied.

Only needed when the trunk was trained on a GPU: `Flux.gpu` copies the arrays to
the device, so the trained values have to come home before `save_weights`. On
CPU the arrays are shared and `Flux.update!` has already written through, which
makes this a self-copy.
"""
function copy_backbone!(yolo, trained)
    ncopied = 0
    for ((conv, bn), (tconv, tbn)) in zip(YOLO.conv_bn_layers(yolo_model(yolo).chain),
                                          YOLO.conv_bn_layers(trained))
        copyto!(conv.weight, tconv.weight)
        conv.bias isa AbstractArray && copyto!(conv.bias, tconv.bias)
        if bn !== nothing
            tbn === nothing && error("model layer $(ncopied + 1) has batch-norm but the trained one does not")
            copyto!(bn.β, tbn.β); copyto!(bn.γ, tbn.γ)
            copyto!(bn.μ, tbn.μ); copyto!(bn.σ², tbn.σ²)
        end
        ncopied += 1
    end
    return ncopied
end
