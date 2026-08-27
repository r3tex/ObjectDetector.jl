# The classification head that turns a YOLO trunk into an ImageNet classifier.
#
# Getting the trunk out of the model, and putting trained weights back, is the
# package's job: see `backbone` and `copy_backbone!`.

using Flux
using ObjectDetector
using ObjectDetector: backbone

"""
    classifier(yolo; nclasses, stop_layer = 15)

Trunk plus a global-average-pool classification head, which is how darknet
pre-trains a backbone on ImageNet. Returns a named `Chain`, so the trunk can be
recovered with `model[:backbone]` and handed to `copy_backbone!`.
"""
function classifier(yolo; nclasses::Int, stop_layer::Int = 15)
    trunk = backbone(yolo, stop_layer)
    channels = size(last(filter(l -> l isa Flux.Conv, collect(trunk))).weight, 4)
    head = Chain(GlobalMeanPool(), Flux.flatten, Dense(channels => nclasses))
    return Chain(; backbone = trunk, classifier = head)
end
