# [Pre-training a backbone on ImageNet](@id imagenet-backbone)

This tutorial trains a YOLO model's convolutional trunk as an image classifier
using [ImageNetDataset.jl](https://github.com/Julia-XAI/ImageNetDataset.jl), then
writes it back out as Darknet weights that [`train!`](@ref training) can fine-tune
into a detector.

The runnable code is in
[`examples/imagenet_backbone/`](https://github.com/r3tex/ObjectDetector.jl/tree/master/examples/imagenet_backbone).

## What ImageNet can and cannot train

ImageNet (ILSVRC-2012, the dataset ImageNetDataset.jl loads) is a *classification*
dataset: one label per image, no bounding boxes. The YOLO loss needs boxes, so
**a detector cannot be trained on ImageNet end to end**.

What ImageNet is for in the YOLO recipe is the trunk. Darknet distributes the
result as `yolov3-tiny.conv.15`: the first 15 layers, used as the initialisation
for detection training on a dataset that does have boxes. That is the file this
tutorial produces, and it is the same split the package's own fine-tuning recipe
restores with `weights_stop_layer = 15`.

Reading `yolov3-tiny.cfg`, the split is:

| Darknet layers | Contents | Role |
|:--|:--|:--|
| 0-14 | 9 convolutions and 6 max-pools | trunk, the `conv.15` split |
| 15-23 | detection convolutions, route, upsample, two `[yolo]` heads | needs boxes |

Layers 0-12 (7 convolutions, ending at 1024 channels) are the part both detection
branches share; the example takes `--stop_layer` to choose between them, counting
cfg blocks the same way `weights_stop_layer` does.

## The data

ImageNetDataset.jl deliberately does not download ImageNet: it is served from
[image-net.org](https://image-net.org/) behind a registration and terms-of-access
form. Download it manually, following
[the package's installation instructions](https://Julia-XAI.github.io/ImageNetDataset.jl/dev/installation/),
and you get:

```
ILSVRC/
├── train/n01440764/n01440764_10026.JPEG ...
├── val/n01440764/ILSVRC2012_val_00000293.JPEG ...
└── devkit/data/meta.mat
```

which loads with:

```julia
using ImageNetDataset
trainset = ImageNet(:train; dir = "/path/to/ILSVRC")
valset   = ImageNet(:val;   dir = "/path/to/ILSVRC")
```

### Working on a subset

The full training split is 1.28M images, about 140 GB. To get the pipeline running
first, [Imagenette](https://github.com/fastai/imagenette) is a 10-class subset of
*real* ImageNet images, distributed without registration and already in the
`<split>/<wnid>/*.JPEG` layout ImageNetDataset expects:

```
curl -LO https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-320.tgz
tar xzf imagenette2-320.tgz
```

Use the `-320` build rather than `-160`, so there are enough pixels for a 224x224
crop without upscaling.

`ImageNet(split; dir)` will not read a subset, because that constructor reads the
devkit metadata and asserts the exact ILSVRC file counts:

```julia
@assert length(paths) == TRAINSET_SIZE   # 1_281_167
```

So build the struct directly. Everything downstream, the transforms, indexing,
`convert2image`, `class`, is unchanged:

```julia
function imagenet_subset(root, split; transform = CenterCropNormalize(), classnames = IMAGENETTE_CLASSES)
    dir = joinpath(root, String(split))
    paths = ImageNetDataset.get_file_paths(dir)
    imagewnids = ImageNetDataset.path_to_wnid.(paths)
    wnids = sort!(unique(imagewnids))
    wnid_to_label = Dict(wnids .=> eachindex(wnids))
    metadata = Dict{String, Any}(
        "class_WNIDs" => wnids,
        "class_names" => [get(classnames, w, [w]) for w in wnids],
        "class_description" => [join(get(classnames, w, [w]), ", ") for w in wnids],
        "wnid_to_label" => wnid_to_label,
    )
    targets = [wnid_to_label[w] for w in imagewnids]
    return ImageNet(split, transform, paths, targets, metadata)
end
```

Labels are assigned by sorted WNID, which is what ImageNetDataset does for the full
set too, so the two paths stay consistent.

### Preprocessing

ImageNetDataset supplies the transforms: random crops for training, center crops
for validation, both normalised with the usual ImageNet channel statistics.

```julia
RandomCropNormalize(; output_size = (224, 224), open_size = (256, 256))
CenterCropNormalize(; output_size = (224, 224), open_size = (256, 256))
```

`open_size` is passed to JpegTurbo as a preferred decode size, so the JPEG is
DCT-scaled down *during* decode rather than after. That matters, because decoding
is the input bottleneck. Transforms return WHC arrays, already in Flux's layout,
and batches are assembled across threads:

```julia
function makebatch(ds::ImageNet, idxs::AbstractVector{Int}, ncls::Int)
    features = ds[first(idxs)].features
    w, h, c = size(features)
    X = Array{Float32, 4}(undef, w, h, c, length(idxs))
    X[:, :, :, 1] .= features
    @sync for k in 2:lastindex(idxs)
        Threads.@spawn X[:, :, :, k] .= ds[idxs[k]].features
    end
    return X, onehotbatch(ds.targets[idxs], 1:ncls)
end
```

Run Julia with `-t auto` or the decode is serial.

## Getting the trunk out of the model

[`backbone`](@ref) does this. Starting from a randomly initialised detector with
live batch-norm:

```julia
yolo = YOLO.Yolo(cfg, nothing, 1; silent = true, use_gpu = false,
                 weights_stop_layer = 0,     # random-init everything
                 trainable_batchnorm = true) # live batch-norm to train

trunk = backbone(yolo, 15)
```

`stop_layer` counts cfg blocks, the same unit `weights_stop_layer` uses, so the
two halves of the round trip are expressed the same way: 15 is darknet's
`yolov3-tiny.conv.15` split, nine convolutions, and 13 stops at the
1024-channel trunk both detection branches share.

Two things are worth knowing about what comes back.

The chain holds the *same* `Conv` and `BatchNorm` objects the detector holds, not
copies. `Flux.update!` updates in place, which is what the package's own
[`train!`](@ref training) relies on, so training the trunk on CPU trains `yolo`
directly. On a GPU it does not, because `gpu` copies the arrays to the device.

The activations are substituted. The inference path applies them with
`broadcast!` into their own input:

```julia
(l::BroadcastActivation)(x) = broadcast!(l.act, x, x)
```

which is what you want for inference throughput and exactly what Zygote cannot
differentiate through, so `backbone` swaps in a non-mutating `PureActivation`
that defers to the same `_pure` rule the detector's training path uses.

This only works for a trunk that is a plain feed-forward stack. CSP-style
backbones route within the trunk, and `backbone` says so rather than returning
something broken:

```julia
julia> backbone(v4_tiny, 30)
ERROR: ArgumentError: cfg blocks 1:30 contain a RouteLayer, which reads another
layer's output; this model's backbone cannot be split off as a plain chain
```

Adding a global-average-pool head gives the classifier darknet pre-trains with:

```julia
function classifier(yolo; nclasses::Int, stop_layer::Int = 15)
    trunk = backbone(yolo, stop_layer)
    channels = size(last(filter(l -> l isa Flux.Conv, collect(trunk))).weight, 4)
    head = Chain(GlobalMeanPool(), Flux.flatten, Dense(channels => nclasses))
    return Chain(; backbone = trunk, classifier = head)
end
```

At 224x224 the five stride-2 max-pools take the feature map to 7x7, and
`stop_layer = 15` leaves 512 channels there: 7.74M parameters including the head.

## Training

Nothing exotic: `AdamW`, logit cross-entropy, one-hot targets.

```julia
model = todevice(classifier(yolo; nclasses = ncls, stop_layer = opts.stop_layer))
state = Flux.setup(Flux.AdamW(opts.lr), model)

for epoch in 1:opts.epochs
    Flux.trainmode!(model)
    for (X, Y) in batches(trainset, ncls; batchsize = opts.batchsize)
        x, y = todevice(X), todevice(Y)
        loss, grads = Flux.withgradient(m -> Flux.logitcrossentropy(m(x), y), model)
        Flux.update!(state, model, grads[1])
    end
end
```

```
julia --project -t auto train.jl --data /path/to/imagenette2-320 --epochs 5
```

Add `--full` to go through `ImageNet(split; dir)` instead of the subset loader,
and `--stop_layer 13` to stop at the shared 1024-channel trunk.

## Results

Five epochs on Imagenette from random initialisation, on an Apple M5 Pro via
Metal:

```
backend metal, 5 threads | 10 classes, 9469 train / 3925 val images
yolov3-tiny.cfg: cfg blocks 1-15, 7742874 parameters at 224 x 224
  epoch 1 done in 107.0s | train loss 1.4714 | val loss 1.4871 | val acc 0.537
  epoch 2 done in  55.6s | train loss 1.0754 | val loss 1.3995 | val acc 0.581
  epoch 3 done in  55.6s | train loss 0.9349 | val loss 1.0137 | val acc 0.661
  epoch 4 done in  53.9s | train loss 0.8359 | val loss 0.8996 | val acc 0.706
  epoch 5 done in  54.4s | train loss 0.7511 | val loss 1.3316 | val acc 0.610
9 conv blocks in the detector's trunk; wrote runs/yolov3-tiny-imagenet.weights
```

About 55 s per epoch after the first, which carries compilation: roughly 170
images/s including the input pipeline.

Read the accuracy column with suspicion. Training loss falls steadily, but
validation accuracy peaks at 0.706 in epoch 4 and drops to 0.610 in epoch 5, and
a second run of the same script peaked at 0.728. 9469 images over 10 classes is
orders of magnitude less data than the trunk wants, and that gap is what
overfitting on a subset looks like: the number moves several points run to run
and the last epoch is not reliably the best one.

So this is a working pipeline, not a pre-trained backbone. A real run is the full
1.28M images for on the order of a hundred epochs; the point of the subset is to
exercise every part of the pipeline before committing to that.

## Handing the trunk back to the detector

Training on Metal leaves the trained values on the device, so they are copied
back into the model before it is written out with the package's own
[`save_weights`](@ref):

```julia
ncopied = copy_backbone!(yolo, Flux.cpu(model)[:backbone])
save_weights(yolo, joinpath(opts.out, "yolov3-tiny-imagenet.weights"))
```

[`copy_backbone!`](@ref) pairs the model's convolutions with the trained ones and
`copyto!`s kernels and batch-norm parameters across. On CPU it is a self-copy,
since `Flux.update!` has already written through. Because the
model was built with `trainable_batchnorm = true`, `save_weights` writes true
Darknet batch-norm parameters rather than folded identity ones.

The result is an ordinary Darknet weights file, 35,434,956 bytes, that reloads
with the same cfg and fine-tunes with [`train!`](@ref training):

```julia
m = YOLO.Yolo(cfg, "runs/yolov3-tiny-imagenet.weights", 1; trainable_batchnorm = true)
res = train!(m, samples; epochs = 3, batchsize = 4, lr = 1f-4)
```

```
reloaded ok, input size (416, 416, 3, 1)
inference ok, result cols: 112
train! losses: Float32[58.6824, 53.6988, 49.9396]
```

The detection heads are still random, which is what you want: they are the layers
`train!` learns on the boxed dataset. Only the trunk carries anything over from
ImageNet, which is exactly the division of labour `weights_stop_layer = 15`
expresses in the [fine-tuning recipe](@ref training).

## Speed

See [Acceleration](@ref acceleration) for the backend comparison and the
Metal-specific pitfalls behind these numbers.
