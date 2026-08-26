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
branches share; the example takes `--nconv` to choose between them.

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

The layers come from the package rather than being rebuilt: `YOLO.Yolo` already
parses the cfg, constructs the convolutions and batch-norms and initialises them.
Starting from a randomly initialised detector with live batch-norm:

```julia
yolo = YOLO.Yolo(cfg, nothing, 1; silent = true, use_gpu = false,
                 weights_stop_layer = 0,     # random-init everything
                 trainable_batchnorm = true) # live batch-norm to train
```

Its `chain` is a `Chain` of sub-chains, grouped so the skip layers can find their
buffers. Flattening it gives a plain layer list, and the trunk is the prefix up to
the `nconv`-th convolution:

```julia
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
    return layers
end
```

For `yolov3-tiny.cfg` that is 33 flattened layers for `nconv = 9`, or 27 for
`nconv = 7`.

One layer has to be swapped. The inference path uses `YOLO.BroadcastActivation`,
which writes its result back into its own input:

```julia
(l::BroadcastActivation)(x) = broadcast!(l.act, x, x)
```

That is exactly what you want for inference throughput and exactly what Zygote
cannot differentiate through. The package already handles this internally with
`YOLO._pure`; the example does the same thing with a small non-mutating
stand-in:

```julia
struct PureActivation{F}
    act::F
end
(l::PureActivation)(x) = l.act.(x)

pure(l::YOLO.BroadcastActivation) = PureActivation(l.act)
pure(l) = l
```

Adding a global-average-pool head gives the classifier Darknet pre-trains with:

```julia
function classifier(yolo; nclasses::Int, nconv::Int = 9)
    layers = map(pure, backbone_layers(yolo; nconv))
    channels = size(last(filter(l -> l isa Flux.Conv, layers)).weight, 4)
    backbone = Chain(layers...)
    head = Chain(GlobalMeanPool(), Flux.flatten, Dense(channels => nclasses))
    return Chain(; backbone, classifier = head)
end
```

At 224x224 the five stride-2 max-pools take the feature map to 7x7, and
`nconv = 9` leaves 512 channels there: 7.74M parameters including the head.

!!! note "The trunk is shared, the training is not"
    The `Conv` and `BatchNorm` objects in the classifier are the same objects the
    detector holds. But `Optimisers.update!` is functional: it rebuilds the model
    around fresh arrays rather than writing through them. So training the
    classifier does *not* update `yolo`, and the result has to be copied back
    explicitly.

## Training

Nothing exotic: `AdamW`, logit cross-entropy, one-hot targets.

```julia
model = todevice(classifier(yolo; nclasses = ncls, nconv = opts.nconv))
state = Optimisers.setup(Optimisers.AdamW(opts.lr), model)

for epoch in 1:opts.epochs
    Flux.trainmode!(model)
    for (X, Y) in batches(trainset, ncls; batchsize = opts.batchsize)
        x, y = todevice(X), todevice(Y)
        loss, grads = Flux.withgradient(m -> Flux.logitcrossentropy(m(x), y), model)
        state, model = Optimisers.update!(state, model, grads[1])
    end
end
```

```
julia --project -t auto train.jl --data /path/to/imagenette2-320 --epochs 5
```

Add `--full` to go through `ImageNet(split; dir)` instead of the subset loader,
and `--nconv 7` to stop at the shared 1024-channel trunk.

## Results

Five epochs on Imagenette from random initialisation, on an Apple M5 Pro via
Metal:

```
backend metal, 5 threads | 10 classes, 9469 train / 3925 val images
yolov3-tiny.cfg: first 9 conv blocks, 7742874 parameters at 224 x 224
  epoch 1 done in 105.3s | train loss 1.4157 | val loss 1.3569 | val acc 0.562
  epoch 2 done in  55.5s | train loss 1.0568 | val loss 2.6792 | val acc 0.424
  epoch 3 done in  54.7s | train loss 0.8928 | val loss 1.0007 | val acc 0.676
  epoch 4 done in  51.0s | train loss 0.7887 | val loss 0.9428 | val acc 0.701
  epoch 5 done in  52.0s | train loss 0.7337 | val loss 0.9186 | val acc 0.728
copied 9 conv blocks into the detector; wrote runs/yolov3-tiny-imagenet.weights
```

About 52 s per epoch after the first, which carries compilation: roughly 180
images/s including the input pipeline. Top-1 reaches 72.8% on the validation
split.

That is a working pipeline, not a pre-trained backbone. 9469 images over 10
classes is orders of magnitude less data than the trunk wants, and the validation
loss spiking at epoch 2 while training loss falls steadily is what that looks
like. A real run is the full 1.28M images for on the order of a hundred epochs;
the point of the subset is to exercise every part of the pipeline before
committing to that.

## Handing the trunk back to the detector

The trained parameters live in fresh arrays, so they are copied back into the
model and written out with the package's own [`save_weights`](@ref):

```julia
ncopied = copy_backbone!(yolo, Flux.cpu(model)[:backbone])
save_weights(yolo, joinpath(opts.out, "yolov3-tiny-imagenet.weights"))
```

`copy_backbone!` walks the model's flattened layers alongside the trained ones and
`copyto!`s the convolution kernels and batch-norm parameters in place, matching on
the layers that carry parameters rather than by position. Because the
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
inference ok, result cols: 3
train! losses: Float32[57.5662, 50.512, 44.8222]
```

The detection heads are still random, which is what you want: they are the layers
`train!` learns on the boxed dataset. Only the trunk carries anything over from
ImageNet, which is exactly the division of labour `weights_stop_layer = 15`
expresses in the [fine-tuning recipe](@ref training).

## Speed

See [Acceleration](@ref acceleration) for the backend comparison and the
Metal-specific pitfalls behind these numbers.
