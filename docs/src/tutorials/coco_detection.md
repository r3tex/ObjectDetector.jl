# [Detection training on COCO](@id coco-detection)

This tutorial trains a detector on COCO with [`train!`](@ref training), reading the
official annotations rather than a darknet-converted copy of them.

The runnable code is in
[`examples/coco_detection/`](https://github.com/r3tex/ObjectDetector.jl/tree/master/examples/coco_detection).

```@meta
CurrentModule = ObjectDetector
```

## The data

COCO's val2017 split is 5000 images and about 1 GB, which is enough to exercise
the whole pipeline. train2017 is 118k images and about 18 GB, and is what a real
run uses. Both come from the same place and unpack into the same layout:

```
curl -LO http://images.cocodataset.org/zips/val2017.zip
curl -LO http://images.cocodataset.org/annotations/annotations_trainval2017.zip
unzip -q val2017.zip && unzip -q annotations_trainval2017.zip
```

```
coco/
├── val2017/000000000139.jpg ...
└── annotations/instances_val2017.json
```

## Converting the annotations

Two things have to be converted, and both are easy to get wrong.

**Box coordinates.** COCO stores pixel `[x, y, w, h]` from the top-left corner.
[`TrainSample`](@ref training) wants normalized `[class, cx, cy, w, h]`, so the
corner becomes a center and everything is divided by the image's own width and
height:

```julia
x, y, w, h = a["bbox"]
_, iw, ih = info[a["image_id"]]
(label[a["category_id"]], (x + w / 2) / iw, (y + h / 2) / ih, w / iw, h / ih)
```

**Class ids.** COCO's category ids run 1 to 90 with gaps, for 80 classes. The
model wants a contiguous 1-based index, and *which* index matters if the result is
to stay compatible with `coco.names` and the pretrained weights.

Map by sorted category id, not by name:

```julia
cats = JSON3.read(read(annotationfile), Dict{String,Any})["categories"]
order = sortperm([c["id"] for c in cats])
```

Sorted id order is darknet's class order. Name order is not, because darknet
renames several classes: its `coco.names` says `motorbike` where COCO says
`motorcycle`, `aeroplane` for `airplane`, `sofa` for `couch`, `tvmonitor` for
`tv`. Matching on names silently mislabels those classes; matching on sorted ids
cannot.

Crowd regions (`iscrowd == 1`) are dropped. They mark an unspecified number of
objects with one box, so they are neither a single object nor background, and
training on them teaches the model a box that does not correspond to a thing.

The result is a `TrainSample` per image, holding the path rather than the pixels
so that `train!` can load lazily:

```julia
data = coco_samples("coco/val2017", "coco/annotations/instances_val2017.json")
train!(model, data; image_loader = FileIO.load, ...)
```

## Choosing what to start from

`--from` picks the starting weights, and the three choices are genuinely
different jobs.

`pretrained` fine-tunes the COCO-pretrained detector on all 80 classes. Batch-norm
stays folded into the convolutions, which is what the package does by default and
what suits adapting weights that are already close.

`backbone` starts from an ImageNet-pre-trained trunk, which is what the
[backbone tutorial](@ref imagenet-backbone) produces. The trunk loads, the heads
are random, and `trainable_batchnorm = true` keeps batch-norm live because the
model has real learning left to do.

`scratch` random-initializes everything, and wants far more data and epochs than
this example is set up for.

## Changing the class count

Restricting to a subset of classes is the common real use of this, and it means
rebuilding the ends of the network. Each `[yolo]` layer's `classes` changes, and
so does the filter count of the convolution feeding it, which is
`anchors_per_head * (5 + classes)`:

```julia
for (i, (blockidx, nanchors)) in enumerate(head_conv_blocks(cfgfile))
    push!(changes, (:yolo, i, :classes, nclasses))
    push!(changes, (:convolutional, blockidx, :filters, nanchors * (5 + nclasses)))
end
```

Reading the anchor count out of the cfg rather than assuming 3 keeps this working
across model families. The pretrained weights are then loaded only up to the
backbone split, since the reshaped heads cannot take theirs:

```julia
YOLO.Yolo(cfgfile, weightfile, 1; cfgchanges = changes,
          weights_stop_layer = 15, trainable_batchnorm = true)
```

!!! note "cfgchanges is dispatched on element type"
    `cfgchanges` must be a concretely typed vector, such as
    `Tuple{Symbol,Int,Symbol,Int}[]`. A `Vector{Any}` holding the same tuples is a
    `MethodError`.

## Running it

```
julia --project -t auto train.jl --data coco --epochs 3 --lr 1e-4
```

All 80 classes, the whole of val2017, on an Apple M5 Pro via Metal:

```
v3_tiny_COCO val2017, 80 classes | 4952 images, 36335 boxes | from pretrained
backend metal, 5 threads, 416x416, batch 8
epoch 1/3  loss: 4.647834
epoch 2/3  loss: 3.6414304
epoch 3/3  loss: 3.109242
3 epochs in 6.5 min | loss 4.648 -> 3.109
```

About two minutes per epoch over 4952 images, so roughly 40 images/s including
JPEG decode.

Cutting the model down to two classes, where the heads are rebuilt and start
random, moves further because it has more to learn:

```
julia --project -t auto train.jl --data coco --classes person,car \
    --limit 400 --epochs 4 --lr 1e-3 --warmup 50
```

```
v3_tiny_COCO val2017, 2 classes | 400 images, 1762 boxes | from pretrained
4 epochs in 1.9 min | loss 5.731 -> 2.146
```

What neither of these is, is a trained detector. Fine-tuning on val2017 for three
epochs is a pipeline check: the split is 5000 images against train2017's 118k, and
it is the set you would evaluate on, so the numbers say nothing about
generalization. A real run points `--split` at train2017 and takes days rather
than minutes.

`train!` writes a darknet-format checkpoint each epoch when given
`checkpoint_dir`, and the final `save_weights` reloads with the same cfg and the
same `cfgchanges`.
