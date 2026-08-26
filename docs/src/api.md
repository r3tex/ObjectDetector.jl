# API reference

```@meta
CurrentModule = ObjectDetector
```

## Preparing images

A model expects a letterboxed batch at its own input resolution, with Julia's
column-major image permuted to Darknet's row-major convention. `prepare_image`
does both and returns the padding needed to map detections back to the original
image.

```@docs
prepare_image
resizekern
sizethatfits
```

`prepare_image!` writes into an existing destination array instead of allocating
one, which is what `emptybatch` plus a batch slice is for.

## Batches

```@docs
emptybatch
```

## Drawing results

```@docs
draw_boxes
```

## Models

Models are built from a Darknet `.cfg` and a matching `.weights` file:

```julia
YOLO.Yolo(cfgfile::String, weightfile::Union{Nothing,String}, batchsize::Int = 1; kwargs...)
```

See [Pretrained models](@ref pretrained-models) for the keywords and for the
constructors that ship with the package.

```@docs
get_input_size
```

## Training

`train!`, `save_weights`, `TrainSample`, `load_darknet_dataset` and
`load_darknet_labels` are documented on the [Training](@ref training) page.

## Benchmarking

```@docs
benchmark
```
