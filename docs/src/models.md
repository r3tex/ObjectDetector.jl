# [Pretrained models](@id pretrained-models)

The Darknet YOLO models pretrained on COCO are available as lazily-downloaded
Julia artifacts, so the weights are fetched on first use rather than at install
time.

```julia
YOLO.v2_COCO()
YOLO.v2_tiny_COCO()

YOLO.v3_COCO()
YOLO.v3_spp_608_COCO()
YOLO.v3_tiny_COCO()

YOLO.v4_COCO()
YOLO.v4_tiny_COCO()

YOLO.v7_COCO()
YOLO.v7_tiny_COCO()

# Scaled-YOLOv4 family
YOLO.v4_csp_COCO()
YOLO.v4_csp_x_swish_COCO()
YOLO.v4x_mish_COCO()
YOLO.v4_p5_COCO()
YOLO.v4_p6_COCO()

# larger yolov7
YOLO.v7x_COCO()
```

Each defaults to its native input size. Native sizes are 512 for `v4_csp`, 640 for
`v4_csp_x_swish`, `v4x_mish` and `v7x`, 896 for `v4_p5`, 1280 for `v4_p6`, and 416
for everything else.

## Changing the input size

```julia
YOLO.v3_COCO(w = 416, h = 416)
```

Convenience constructors exist for common sizes:

```julia
YOLO.v2_608_COCO()
YOLO.v2_tiny_416_COCO()
YOLO.v3_320_COCO()
YOLO.v3_416_COCO()
YOLO.v3_608_COCO()
YOLO.v3_spp_608_COCO()
YOLO.v3_tiny_416_COCO()
```

Anything else in the config can be changed after the cfg is read but before the
model is built:

```julia
yolomod = YOLO.v3_COCO(silent = false,
    cfgchanges = [(:net, 1, :width, 512), (:net, 1, :height, 384)])
```

`cfgchanges` is a vector of `(layer symbol, ith layer matching that symbol, field
symbol, value)` tuples. If `cfgchanges` is given, `w` and `h` are ignored.

## Custom models

```julia
YOLO.Yolo("path/to/model.cfg", "path/to/model.weights", 1)
```

where the trailing `1` is the batch size. Useful keywords:

| Keyword | |
|:--|:--|
| `silent` | suppress the layer-by-layer construction print |
| `use_gpu` | opt out of the GPU even when one is available |
| `cfgchanges` | edit the cfg before building, as above |
| `weights_stop_layer` | load weights only up to this cfg block; later layers are randomly initialised |
| `allow_partial_weights` | accept a truncated weights file such as `yolov3-tiny.conv.15` |
| `trainable_batchnorm` | keep live batch-norm layers instead of folding them into the convolutions |
| `disallow_bumper` | opt out of the `AllocArrays` CPU allocator |

`YOLO.YOLO_MODELS` maps a model name to a function returning its `(cfg, weights)`
paths, which is the way to get at a pretrained cfg/weights pair without building
the model:

```julia
cfg, weights = YOLO.YOLO_MODELS["v3_tiny_COCO"]()
```
