# ObjectDetector.jl

Object detection via YOLO in Julia. YOLO models are loaded directly from Darknet
`.cfg` and `.weights` files as Flux models, so a Darknet architecture runs without
a conversion step. Uses CUDA if available.

Supported families: `v2`, `v2-tiny`, `v3`, `v3-spp`, `v3-tiny`, `v4`, `v4-tiny`,
`v4-csp`, `v4-csp-x-swish`, `v4x-mish`, `v4-p5`, `v4-p6` (Scaled-YOLOv4), `v7`,
`v7-tiny`, `v7x`. Other less standard models may work too.

All supported models have result parity with
[AlexeyAB/darknet](https://github.com/AlexeyAB/darknet) and are tested directly
against [Darknet.jl](https://github.com/IanButterworth/Darknet.jl).

## Installation

Requires Julia 1.10+.

```julia
pkg> add ObjectDetector
```

For CUDA acceleration, also add and load `CUDA` and `cuDNN`.

## Running a model on an image

```julia
using ObjectDetector, FileIO, ImageIO

yolomod = YOLO.v3_608_COCO(batch = 1, silent = true)

batch = emptybatch(yolomod)              # uses the GPU if one is available
img = load(joinpath(dirname(dirname(pathof(ObjectDetector))),
                    "test", "images", "dog-cycle-car.png"))

batch[:, :, :, 1], padding = prepare_image(img, yolomod)
res = yolomod(batch, detect_thresh = 0.5, overlap_thresh = 0.8)
```

Each column of `res` is one detection:

```julia
i = 1
bbox                     = res[1:4, i]
objectness_score         = res[5, i]
selected_class_confidence = res[end-2, i]
selected_class_id        = res[end-1, i]
batch_id                 = res[end, i]
```

Julia is column-major and Darknet is row-major, so the image matrix needs its
first two dimensions permuted before going into the batch, or features come out
rotated 90°. `prepare_image` does that conversion, along with the aspect-preserving
letterbox, and returns the padding needed to map boxes back.

Non-square models load fine, but each dimension must be an integer multiple of the
network's maximum stride: 32 for most models, 64 for `v4_p6`.

## Drawing the result

```julia
imgBoxes = draw_boxes(img, yolomod, padding, res)
save("result.png", imgBoxes)
```

## Where to go next

- [Pretrained models](@ref pretrained-models): what ships as an artifact, sizes,
  and loading your own `.cfg`/`.weights`.
- [Training](@ref training): fine-tuning and from-scratch detection training.
- [Pre-training a backbone on ImageNet](@ref imagenet-backbone): the classification
  step of the Darknet recipe, using
  [ImageNetDataset.jl](https://github.com/Julia-XAI/ImageNetDataset.jl).
- [Acceleration](@ref acceleration): CPU BLAS, CUDA, and measured notes on Apple
  silicon.
