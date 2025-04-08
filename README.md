# ObjectDetector.jl

Object detection via YOLO in Julia. YOLO models are loaded directly from Darknet .cfg and .weights files as Flux models.
Uses CUDA, if available.

## Installation

Requires julia v1.10+. From the Julia REPL, type `]` to enter the Pkg REPL mode and run:

```
pkg> add ObjectDetector
```

As of ObjectDetector v0.3, if you want to use CUDA accelleration you will also need to
add `CUDA` and `cuDNN` to your project and load both packages.

## Usage

![prettyprint example](examples/prettyprint.png)

### Loading and running on an image
```julia
using ObjectDetector, FileIO, ImageIO

yolomod = YOLO.v3_608_COCO(batch=1, silent=true) # Load the YOLOv3-tiny model pretrained on COCO, with a batch size of 1

batch = emptybatch(yolomod) # Create a batch object. Automatically uses the GPU if available

img = load(joinpath(dirname(dirname(pathof(ObjectDetector))),"test","images","dog-cycle-car.png"))

batch[:,:,:,1], padding = prepare_image(img, yolomod) # Send resized image to the batch

res = yolomod(batch, detect_thresh=0.5, overlap_thresh=0.8) # Run the model on the length-1 batch
```

Note that while the convention in Julia is column-major, where images are loaded
such that a _widescreen_ image matrix would have a smaller 1st dimension than 2nd.
Darknet is row-major, so the image matrix needs to have its first and second dims
permuted before being passed to batch. Otherwise features may not be detected due to
being rotated 90º. The function `prepare_image()` includes this conversion automatically.

Also, non-square models can be loaded, but care should be taken to ensure that each
dimension is an integer multiple of the filter size of the first conv layer (typically 16 or 32).


### CPU allocations management

On CPU an `AllocArrays` & `Adapt` - based allocator is used to reduce allocations.

To opt out of the allocator use `disable_bumper=true`.
i.e.
```julia
yolomod = YOLO.v3_608_COCO(batch=1, disable_bumper=true)
```

### Visualizing the result
```julia
imgBoxes = draw_boxes(img, yolomod, padding, res)
save("result.png", imgBoxes)
```
![dog-cycle-car with boxes](test/results/dog-cycle-car/v3_608_COCO.png)


## Pretrained Models
The darknet YOLO models from https://pjreddie.com/darknet/yolo/ that are pretrained on the COCO dataset are available:

```julia
YOLO.v2_COCO() #Currently broken (weights seem bad, model may work with custom weights)
YOLO.v2_tiny_COCO()

YOLO.v3_COCO()
YOLO.v3_spp_608_COCO()
YOLO.v3_tiny_COCO()

YOLO.v4_COCO()
YOLO.v4_tiny_COCO()

YOLO.v7_COCO()
YOLO.v7_tiny_COCO()
```
Their width and height can be modified with:
```julia
YOLO.v3_COCO(w=416,h=416)
```
and further configurations can be modified by editing the .cfg file structure after its read, but before its loaded:
```julia
yolomod = YOLO.v3_COCO(silent=false, cfgchanges=[(:net, 1, :width, 512), (:net, 1, :height, 384)])
```
`cfgchanges` takes the form of a vector of tuples with:
`(layer symbol, ith layer that matches given symbol, field symbol, value)`
Note that if `cfgchanges` is provided, optional `h` and `w` args are ignored.

Also, convenient sized models can be loaded via:
```julia
YOLO.v2_608_COCO()
YOLO.v2_tiny_416_COCO()

YOLO.v3_320_COCO()
YOLO.v3_416_COCO()
YOLO.v3_608_COCO()
YOLO.v3_spp_608_COCO()
YOLO.v3_tiny_416_COCO()
etc.
```

Or custom models can be loaded with:
```julia
YOLO.Yolo("path/to/model.cfg", "path/to/weights.weights", 1) # `1` is the batch size.
```

For instance the pretrained models are defined as:
```julia
function v3_COCO(;batch=1, silent=false, cfgchanges=nothing, w=416, h=416)
    cfgchanges=[(:net, 1, :width, w), (:net, 1, :height, h)]
    Yolo(joinpath(ObjectDetector.YOLO.models_dir(),"yolov3-416.cfg"), joinpath(artifact"yolov3-COCO", "yolov3-COCO.weights"), batch, silent=silent, cfgchanges=cfgchanges)
end
```

The weights are stored as lazily-loaded julia artifacts (introduced in Julia 1.3).

## Benchmarking

Pretrained models can be easily tested with `ObjectDetector.benchmark()`.

Note that the benchmark was run once before the examples here. Initial load time
of the first model loaded is typically between 3-20 seconds.

A desktop with a GTX 2060:
```
julia> ObjectDetector.benchmark()
┌──────────────────┬─────────┬───────────────┬──────┬──────────────┬────────────────┐
│            Model │ loaded? │ load time (s) │ ran? │ run time (s) │ run time (fps) │
├──────────────────┼─────────┼───────────────┼──────┼──────────────┼────────────────┤
│ v2_tiny_416_COCO │    true │          0.16 │ true │       0.0037 │          266.7 │
│ v3_tiny_416_COCO │    true │         0.243 │ true │       0.0042 │          236.4 │
│      v3_320_COCO │    true │         1.264 │ true │       0.0209 │           47.8 │
│      v3_416_COCO │    true │         1.456 │ true │        0.031 │           32.3 │
│      v3_608_COCO │    true │         2.423 │ true │       0.0686 │           14.6 │
└──────────────────┴─────────┴───────────────┴──────┴──────────────┴────────────────┘
```

A M2 Macbook Pro (CPU-only, no CUDA):
```
julia> ObjectDetector.benchmark()
┌──────────────────┬─────────┬───────────────┬──────────┬──────────────┬────────────────┬─────────────┐
│            Model │ loaded? │ load time (s) │ #results │ run time (s) │ run time (fps) │ allocations │
├──────────────────┼─────────┼───────────────┼──────────┼──────────────┼────────────────┼─────────────┤
│ v2_tiny_416_COCO │    true │         0.103 │      845 │       0.0372 │           26.9 │ 698.984 KiB │
│ v3_tiny_416_COCO │    true │         0.088 │     2535 │       0.0422 │           23.7 │   1.907 MiB │
│ v4_tiny_416_COCO │    true │         0.097 │     2535 │       0.0553 │           18.1 │   1.915 MiB │
│ v7_tiny_416_COCO │    true │         0.162 │    10647 │       0.1222 │            8.2 │   7.729 MiB │
│      v3_416_COCO │    true │         0.676 │    10647 │       0.3511 │            2.8 │   7.916 MiB │
│      v4_416_COCO │    true │         1.102 │    10647 │       0.6237 │            1.6 │   8.066 MiB │
│      v7_416_COCO │    true │         0.699 │    10647 │       0.5414 │            1.8 │   7.886 MiB │
└──────────────────┴─────────┴───────────────┴──────────┴──────────────┴────────────────┴─────────────┘
```

## Examples

All run with `detect_thresh = 0.5`, `overlap_thresh = 0.5`

### YOLO.v2_tiny_416_COCO
![v2_tiny_416_COCO](test/results/dog-cycle-car/v2_tiny_416_COCO.png)

### YOLO.v3_tiny_416_COCO
![v3_tiny_416_COCO](test/results/dog-cycle-car/v3_tiny_416_COCO.png)

### YOLO.v3_320_COCO
![v3_320_COCO](test/results/dog-cycle-car/v3_320_COCO.png)

### YOLO.v3_416_COCO
![v3_416_COCO](test/results/dog-cycle-car/v3_416_COCO.png)

### YOLO.v3_608_COCO
![v3_608_COCO](test/results/dog-cycle-car/v3_608_COCO.png)

###

