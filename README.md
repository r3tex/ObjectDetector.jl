# ObjectDetector.jl

Object detection via YOLO in Julia.

YOLO models are loaded directly from Darknet .cfg and .weights files as Flux models.


| **Platform**                                                               | **Build Status**                                                                                |
|:-------------------------------------------------------------------------------:|:-----------------------------------------------------------------------------------------------:|
| x86 & AARCH Linux, MacOS | [![][travis-img]][travis-url] |


## Installation

Requires julia v1.3+

The package can be installed with the Julia package manager.
From the Julia REPL, type `]` to enter the Pkg REPL mode and run:

```
pkg> add ObjectDetector
```


## Example Usage (WIP)

### Loading and running on an image
```julia
using ObjectDetector

mod = YOLO.v3_tiny_416_COCO()

batch = emptybatch(mod) # Create a batch object. Automatically uses the GPU if available

img = load(joinpath(dirname(dirname(pathof(YOLO))),"test","images","dog-cycle-car.png"))

batch[:,:,:,1] .= gpu(resizePadImage(img, mod)) # Send resized image to the batch

res = mod(batch) # Run the model on the length-1 batch
```

## Pretrained Models
Most of the darknet models that are pretrained on the COCO dataset are available:
```julia
YOLO.v2_tiny_416_COCO()
YOLO.v3_320_COCO()
YOLO.v3_416_COCO()
YOLO.v3_608_COCO()
YOLO.v3_tiny_416_COCO()
```
The following are available but do not load due to bugs (work in progress)
```julia
YOLO.v2_608_COCO()
YOLO.v3_608_spp_COCO()
```

Or custom models can be loaded with:
```julia
YOLO.yolo("path/to/model.cfg", "path/to/weights.weights", 1)
```
where `1` is the batch size.

For instance the pretrained models are defined as:
```julia
v2_608_COCO(;batch=1, silent=false) = yolo(joinpath(models_dir,"yolov2-608.cfg"), getArtifact("yolov2-COCO"), batch, silent=silent)
```

The weights are stored as lazily-loaded julia artifacts.

## Benchmarking

Pretrained models can be easily tested with `ObjectDetector.benchmark()`.

Note that the benchmark was run once before the examples here. Initial load time
of the first model loaded will be ~20 seconds

A desktop with a GTX 2060:
```
julia> ObjectDetector.benchmark()

┌──────────────────┬─────────┬───────────────┬──────┬──────────────┬────────────────┬──────────────────┐
│            Model │ loaded? │ load time (s) │ ran? │ run time (s) │ run time (fps) │ objects detected │
├──────────────────┼─────────┼───────────────┼──────┼──────────────┼────────────────┼──────────────────┤
│ v2_tiny_416_COCO │    true │         0.166 │ true │       0.0037 │          272.6 │                1 │
│ v3_tiny_416_COCO │    true │         0.262 │ true │       0.0041 │          241.7 │                1 │
│      v3_320_COCO │    true │         1.298 │ true │         0.02 │           50.0 │                2 │
│      v3_416_COCO │    true │         1.605 │ true │       0.0296 │           33.8 │                3 │
│      v3_608_COCO │    true │         2.219 │ true │       0.0618 │           16.2 │                2 │
└──────────────────┴─────────┴───────────────┴──────┴──────────────┴────────────────┴──────────────────┘
```

A 2019 Macbook Pro (CPU-only, no CUDA)
```
┌──────────────────┬─────────┬───────────────┬──────┬──────────────┬────────────────┬──────────────────┐
│            Model │ loaded? │ load time (s) │ ran? │ run time (s) │ run time (fps) │ objects detected │
├──────────────────┼─────────┼───────────────┼──────┼──────────────┼────────────────┼──────────────────┤
│ v2_tiny_416_COCO │    true │         0.307 │ true │       0.1537 │            6.5 │                1 │
│ v3_tiny_416_COCO │    true │         0.318 │ true │       0.2088 │            4.8 │                1 │
│      v3_320_COCO │    true │         2.064 │ true │       1.5278 │            0.7 │                1 │
│      v3_416_COCO │    true │         4.221 │ true │       2.7166 │            0.4 │                1 │
│      v3_608_COCO │    true │         8.334 │ true │       4.2348 │            0.2 │                1 │
└──────────────────┴─────────┴───────────────┴──────┴──────────────┴────────────────┴──────────────────┘
```


[discourse-tag-url]: https://discourse.julialang.org/tags/yolo

[travis-img]: https://travis-ci.com/ianshmean/ObjectDetector.jl.svg?branch=master
[travis-url]: https://travis-ci.com/ianshmean/ObjectDetector.jl

[codecov-img]: https://codecov.io/gh/ianshmean/ObjectDetector.jl/branch/master/graph/badge.svg
[codecov-url]: https://codecov.io/gh/ianshmean/ObjectDetector.jl

[coveralls-img]: https://coveralls.io/repos/github/ianshmean/ObjectDetector.jl/badge.svg?branch=master
[coveralls-url]: https://coveralls.io/github/ianshmean/ObjectDetector.jl?branch=master

[issues-url]: https://github.com/ianshmean/ObjectDetector.jl/issues
