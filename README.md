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


## Example Usage

```julia
julia> using ObjectDetector
julia> yolomod = YOLO.v3_416_COCO()
(1) conv(3,3->32) => (2) conv(3,32->64) => (3) conv(1,64->32) => (4) conv(3,32->64) =>
(5) shortcut(2,4) => (6) conv(3,64->128) => (7) conv(1,128->64) => (8) conv(3,64->128) =>
(9) shortcut(6,8) => (10) conv(1,128->64) => (11) conv(3,64->128) =>
(12) shortcut(9,11) => (13) conv(3,128->256) => (14) conv(1,256->128) => (15) conv(3,128->256) =>
(16) shortcut(13,15) => (17) conv(1,256->128) => (18) conv(3,128->256) =>
(19) shortcut(16,18) => (20) conv(1,256->128) => (21) conv(3,128->256) =>
(22) shortcut(19,21) => (23) conv(1,256->128) => (24) conv(3,128->256) =>
(25) shortcut(22,24) => (26) conv(1,256->128) => (27) conv(3,128->256) =>
(28) shortcut(25,27) => (29) conv(1,256->128) => (30) conv(3,128->256) =>
(31) shortcut(28,30) => (32) conv(1,256->128) => (33) conv(3,128->256) =>
(34) shortcut(31,33) => (35) conv(1,256->128) => (36) conv(3,128->256) =>
(37) shortcut(34,36) => (38) conv(3,256->512) => (39) conv(1,512->256) => (40) conv(3,256->512) =>
(41) shortcut(38,40) => (42) conv(1,512->256) => (43) conv(3,256->512) =>
(44) shortcut(41,43) => (45) conv(1,512->256) => (46) conv(3,256->512) =>
(47) shortcut(44,46) => (48) conv(1,512->256) => (49) conv(3,256->512) =>
(50) shortcut(47,49) => (51) conv(1,512->256) => (52) conv(3,256->512) =>
(53) shortcut(50,52) => (54) conv(1,512->256) => (55) conv(3,256->512) =>
(56) shortcut(53,55) => (57) conv(1,512->256) => (58) conv(3,256->512) =>
(59) shortcut(56,58) => (60) conv(1,512->256) => (61) conv(3,256->512) =>
(62) shortcut(59,61) => (63) conv(3,512->1024) => (64) conv(1,1024->512) => (65) conv(3,512->1024) =>
(66) shortcut(63,65) => (67) conv(1,1024->512) => (68) conv(3,512->1024) =>
(69) shortcut(66,68) => (70) conv(1,1024->512) => (71) conv(3,512->1024) =>
(72) shortcut(69,71) => (73) conv(1,1024->512) => (74) conv(3,512->1024) =>
(75) shortcut(72,74) => (76) conv(1,1024->512) => (77) conv(3,512->1024) => (78) conv(1,1024->512) => (79) conv(3,512->1024) => (80) conv(1,1024->512) => (81) conv(3,512->1024) => (82) conv(1,1024->255) => (83) YOLO ||
(84) route(80,) => (85) conv(1,512->256) => (86) upsample(2) =>
(87) route(86,62) => (88) conv(1,768->256) => (89) conv(3,256->512) => (90) conv(1,512->256) => (91) conv(3,256->512) => (92) conv(1,512->256) => (93) conv(3,256->512) => (94) conv(1,512->255) => (95) YOLO ||
(96) route(92,) => (97) conv(1,256->128) => (98) upsample(2) =>
(99) route(98,37) => (100) conv(1,384->128) => (101) conv(3,128->256) => (102) conv(1,256->128) => (103) conv(3,128->256) => (104) conv(1,256->128) => (105) conv(3,128->256) => (106) conv(1,256->255) => (107) YOLO ||

Generating chains and outputs:
1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30

DarkNet 0.2.0
WxH: 416x416   channels: 3   batchsize: 1
gridsize: 13   classes: 80   thresholds: Detect 0.6. Overlap 0.4
```

### Loading and running on an image
```julia
using ObjectDetector, FileIO

yolomod = YOLO.v3_tiny_416_COCO(batch=1, silent=true) # Load the YOLOv3-tiny model pretrained on COCO, with a batch size of 1

batch = emptybatch(yolomod) # Create a batch object. Automatically uses the GPU if available

img = load(joinpath(dirname(dirname(pathof(ObjectDetector))),"test","images","dog-cycle-car.png"))

batch[:,:,:,1] .= gpu(resizePadImage(img, yolomod)) # Send resized image to the batch

res = yolomod(batch) # Run the model on the length-1 batch
```

### Visualzing the result
```julia
imgBoxes = drawBoxes(img, res)
save(joinpath(@__DIR__,"img.png"), imgBoxes)
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

A 2019 Macbook Pro (CPU-only, no CUDA)
```
┌──────────────────┬─────────┬───────────────┬──────┬──────────────┬────────────────┐
│            Model │ loaded? │ load time (s) │ ran? │ run time (s) │ run time (fps) │
├──────────────────┼─────────┼───────────────┼──────┼──────────────┼────────────────┤
│ v2_tiny_416_COCO │    true │         0.305 │ true │       0.1383 │            7.2 │
│ v3_tiny_416_COCO │    true │         0.267 │ true │       0.1711 │            5.8 │
│      v3_320_COCO │    true │         1.617 │ true │       0.8335 │            1.2 │
│      v3_416_COCO │    true │         2.377 │ true │       1.4138 │            0.7 │
│      v3_608_COCO │    true │         4.239 │ true │       3.1122 │            0.3 │
└──────────────────┴─────────┴───────────────┴──────┴──────────────┴────────────────┘
```


[discourse-tag-url]: https://discourse.julialang.org/tags/yolo

[travis-img]: https://travis-ci.com/ianshmean/ObjectDetector.jl.svg?branch=master
[travis-url]: https://travis-ci.com/ianshmean/ObjectDetector.jl

[codecov-img]: https://codecov.io/gh/ianshmean/ObjectDetector.jl/branch/master/graph/badge.svg
[codecov-url]: https://codecov.io/gh/ianshmean/ObjectDetector.jl

[coveralls-img]: https://coveralls.io/repos/github/ianshmean/ObjectDetector.jl/badge.svg?branch=master
[coveralls-url]: https://coveralls.io/github/ianshmean/ObjectDetector.jl?branch=master

[issues-url]: https://github.com/ianshmean/ObjectDetector.jl/issues
