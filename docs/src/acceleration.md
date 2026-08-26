# [Acceleration](@id acceleration)

Where the time goes, and what actually moves the needle. The inference notes come
from the package's own tuning; the training numbers were measured while
[pre-training a backbone](@ref imagenet-backbone) on an Apple M5 Pro, and
`examples/imagenet_backbone/bench.jl` reproduces them.

## CPU

The forward pass on CPU is dominated by BLAS matrix multiplies, because NNlib's
convolutions are im2col plus a GEMM. The BLAS backend therefore matters more than
anything else:

- **Apple silicon**: load
  [AppleAccelerate.jl](https://github.com/JuliaLinearAlgebra/AppleAccelerate.jl)
  before running. Apple's AMX-backed sgemm beats the default OpenBLAS
  substantially (~35% end-to-end for `v3_416_COCO` on an M2 Pro):

  ```julia
  using AppleAccelerate, ObjectDetector
  ```
- **Intel**: [MKL.jl](https://github.com/JuliaLinearAlgebra/MKL.jl) plays the same
  role.
- BLAS threading, not Julia's `-t`, controls convolution parallelism.
  `LinearAlgebra.BLAS.set_num_threads` is the knob; the default is usually right.
- For throughput, batch images (`YOLO.v3_416_COCO(batch = N)`) rather than making
  repeated single-image calls.

On CPU an [AllocArrays](https://github.com/ericphanson/AllocArrays.jl)-based
allocator reduces allocations. Opt out with `disallow_bumper = true`.

## CUDA

Add and load `CUDA` and `cuDNN`; `ext/CUDAExt.jl` then supplies GPU kernels for
the detection post-processing (`clipdetect!`, `findmax!`, `keepdetections`) as
well as `maxpool` and `upsample`, and marks `CuArray` as slow for scalar
indexing so the NMS path takes its batched route.

## Apple silicon, for training

Metal.jl is not one of the package's extensions, but it does not need to be for
training: the training path is plain Flux, so `Flux.gpu` moves a model to Metal
and the gradient runs there. That is what the pre-training example does.

### Measurements

One training step, forward plus backward plus an `AdamW` update, of the
YOLOv3-tiny trunk classifier (9 conv blocks, 7.74M parameters) at 224x224. Apple
M5 Pro (16 GPU cores, 5 performance and 10 efficiency cores, 24 GB), Julia 1.12.6
with `-t auto`, Flux 0.16.11, Metal.jl 1.10.3. One process per row.

| Backend | Batch | s/step | img/s |
|:--|--:|--:|--:|
| Metal | 32 | 0.630 | 50.8 |
| Metal | 64 | 0.841 | 76.1 |
| **Metal** | **128** | **1.155** | **110.8** |
| CPU + AppleAccelerate | 64 | 1.774 | 36.1 |
| CPU, stock OpenBLAS | 64 | 2.316 | 27.6 |

Metal is about **2x** the best CPU configuration at the same batch size and **3x**
at batch 128; Accelerate is worth about **1.3x** over stock OpenBLAS on the CPU.
Bigger batches keep helping on Metal well past the point where they stop helping
on the CPU.

Treat single runs as indicative rather than precise: repeated measurements of the
same configuration varied by 5-10%, and by considerably more when warm-up was too
short.

### Warm up before timing anything on Metal

This is the trap, and it is easy to draw exactly the wrong conclusion from it.
Metal.jl grows its buffer pool over the first several steps. Timed with two
warm-up steps in a process shared with earlier configurations, a batch-32 run
measured **14.5 img/s** and looked far slower than the CPU. The same configuration
in a fresh process with four warm-up steps measured **88.4 img/s**.

Warm up generously, and measure one configuration per process.

### Broadcast throughput falls off with array rank

Copying a 96 MiB array on Metal runs at ~205 GB/s. Broadcasting over the same
memory does not come close, and how far off depends on the rank of the array:

| Operation | Effective bandwidth |
|:--|--:|
| `copyto!(y, x)` | 205 GB/s |
| `y .= x .+ 1`, 1-D | 140 GB/s |
| ... 2-D | 95 GB/s |
| ... 4-D (W, H, C, N) | 31 GB/s |

Element-wise work on activations is therefore much less favourable on Metal than
the convolutions are, which are 4-8x the CPU. Where an operation does not care
about shape, reshaping a WHCN array to a vector before broadcasting is worth
several times the bandwidth.

What does *not* follow from this is that fusing activation layers into the
batch-norm helps. That was measured too: folding the `[convolutional]` block's
activation into the preceding `BatchNorm`, which cuts the trunk from 33 layers to
24 with bit-identical outputs, changed throughput by less than the run-to-run
noise on both backends (Metal 71.4 fused against 76.1 unfused, Accelerate 36.8
against 36.1). Flux's batch-norm already writes its output once either way.

### Check for scalar-indexing fallbacks

One unsupported operation falling back to scalar indexing will dominate
everything else. Confirm there are none:

```julia
Metal.allowscalar(false)
```

The trunk, its gradient and the optimiser update all pass with scalar indexing
disabled.

!!! warning "Metal covers training, not inference"
    The training path is generic Flux and runs on Metal. The *inference* path is
    not covered. `ext/CUDAExt.jl` marks `CuArray` as slow for scalar indexing and
    supplies kernels for the detection post-processing; there is no equivalent for
    `MtlArray`, so `fast_scalar_indexing` returns `true` for it and the NMS path
    takes its scalar route, which errors on a GPU array:

    ```
    model uses_gpu: true
    fast_scalar_indexing(MtlArray): true
    ERROR: Scalar indexing is disallowed.
    ```

    Build inference models with `use_gpu = false` on Apple silicon. Closing the
    gap means a Metal extension mirroring `CUDAExt`.

### Give the input pipeline threads

JPEG decoding is the input bottleneck and is embarrassingly parallel, so run Julia
with `-t auto` and decode a batch's observations concurrently. Passing `open_size`
to ImageNetDataset's transforms also lets JpegTurbo scale the image down during
decode rather than after, which is a large saving.
