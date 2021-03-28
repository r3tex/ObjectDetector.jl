# Changelog

## v0.2

### Breaking changes
- Make empty output type stable and have correct empty shape https://github.com/r3tex/ObjectDetector.jl/pull/61
  Previously an empty result would erroneously return as either `CUDA.zeros(Float32, 1, 1)` if CUDA was enabled or
  `zeros(Float32, 1, 1)` if CPU-only. This was poor behavior as the latter stages of the yolo model always happens
  on CPU, so the output should always be `Matrix{Float32}`.
  Additionally, the `(1,1)` size of the output was illogical and hard to handle.
  Zero detections are now returned as `Matrix{Float32}` with a size that is stable in the first dimension.
  i.e. `89×0 Matrix{Float32}` for `YOLO.v3_608_COCO`

### Bugfixes
- Fix batches > 1 https://github.com/r3tex/ObjectDetector.jl/pull/60
  It turns out that batches > 1 were broken due to the overlapthreshold check not being batch page specific, meaning
  it was rejecting bboxes from other pages if they overlapped too much. Fixed and added tests to catch it
