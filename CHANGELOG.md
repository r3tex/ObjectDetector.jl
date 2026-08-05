# Changelog

## Unreleased

### Bugfixes
- Fix the reorg (passthrough) layer and the batchnorm read order for pre-0.2 darknet weight
  headers, and apply softmax (not sigmoid) to region-layer class scores when the cfg enables
  `softmax=1` (as the bundled v2 cfgs do; with `softmax=0` class scores are now left linear,
  matching darknet). Together these fix `v2_COCO` (previously failed to load) and
  `v2_tiny_COCO` (previously deviated from darknet); both now match AlexeyAB darknet output
  and are enabled in the test suite.
- Fix `overridecfg!`/`cfgchanges` targeting any layer other than `:net` (e.g.
  `(:yolo, 1, :classes, n)`), which previously errored or edited the wrong block.
- Fix soft-NMS (`nms_kind=soft`): decayed scores are now written back into the results,
  duplicate keeps are avoided, and boxes whose decayed score falls below `detect_thresh`
  are pruned from the returned detections.
- Fix the class-score maximum (CPU and CUDA) scanning one row past the last class into a
  zero-filled scratch attribute; with non-positive class scores (e.g. region `softmax=0`)
  the reported class index could point one past the last real class.
- Fix `prepare_image!` returning a bare 1-channel array (no padding tuple) for matching-size
  2D `Float32` inputs.
- CUDA: fix the last detection being dropped in `keepdetections`, an invalid class index
  when all class scores are non-positive, and a `>`/`>=` threshold boundary mismatch vs CPU.

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
