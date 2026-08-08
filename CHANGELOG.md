# Changelog

## Unreleased

### Training support
- Add `train!` for training `[yolo]`-output models — classic decode (v3 family, v4,
  v4-tiny) and `new_coords=1` scaled decode (scaled-YOLOv4 family, v7 family) —
  with a CIoU + BCE loss, best-anchor target assignment, and a differentiable
  non-mutating forward path alongside the existing fast buffered inference path.
- Add `trainable_batchnorm=true` construction mode keeping live batchnorm layers for
  from-scratch training (default remains folded batchnorm for fastest inference);
  plus `warmup_batches` (darknet-style burn-in) and `flip_augment` options.
- Add `TrainSample`, `load_darknet_dataset`/`load_darknet_labels` (darknet label
  format), and letterbox-aware box transforms for batch assembly.
- Add `save_weights` to write darknet-format `.weights` files that round-trip with
  the same cfg (batchnorm-folded convs are written with identity batchnorm).
- Add transfer-learning weight loading: `weights_stop_layer` (random-init layers past
  a backbone split, enabling custom class counts via `cfgchanges`) and
  `allow_partial_weights` (truncated `.conv.XX` backbone files).

## v1.2.0 - 2026-08-07

Deep review series: #131 (bugfixes), #132 (performance), #137 (new models), #138 (quality).

### New models
- Add the Scaled-YOLOv4 family and yolov7x, all with official COCO weights:
  `v4_csp_COCO`, `v4_csp_x_swish_COCO`, `v4x_mish_COCO`, `v4_p5_COCO`, `v4_p6_COCO`,
  `v7x_COCO`. Pretrained models now default to their native input size
  (416 for the previously-supported models, unchanged).

### Performance
- Batchnorm is now folded into the conv weights at load time (as darknet's
  `fuse_conv_batchnorm` does), removing the BatchNorm pass entirely: ~4-9% faster forward
  pass on CPU depending on the model. Outputs shift only by float rounding (< 1e-6).
- Post-processing is significantly cheaper when many detections are kept. For `v3_416_COCO`
  with all 10647 candidates retained (CPU, `detect_thresh=0`), post-processing allocations
  drop from 78 MiB to 23 MiB and NMS time by ~20%: detections are grouped for NMS with one
  global sort instead of per-batch/per-class scans, the NMS inner loop no longer allocates
  per suppression round, output blocks are flattened with a single allocation, and the
  per-head transforms run as fused in-place broadcasts.
- NMS output columns are now ordered by batch, then ascending class id, then descending
  score (classes were previously in first-appearance order).

### Quality
- `benchmark()` moved behind a package extension: load `BenchmarkTools` and `PrettyTables`
  to use it. It is now keyed by model name, covers all models, and its table printing works
  under PrettyTables v3.
- Added a precompile workload: time-to-first-inference drops from ~13s to under 1s.
- Input size validation now checks against the network's actual maximum stride (computed
  from the cfg) instead of the first conv's filter count, which wrongly rejected valid
  sizes (e.g. yolov7x at 416).
- Flux compat narrowed to tested versions (0.14.1+).
- Missing test references now fail the test suite instead of being silently self-blessed;
  the reference generator is committed at `dev/generate_test_references.jl`.
- Assorted dead code removed.

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
