# Review backlog

Open items from the 2026-08 review series (deep bug review → performance →
model coverage → code quality). Everything actionable that was found and *not*
already fixed in #131/#132/#137/#138 is recorded here so it isn't lost.
Effort: S (< half day), M (days), L (larger / needs design).

## Correctness & robustness

- **[M] Thread-safety / reentrancy.** Inference mutates model state
  (`yolo.W[0] = img`; `_add`/`_broadcast` write into stored buffers), so two
  concurrent calls on one model corrupt each other. Related latent fragility:
  a chain *beginning* with a shortcut or bare activation would mutate a stored
  `W` buffer that later routes read — shipped cfgs avoid this only by
  construction order. Fix directions: per-call buffer set, or a lock plus a
  documented `Threads`-safety statement. At minimum the README should state
  the model is not reentrant.
- **[S] cfg parser gaps.** `cfgparse` fails on scientific notation (`1e-3`)
  and mixed alphanumeric values; `cfgsplit` breaks on `=` inside values.
  Comments/blank lines were fixed in #137; these remain for exotic custom
  cfgs.
- **[S] `yolov3-tiny-prn` fails to load** (`KeyError: 31` in the route/skip
  bookkeeping). Niche edge model, but the failure mode may indicate a real
  route-indexing hole worth understanding.

## GPU

- **[M] No GPU CI.** The CUDA extension is review-verified only; several bugs
  fixed in #131 (dropped last detection, class-index underflow) had lived
  there unnoticed because no runner exercises it. Buildkite GPU CI (as used
  by JuliaGPU) would close this permanently.
- **[S] Kernel launch hygiene.** `kern_clipdetect` launches `cols × 1024`
  threads for `cols` work items; `kern_findmax!`/`kern_keepdetections` use
  `threads=rows`, capping at 1024 rows (≈1016 classes) and leaving most
  threads idle. The file's own comment agrees ("CAN BE OPTIMIZED").
- **[S] GPU paths for the new fast post-processing.** `keepdetections(::Vector)`
  and `flatten_with_attributes` fall back to the old cat/permutedims path on
  CuArray via `fast_scalar_indexing`; native equivalents would carry the
  #132 allocation wins to GPU.
- **[L] Metal.jl backend** for Apple silicon — the largest available speedup
  for Mac users (forward pass is gemm-bound; see README BLAS notes). Needs
  either Metal versions of the post-processing kernels or a
  forward-on-GPU / post-process-on-CPU split.

## Performance (declined or residual)

- **[S] `conv_bias_act` fusion.** With batchnorm folded (#132), NNlib's fused
  conv+bias+activation could remove the remaining separate activation pass.
  Incremental (~few %) on top of the fold.
- Considered and **rejected**: faster sigmoid/exp approximations (changes
  numerics vs darknet parity); in-place `conv!` into the `W` buffers (the
  pass is not allocation-bound — bumper already recycles ~890 MiB to
  ~260 KiB); avoiding the O(n²) IoU loop at `overlap_thresh=1.0` (inherent
  to hard NMS when nothing suppresses).

## Models

- **[M] `cspx-p7-mish` (1536², stride 128).** Needs a `[sam]` layer
  (elementwise multiply with a referenced layer — small, analogous to
  `:add`). The old size-validation blocker was fixed in #138. Verify
  official weights availability before starting; test at a reduced size
  (CI memory).
- **[?] Non-COCO pretrained variants** (yolov3-openimages, VOC) — official
  weights exist; add on demand.
- **Out of scope by decision:** PyTorch-era YOLOs (v5/v6/v8+). Supporting
  them means an ONNX import path — a different architecture (and, for
  Ultralytics models, AGPL licensing questions). Revisit only as a
  deliberate project.

## API & structure (deferred refactors)

- **[M] Structured results API.** A `Detection` struct (bbox, class id,
  class name, confidence, batch) plus a bundled class-name accessor, so
  users stop indexing `res[end-2, i]` by convention. Additive and
  non-breaking; arguably the highest-value UX item here.
- **[S] Coordinate-mapping helper.** Invert the letterbox (unpad + rescale
  to original image coordinates) as a public function; the math currently
  lives only inside `draw_boxes`.
- **[M] Concrete post-processing state.** Replace the per-output
  `Dict{Symbol,Any}` (and eventually `cfg::Dict{Symbol,Any}`) with typed
  structs. Touches the Adapt/Functors traversal used by the GPU and bumper
  paths — needs care, big readability/type-stability payoff.
- **[M] Decompose the `Yolo` constructor** (~300 lines: parse → layers →
  chains → output precompute) into testable stages.
- **[L] `prepareimage.jl` dispatch redesign.** The `#TODO: Make this
  multiple-dispatchy` if/elseif ladders. Most user-facing surface in the
  package; wants its own PR gated on the existing `prepare_image` test
  matrix.
- **[S] Drawing deps behind an extension.** Cairo/ImageDraw/Colors are hard
  deps used only by `draw_boxes`. Same treatment as `BenchmarkExt` (#138),
  but `draw_boxes` is exported and central, so it needs a deprecation-aware
  story.

## Process

- **[S] Documentation site** (Documenter.jl) — the README carries everything
  today.
- **PkgServer artifact 404s** on CI ("Failure artifact … Downloading
  artifact") are cosmetic: the pkg server can't know artifacts that aren't in
  a registered release yet, and the GitHub fallback succeeds. Self-resolves
  after the next release is registered.
- **Nightly CI failures** are an upstream GPUCompiler-on-julia-nightly
  precompile segfault, unrelated to this package.
