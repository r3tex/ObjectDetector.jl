########################################################
##### TRAINING #########################################
########################################################

import ..train!, ..save_weights, ..TrainSample, ..make_training_batch
import Random

# Pure (non-mutating) forward pass, differentiable with Zygote. The regular
# inference path mutates preallocated buffers for speed; these methods run the
# same layers functionally, resolving skip layers through `outs`, the tuple of
# chain outputs computed so far, instead of through their captured buffers.
_pure(l, outs, x) = l(x) # Flux.Conv, MaxPoolLayer, Reorg are already non-mutating
function _pure(c::Flux.Chain, outs, x)
    for l in c.layers
        x = _pure(l, outs, x)
    end
    return x
end
_pure(l::BroadcastActivation, outs, x) = l.act.(x)
_pure(l::Upsample, outs, x) = Flux.NNlib.upsample_nearest(x, (l.stride, l.stride))
_pure(l::RouteLayer, outs, x) = l.channels === nothing ? outs[l.src] : outs[l.src][:, :, l.channels, :]
_pure(l::ShortcutAdd, outs, x) = l.act.(x .+ outs[l.src])
_pure(l::CatLayer, outs, x) = cat(map(s -> outs[s], l.srcs)...; dims=3)

"""
    raw_head_outputs(chain, x, headidxs)

Run the model chain functionally (non-mutating, differentiable) on input `x`
and return the raw (pre-decode) outputs of the chains at `headidxs`.
"""
function raw_head_outputs(chain::Flux.Chain, x, headidxs)
    outs = ()
    for c in chain.layers
        x = _pure(c, outs, x)
        outs = (outs..., x)
    end
    return map(i -> outs[i], headidxs)
end

# Static per-head metadata needed by the loss, precomputed once per training run
struct TrainHead{O,A}
    chainidx::Int
    w::Int
    h::Int
    na::Int
    nc::Int
    sxy::Float32
    new_coords::Bool              # scaled-YOLOv4/v7 decode; raw outputs are post-sigmoid
    offset::O                     # (w, h, 2, na, 1) cell offsets, grid units
    anchors_grid::A               # (1, 1, 2, na, 1) anchor sizes, grid units
    anchors_norm::Matrix{Float32} # 2×na, normalized to model input (for target assignment)
end

function train_heads(yolo::Yolo)
    cfg = yolo.cfg
    cfg[:laststage] === :yolo || error("Training is not supported for [region]-output (yolov2-family) models")
    maybe_gpu(z) = uses_gpu(yolo) ? gpu(z) : z
    heads = TrainHead[]
    for (i, out) in enumerate(yolo.out)
        o = cfg[:output][i]
        new_coords = o[:new_coords] == 1
        # classic heads emit logits (linear final conv); new_coords heads emit
        # post-sigmoid values (logistic final conv). The loss depends on which.
        expected_act = new_coords ? "logistic" : "linear"
        out[:final_conv_activation] == expected_act ||
            error("Training with new_coords=$(o[:new_coords]) requires a \"$expected_act\" activation on the conv layer preceding each [yolo] layer (got \"$(out[:final_conv_activation])\")")
        w, h, attrs, na, _ = out[:size]
        nc = attrs - 5
        stridew = cfg[:width] ÷ w
        strideh = cfg[:height] ÷ h
        anchormask = haskey(o, :mask) ? o[:mask] .+ 1 : (1:length(o[:anchors])÷2)
        apx = reshape(Float32.(o[:anchors]), 2, :)[:, anchormask] # anchor sizes in pixels
        offset = zeros(Float32, w, h, 2, na, 1)
        for j in 0:w-1, k in 0:h-1
            offset[j+1, k+1, 1, :, 1] .= j
            offset[j+1, k+1, 2, :, 1] .= k
        end
        agrid = zeros(Float32, 1, 1, 2, na, 1)
        agrid[1, 1, 1, :, 1] .= apx[1, :] ./ stridew
        agrid[1, 1, 2, :, 1] .= apx[2, :] ./ strideh
        anorm = apx ./ Float32[cfg[:width], cfg[:height]]
        push!(heads, TrainHead(out[:idx], w, h, na, nc, Float32(out[:scale_x_y]), new_coords,
                               maybe_gpu(offset), maybe_gpu(agrid), anorm))
    end
    return heads
end

"""
    build_targets(heads, boxes_batch)

Build training targets from ground-truth boxes. Each box is assigned to the
single anchor (across all heads) with the best shape-IoU, at the grid cell
containing the box center. Returns per-head named tuples of target arrays.
`tw`/`th` default to 1 (not 0) so the CIoU term stays finite at unassigned
cells, where it is masked out anyway.
"""
function build_targets(heads, boxes_batch)
    nb = length(boxes_batch)
    tgts = map(heads) do hd
        (tobj = zeros(Float32, hd.w, hd.h, 1, hd.na, nb),
         tx   = zeros(Float32, hd.w, hd.h, 1, hd.na, nb),
         ty   = zeros(Float32, hd.w, hd.h, 1, hd.na, nb),
         tw   = ones(Float32, hd.w, hd.h, 1, hd.na, nb),
         th   = ones(Float32, hd.w, hd.h, 1, hd.na, nb),
         twr  = zeros(Float32, hd.w, hd.h, 1, hd.na, nb),
         thr  = zeros(Float32, hd.w, hd.h, 1, hd.na, nb),
         tcls = zeros(Float32, hd.w, hd.h, hd.nc, hd.na, nb),
         npos = Ref(0))
    end
    for (b, boxes) in enumerate(boxes_batch)
        for n in axes(boxes, 2)
            cls = round(Int, boxes[1, n])
            cx, cy, bw, bh = boxes[2, n], boxes[3, n], boxes[4, n], boxes[5, n]
            (bw > 0 && bh > 0) || continue
            best_iou = -1f0
            best = (1, 1)
            for (hi, hd) in enumerate(heads), a in 1:hd.na
                aw, ah = hd.anchors_norm[1, a], hd.anchors_norm[2, a]
                inter = min(bw, aw) * min(bh, ah)
                iou = inter / (bw * bh + aw * ah - inter)
                if iou > best_iou
                    best_iou = iou
                    best = (hi, a)
                end
            end
            hi, a = best
            hd = heads[hi]
            t = tgts[hi]
            1 <= cls <= hd.nc || error("Box class index $cls outside the model's class range 1:$(hd.nc)")
            gi = clamp(floor(Int, cx * hd.w) + 1, 1, hd.w)
            gj = clamp(floor(Int, cy * hd.h) + 1, 1, hd.h)
            t.tobj[gi, gj, 1, a, b] == 0 || continue # cell/anchor already claimed by another box
            t.tobj[gi, gj, 1, a, b] = 1
            t.tx[gi, gj, 1, a, b] = cx * hd.w
            t.ty[gi, gj, 1, a, b] = cy * hd.h
            t.tw[gi, gj, 1, a, b] = bw * hd.w
            t.th[gi, gj, 1, a, b] = bh * hd.h
            # raw-space wh targets (the raw head output value that decodes to
            # the target size), for the stabilizing regression term in the loss
            rw = bw / hd.anchors_norm[1, a]
            rh = bh / hd.anchors_norm[2, a]
            if hd.new_coords
                t.twr[gi, gj, 1, a, b] = sqrt(rw) / hd.sxy
                t.thr[gi, gj, 1, a, b] = sqrt(rh) / hd.sxy
            else
                t.twr[gi, gj, 1, a, b] = log(rw)
                t.thr[gi, gj, 1, a, b] = log(rh)
            end
            t.tcls[gi, gj, cls, a, b] = 1
            t.npos[] += 1
        end
    end
    return tgts
end

# numerically stable binary cross-entropy on a logit
bce_logit(x, z) = max(x, 0f0) - x * z + log1p(exp(-abs(x)))

# binary cross-entropy on a probability (for post-sigmoid new_coords heads)
bce_prob(p, z) = -(z * log(p + 1f-7) + (1f0 - z) * log(1f0 - p + 1f-7))

"""
    box_ciou(px, py, pw, ph, tx, ty, tw, th)

Complete-IoU (CIoU) between a predicted and a target box, both given as
center/size. Scalar function intended for broadcasting; differentiable.
"""
function box_ciou(px, py, pw, ph, tx, ty, tw, th)
    ϵ = 1f-7
    px1 = px - pw / 2; px2 = px + pw / 2
    py1 = py - ph / 2; py2 = py + ph / 2
    tx1 = tx - tw / 2; tx2 = tx + tw / 2
    ty1 = ty - th / 2; ty2 = ty + th / 2
    iw = max(min(px2, tx2) - max(px1, tx1), 0f0)
    ih = max(min(py2, ty2) - max(py1, ty1), 0f0)
    inter = iw * ih
    union_ = pw * ph + tw * th - inter + ϵ
    iou = inter / union_
    cw = max(px2, tx2) - min(px1, tx1)
    ch = max(py2, ty2) - min(py1, ty1)
    c2 = cw^2 + ch^2 + ϵ
    ρ2 = (px - tx)^2 + (py - ty)^2
    v = (4f0 / Float32(π)^2) * (atan(tw / (th + ϵ)) - atan(pw / (ph + ϵ)))^2
    α = v / (1f0 - iou + v + ϵ)
    return iou - ρ2 / c2 - α * v
end

"""
    train_loss(chain, x, heads, tgts; box_weight=1, obj_weight=1, cls_weight=1, noobj_weight=0.5)

YOLO training loss on a batch: CIoU box loss and BCE class loss averaged over
assigned (positive) cells, plus BCE objectness loss with positives and
negatives normalized separately (`noobj_weight` scales the negative part).
"""
function train_loss(chain, x, heads, tgts; box_weight=1f0, obj_weight=1f0, cls_weight=1f0, noobj_weight=0.5f0)
    raws = raw_head_outputs(chain, x, map(hd -> hd.chainidx, heads))
    lbox = 0f0
    lobj = 0f0
    lcls = 0f0
    for (hi, hd) in enumerate(heads)
        t = tgts[hi]
        raw = reshape(raws[hi], hd.w, hd.h, 5 + hd.nc, hd.na, :)
        txy_raw = raw[:, :, 1:2, :, :]
        twh_raw = raw[:, :, 3:4, :, :]
        tobj_raw = raw[:, :, 5:5, :, :]
        tcls_raw = raw[:, :, 6:(5 + hd.nc), :, :]
        # decode to grid units (same transform as inference, minus the pixel
        # scaling). Classic heads emit logits; new_coords heads emit
        # post-sigmoid values, so the box decode differs and the BCE terms
        # operate on probabilities instead of logits.
        if hd.new_coords
            pxy = (txy_raw .* hd.sxy .- (hd.sxy - 1f0) / 2f0) .+ hd.offset
            pwh = (twh_raw .* hd.sxy).^2 .* hd.anchors_grid
            obj_bce = bce_prob.(tobj_raw, t.tobj)
            cls_bce = bce_prob.(tcls_raw, t.tcls)
        else
            pxy = (σ.(txy_raw) .* hd.sxy .- (hd.sxy - 1f0) / 2f0) .+ hd.offset
            pwh = exp.(clamp.(twh_raw, -9f0, 9f0)) .* hd.anchors_grid
            obj_bce = bce_logit.(tobj_raw, t.tobj)
            cls_bce = bce_logit.(tcls_raw, t.tcls)
        end
        px = pxy[:, :, 1:1, :, :]; py = pxy[:, :, 2:2, :, :]
        pw = pwh[:, :, 1:1, :, :]; ph = pwh[:, :, 2:2, :, :]
        npos = t.npos
        nneg = length(t.tobj) - npos
        lobj += sum(obj_bce .* t.tobj) / max(npos, 1) +
                noobj_weight * sum(obj_bce .* (1f0 .- t.tobj)) / max(nneg, 1)
        if npos > 0
            ci = box_ciou.(px, py, pw, ph, t.tx, t.ty, t.tw, t.th)
            lbox += sum((1f0 .- ci) .* t.tobj) / npos
            # Stabilizing raw-space wh regression. It shares its optimum with
            # the CIoU term (both are minimized when the decoded size equals
            # the target), but unlike CIoU its gradient does not vanish when
            # the prediction badly over/undershoots — without it the exp
            # decode of classic heads can run away irrecoverably.
            lbox += sum(((twh_raw[:, :, 1:1, :, :] .- t.twr).^2 .+
                         (twh_raw[:, :, 2:2, :, :] .- t.thr).^2) .* t.tobj) / npos
            lcls += sum(cls_bce .* t.tobj) / npos
        end
    end
    return box_weight * lbox + obj_weight * lobj + cls_weight * lcls
end

"""
    train!(model, data::AbstractVector{<:TrainSample}; kwargs...)

Fine-tune / train a YOLO model in place. The model's conv weights are updated;
inference on the model works as usual afterwards (and during training).

By default batchnorm parameters are folded into the conv weights at load
time, so training behaves like fine-tuning with frozen batchnorm statistics —
well suited to adapting pretrained weights. For from-scratch training,
construct the model with `trainable_batchnorm=true` (typically together with
`weights_stop_layer=0` for random init and `warmup_batches > 0`) to keep live,
trainable batchnorm layers.

Keyword arguments:
- `epochs=10`, `batchsize=8` (independent of the model's inference batchsize)
- `lr=0.001`: Adam learning rate (ignored if `opt` is given)
- `opt=nothing`: a Flux/Optimisers optimiser rule to use instead of Adam
- `image_loader=nothing`: function to load images for path-based samples (e.g. `FileIO.load`)
- `shuffle=true`, `rng=Random.default_rng()`
- `warmup_batches=0`: linearly ramp the learning rate from 0 over this many
  batches (darknet's burn-in), recommended for from-scratch training
- `flip_augment=false`: randomly mirror images (and their boxes) horizontally
- `box_weight=1, obj_weight=1, cls_weight=1, noobj_weight=0.5`: loss term weights
- `checkpoint_dir=nothing`: if set, saves `checkpoint-epochNNNN.weights` (darknet format, reloadable with the same cfg) every `checkpoint_every=1` epochs
- `silent=false`

Returns `(; losses)`: the mean loss per epoch.
"""
function train!(yolo::Yolo, data::AbstractVector{<:TrainSample};
                epochs::Int=10, batchsize::Int=8, lr::Real=1f-3, opt=nothing,
                image_loader=nothing, shuffle::Bool=true, rng=Random.default_rng(),
                warmup_batches::Int=0, flip_augment::Bool=false,
                box_weight::Real=1f0, obj_weight::Real=1f0, cls_weight::Real=1f0, noobj_weight::Real=0.5f0,
                checkpoint_dir::Union{Nothing,AbstractString}=nothing, checkpoint_every::Int=1,
                silent::Bool=false)
    isempty(data) && throw(ArgumentError("No training samples provided"))
    heads = train_heads(yolo)
    chain = yolo.chain
    opt_state = Flux.setup(something(opt, Flux.Adam(lr)), chain)
    maybe_gpu(z) = uses_gpu(yolo) ? gpu(z) : z
    box_weight, obj_weight, cls_weight, noobj_weight = Float32.((box_weight, obj_weight, cls_weight, noobj_weight))
    epoch_losses = Float32[]
    nseen = 0 # batches seen, for warmup
    for e in 1:epochs
        idxs = collect(eachindex(data))
        shuffle && Random.shuffle!(rng, idxs)
        total = 0f0
        nbatches = 0
        for lo in 1:batchsize:length(idxs)
            sel = idxs[lo:min(lo + batchsize - 1, length(idxs))]
            x, boxes_batch = make_training_batch(yolo, data[sel]; image_loader)
            if flip_augment
                for i in eachindex(boxes_batch)
                    rand(rng) < 0.5 || continue
                    reverse!(view(x, :, :, :, i); dims=1) # dim 1 is model x (width)
                    boxes_batch[i][2, :] .= 1f0 .- boxes_batch[i][2, :]
                end
            end
            if warmup_batches > 0 && nseen <= warmup_batches
                nseen += 1
                Flux.adjust!(opt_state, Float32(lr) * min(nseen / warmup_batches, 1f0))
            end
            tgts_cpu = build_targets(heads, boxes_batch)
            tgts = [(tobj=maybe_gpu(t.tobj), tx=maybe_gpu(t.tx), ty=maybe_gpu(t.ty),
                     tw=maybe_gpu(t.tw), th=maybe_gpu(t.th),
                     twr=maybe_gpu(t.twr), thr=maybe_gpu(t.thr), tcls=maybe_gpu(t.tcls),
                     npos=t.npos[]) for t in tgts_cpu]
            xg = maybe_gpu(x)
            l, gs = Flux.withgradient(m -> train_loss(m, xg, heads, tgts;
                                                      box_weight, obj_weight, cls_weight, noobj_weight), chain)
            if !isfinite(l)
                @warn "Non-finite loss; skipping batch" epoch=e batch=nbatches+1
                continue
            end
            Flux.update!(opt_state, chain, gs[1])
            total += l
            nbatches += 1
        end
        epoch_loss = total / max(nbatches, 1)
        push!(epoch_losses, epoch_loss)
        !silent && println("epoch $e/$epochs  loss: $epoch_loss")
        if checkpoint_dir !== nothing && (e % checkpoint_every == 0 || e == epochs)
            mkpath(checkpoint_dir)
            save_weights(yolo, joinpath(checkpoint_dir, "checkpoint-epoch$(lpad(e, 4, '0')).weights"))
        end
    end
    return (; losses = epoch_losses)
end

conv_layers(chain) = map(first, conv_bn_layers(chain))

# flatten the chain and pair each Conv with its live BatchNorm, if any
# (present only for models built with `trainable_batchnorm=true`)
function conv_bn_layers(chain)
    layers = Any[]
    _flatten_layers!(layers, chain)
    pairs = Tuple{Any,Any}[]
    for (i, l) in enumerate(layers)
        if l isa Flux.Conv
            bnl = i < length(layers) && layers[i+1] isa Flux.BatchNorm ? layers[i+1] : nothing
            push!(pairs, (l, bnl))
        end
    end
    return pairs
end
_flatten_layers!(out, c::Flux.Chain) = foreach(l -> _flatten_layers!(out, l), c.layers)
_flatten_layers!(out, l) = push!(out, l)

"""
    save_weights(model, path)

Save the model's weights in darknet `.weights` format, loadable with the same
cfg file. For models built with `trainable_batchnorm=true` the live batchnorm
parameters are written directly (fully darknet-compatible). Otherwise, conv
layers that had batchnorm in the cfg are written with identity batchnorm
parameters (the batchnorm was folded into the conv weights at load time),
which fold back to exactly the saved weights when reloaded.
"""
function save_weights(yolo::Yolo, path::AbstractString)
    blocks = get(yolo.cfg, :layerblocks, nothing)
    blocks === nothing && error("Model does not carry its cfg layer blocks; construct the model with this version of ObjectDetector to enable saving")
    convs = conv_bn_layers(yolo.chain)
    nconvblocks = count(b -> first(b) === :convolutional, blocks)
    nconvblocks == length(convs) || error("Internal error: $(nconvblocks) conv cfg blocks but $(length(convs)) conv layers")
    v = yolo.cfg[:darknetversion]
    open(path, "w") do io
        write(io, Int32(v.major), Int32(v.minor), Int32(v.patch))
        if v < v"0.2.0"
            write(io, Int32(yolo.cfg[:seen]))
        else
            write(io, Int32(yolo.cfg[:seen]), Int32(yolo.cfg[:seen_images]))
        end
        i = 0
        for (blocktype, block) in blocks
            blocktype === :convolutional || continue
            i += 1
            c, bnl = convs[i]
            wgt = Array(cpu(c.weight))
            fl = size(wgt, 4)
            if haskey(block, :batch_normalize)
                if bnl === nothing
                    # batchnorm was folded into the conv at load time: write the
                    # conv bias as the bn bias with identity bn parameters
                    write(io, Vector{Float32}(cpu(c.bias)))
                    write(io, ones(Float32, fl))       # bn scale
                    write(io, zeros(Float32, fl))      # bn mean
                    write(io, fill(1f0 - 1f-5, fl))    # bn variance (variance + eps == 1)
                else
                    write(io, Vector{Float32}(cpu(bnl.β)))
                    write(io, Vector{Float32}(cpu(bnl.γ)))
                    write(io, Vector{Float32}(cpu(bnl.μ)))
                    write(io, Vector{Float32}(cpu(bnl.σ²)))
                end
            else
                write(io, Vector{Float32}(cpu(c.bias)))
            end
            write(io, collect(flip(wgt))) # undo the load-time kernel flip
        end
    end
    return path
end
