########################################################
##### DETECTION METRICS ################################
########################################################

"""
    DetectionMetrics

The result of [`evaluate`](@ref).

- `mAP`: mean average precision averaged over IoU thresholds 0.5:0.05:0.95, the
  headline COCO number.
- `mAP50`, `mAP75`: the same at a single IoU threshold.
- `ap`: average precision per class and per threshold, `nclasses × nthresholds`.
  `NaN` for classes with no ground truth, which are left out of the means.
- `ngt`: ground-truth boxes per class.
- `ndet`: detections kept, after thresholding and NMS.
"""
struct DetectionMetrics
    mAP::Float64
    mAP50::Float64
    mAP75::Float64
    ap::Matrix{Float64}
    iou_thresholds::Vector{Float64}
    ngt::Vector{Int}
    ndet::Int
end

function Base.show(io::IO, m::DetectionMetrics)
    nc = count(>(0), m.ngt)
    print(io, "DetectionMetrics(mAP=", round(m.mAP, digits = 4),
        ", mAP50=", round(m.mAP50, digits = 4),
        ", mAP75=", round(m.mAP75, digits = 4),
        ", classes=", nc, ", boxes=", sum(m.ngt), ", detections=", m.ndet, ")")
end

# IoU of two boxes given as (x1, y1, x2, y2). The letterbox maps both detections
# and ground truth by the same diagonal scale and offset, and IoU is invariant
# under that, so this is the same number as in the original image.
function box_iou(a::NTuple{4, Float32}, b::NTuple{4, Float32})
    iw = min(a[3], b[3]) - max(a[1], b[1])
    iw <= 0 && return 0.0
    ih = min(a[4], b[4]) - max(a[2], b[2])
    ih <= 0 && return 0.0
    inter = iw * ih
    areaa = (a[3] - a[1]) * (a[4] - a[2])
    areab = (b[3] - b[1]) * (b[4] - b[2])
    return inter / (areaa + areab - inter)
end

"""
    average_precision(matched, ngt)

Average precision from detections of one class sorted by descending score, where
`matched[i]` says whether detection `i` was a true positive.

Uses COCO's 101-point interpolation over the monotone-decreasing precision
envelope, so the number is comparable to `pycocotools`.
"""
function average_precision(matched::AbstractVector{Bool}, ngt::Int)
    ngt == 0 && return NaN
    isempty(matched) && return 0.0
    tp = 0
    fp = 0
    n = length(matched)
    recall = Vector{Float64}(undef, n)
    precision = Vector{Float64}(undef, n)
    for i in 1:n
        matched[i] ? (tp += 1) : (fp += 1)
        recall[i] = tp / ngt
        precision[i] = tp / (tp + fp)
    end
    for i in (n - 1):-1:1 # precision envelope
        precision[i] = max(precision[i], precision[i + 1])
    end
    total = 0.0
    j = 1
    for k in 0:100
        r = k / 100
        while j <= n && recall[j] < r
            j += 1
        end
        j > n && break # no recall this high; the rest contribute zero
        total += precision[j]
    end
    return total / 101
end

"""
    evaluate(model, data; kwargs...) -> DetectionMetrics

Mean average precision of `model` over `data`, a vector of [`TrainSample`](@ref)
just as [`train!`](@ref) takes.

Ground truth is letterboxed into the model's input space by the same code that
prepares training batches, so detections and targets are compared in one
coordinate system.

Keyword arguments:
- `image_loader=nothing`: needed when the samples hold file paths, as for `train!`
- `detect_thresh=0.001`: deliberately low. Average precision integrates over the
  whole precision-recall curve, so discarding low-confidence detections truncates
  the curve and reports a lower number than the model deserves.
- `overlap_thresh=0.45`: NMS IoU
- `iou_thresholds=0.5:0.05:0.95`: the thresholds AP is averaged over
- `max_dets=100`: detections kept per image, highest scoring first, as COCO does
- `silent=false`
"""
function evaluate(
        model, data::AbstractVector{<:TrainSample};
        image_loader = nothing, detect_thresh::Real = 0.001, overlap_thresh::Real = 0.45,
        iou_thresholds = 0.5:0.05:0.95, max_dets::Int = 100, silent::Bool = false,
    )
    isempty(data) && throw(ArgumentError("No samples to evaluate"))
    cfg = get_cfg(model)
    nclasses = cfg[:output][1][:classes]
    batchsize = get_input_size(model)[4]
    thresholds = collect(Float64, iou_thresholds)

    # (class, score, box) per detection and (class, box) per ground-truth box,
    # each tagged with the image it came from
    detclass = Int[]
    detscore = Float64[]
    detbox = NTuple{4, Float32}[]
    detimage = Int[]
    gtclass = Int[]
    gtbox = NTuple{4, Float32}[]
    gtimage = Int[]

    nimages = 0
    for start in 1:batchsize:length(data)
        idxs = start:min(start + batchsize - 1, length(data))
        x, boxes = make_training_batch(model, view(data, idxs); image_loader)
        if size(x, 4) < batchsize # pad the last chunk to the model's batch size
            padded = zeros(Float32, size(x, 1), size(x, 2), size(x, 3), batchsize)
            padded[:, :, :, 1:size(x, 4)] .= x
            x = padded
        end
        res = Array(model(uses_gpu(model) ? gpu(x) : x; detect_thresh, overlap_thresh))

        for (i, gt) in enumerate(boxes)
            imageid = nimages + i
            for n in axes(gt, 2)
                cx, cy, w, h = gt[2, n], gt[3, n], gt[4, n], gt[5, n]
                (w > 0 && h > 0) || continue
                push!(gtclass, round(Int, gt[1, n]))
                push!(gtbox, (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2))
                push!(gtimage, imageid)
            end
        end

        # keep the highest-scoring `max_dets` of each image, as COCO does
        perimage = [Int[] for _ in 1:length(idxs)]
        for c in axes(res, 2)
            b = round(Int, res[end, c])
            1 <= b <= length(idxs) && push!(perimage[b], c)
        end
        for (i, cols) in enumerate(perimage)
            length(cols) > max_dets &&
                (cols = partialsort(cols, 1:max_dets; by = c -> -res[end - 2, c]))
            for c in cols
                push!(detclass, round(Int, res[end - 1, c]))
                push!(detscore, res[end - 2, c])
                push!(detbox, (res[1, c], res[2, c], res[3, c], res[4, c]))
                push!(detimage, nimages + i)
            end
        end
        nimages += length(idxs)
        !silent && print("\r  evaluated $nimages/$(length(data)) images")
    end
    !silent && println()

    ngt = zeros(Int, nclasses)
    for c in gtclass
        1 <= c <= nclasses && (ngt[c] += 1)
    end

    ap = fill(NaN, nclasses, length(thresholds))
    for c in 1:nclasses
        ngt[c] == 0 && continue
        gsel = findall(==(c), gtclass)
        gts = [(gtimage[g], gtbox[g]) for g in gsel]
        # this class's ground-truth boxes, grouped by image, so matching a
        # detection only scans the boxes it could possibly overlap
        byimage = Dict{Int, Vector{Int}}()
        for (g, (imageid, _)) in enumerate(gts)
            push!(get!(byimage, imageid, Int[]), g)
        end
        noboxes = Int[]

        dsel = findall(==(c), detclass)
        dets = [(detscore[d], detimage[d], detbox[d]) for d in dsel]
        sort!(dets; by = first, rev = true)

        matched = Vector{Bool}(undef, length(dets))
        taken = Vector{Bool}(undef, length(gts))
        for (t, thresh) in enumerate(thresholds)
            # highest-scoring detection first, each taking the best free box
            fill!(taken, false)
            for (d, (_, imageid, box)) in enumerate(dets)
                best = 0.0
                bestg = 0
                for g in get(byimage, imageid, noboxes)
                    taken[g] && continue
                    iou = box_iou(box, gts[g][2])
                    if iou > best
                        best = iou
                        bestg = g
                    end
                end
                if bestg != 0 && best >= thresh
                    taken[bestg] = true
                    matched[d] = true
                else
                    matched[d] = false
                end
            end
            ap[c, t] = average_precision(matched, ngt[c])
        end
    end

    meanap(col) = begin
        vals = filter(!isnan, col)
        isempty(vals) ? NaN : sum(vals) / length(vals)
    end
    permean = [meanap(view(ap, :, t)) for t in eachindex(thresholds)]
    overall = let vals = filter(!isnan, permean)
        isempty(vals) ? NaN : sum(vals) / length(vals)
    end
    at(x) = begin
        i = findfirst(t -> isapprox(t, x; atol = 1.0e-6), thresholds)
        i === nothing ? NaN : permean[i]
    end
    return DetectionMetrics(overall, at(0.5), at(0.75), ap, thresholds, ngt, length(detclass))
end
