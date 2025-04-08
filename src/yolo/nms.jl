"""
    bboxiou!(out, box1, box2; distance::Bool=false)

Compute IoU or Distance-IoU (DIoU) between a single bounding box `box1`
and multiple boxes `box2`, writing results into `out`.

- `box1`: A 4-element vector `[x1, y1, x2, y2]`
- `box2`: A 4×N matrix, where each column is a bounding box
- `out`: A preallocated vector of length N for results
- `distance`: if true, computes DIoU; otherwise standard IoU

Avoids allocations by reusing `out`.
"""
function bboxiou!(out::AbstractArray{T}, box1, box2; distance::Bool=false) where T
    b1x1, b1y1, b1x2, b1y2 = box1
    b1w = b1x2 - b1x1
    b1h = b1y2 - b1y1
    b1area = b1w * b1h

    b1cx = distance ? (b1x1 + b1x2) / 2 : zero(eltype(out))
    b1cy = distance ? (b1y1 + b1y2) / 2 : zero(eltype(out))

    @inbounds for i in eachindex(out)
        b2x1 = box2[1, i]
        b2y1 = box2[2, i]
        b2x2 = box2[3, i]
        b2y2 = box2[4, i]

        # Intersection
        rectx1 = max(b1x1, b2x1)
        recty1 = max(b1y1, b2y1)
        rectx2 = min(b1x2, b2x2)
        recty2 = min(b1y2, b2y2)
        w = max(zero(T), rectx2 - rectx1)
        h = max(zero(T), recty2 - recty1)
        inter = w * h

        b2area = (b2x2 - b2x1) * (b2y2 - b2y1)
        union = b1area + b2area - inter
        iou = inter / union

        if distance
            b2cx = (b2x1 + b2x2) / 2
            b2cy = (b2y1 + b2y2) / 2
            center_dist_sq = (b1cx - b2cx)^2 + (b1cy - b2cy)^2

            enc_x1 = min(b1x1, b2x1)
            enc_y1 = min(b1y1, b2y1)
            enc_x2 = max(b1x2, b2x2)
            enc_y2 = max(b1y2, b2y2)
            c2 = (enc_x2 - enc_x1)^2 + (enc_y2 - enc_y1)^2 + eps(eltype(out))

            out[i] = iou - center_dist_sq / c2
        else
            out[i] = iou
        end
    end
    return out
end

"""
    nms(dets, iou_thresh; kind=:greedynms, beta=0.6f0)

Performs Non-Maximum Suppression (NMS) on a set of detection boxes `dets`, returning the indices
of boxes to keep. This function supports multiple NMS strategies:

Arguments:
- `dets`: A matrix of shape (≥5, N), where each column represents a detection.
          - Rows 1:4 are bounding box coordinates: `[x1, y1, x2, y2]`
          - Row end-2 is the class confidence score (objectness score x max(class probabilities)
- `iou_thresh`: IoU threshold above which boxes are considered duplicates

Keyword Arguments:
- `kind` (`Symbol`): NMS method to use. Supported:
    - `:greedynms` (default): traditional hard-threshold NMS
    - `:diounms`: score-decay using IoU penalty (`score *= 1 - IoU`)
    - `:soft`: Soft-NMS using exponential decay (`score *= exp(-IoU^2 / beta)`)
- `beta` (`Float32`): smoothing factor for soft-NMS (default `0.6`)

Returns:
- `keep`: a vector of column indices in `dets` to retain
"""
function nms(dets::AbstractArray{T}, iou_thresh; kind::Symbol = :greedynms, beta::T = T(0.6)) where T
    N = size(dets, 2)
    idxs = similar(dets, Int, N)
    @inbounds for j in 1:N
        idxs[j] = j
    end

    keep = Vector{Int}()
    ious = similar(dets, T, N)
    scores = similar(dets, T, N)
    @inbounds for j in 1:N
        scores[j] = dets[end-2, j]
    end

    idx_len = N
    while idx_len > 0
        i = idxs[1]
        push!(keep, i)
        if idx_len == 1
            break
        end
        b2_len = idx_len - 1
        b1 = @view dets[1:4, i]
        b2s = @view dets[1:4, idxs[2:idx_len]]

        # Note that even diounms uses iou in inference. During training apparently it uses diou though.
        # That needs a further investigation though
        distance = false
        bboxiou!(view(ious, 1:b2_len), b1, b2s; distance)

        write_idx = 0
        if kind === :greedynms || kind === :diounms
            @inbounds for j in 1:b2_len
                if ious[j] < iou_thresh
                    write_idx += 1
                    idxs[write_idx] = idxs[j+1]
                end
            end
        elseif kind === :soft # untested
            @inbounds for j in 1:b2_len
                decay = exp(-(ious[j]^2) / beta)
                scores[idxs[j+1]] *= decay
            end
            @inbounds for j in 2:idx_len
                key = idxs[j]
                k = j - 1
                while k >= 1 && scores[idxs[k]] < scores[key]
                    idxs[k + 1] = idxs[k]
                    k -= 1
                end
                idxs[k + 1] = key
            end
            write_idx = idx_len - 1
        else
            error("Unknown NMS kind: $kind")
        end
        idx_len = write_idx
    end
    return keep
end

"""
    perform_detection_nms(batchout, overlap_thresh, batchsize)

For each batch `b` in `1:batchsize`, extract the detections from `batchout`,
group them by class, sort each group by the end-2 column (class confidence score) descending, and
run NMS to remove duplicates using bboxiou and overlap_thresh.

Returns a Vector of detection matrices, each of size (num_fields, kept_boxes).

The input `batchout` is a 2D array of shape (num_fields, N), where `N` is the
total number of detections across all batches.

batchout rows:
- 1-4: the bounding box coordinates x1, y1, x2, y2
- 5: the confidence/score.
- scores for each class
- end-2: the class confidence score (for NMS)
- end-1: the class index
- The last row is the batch index
"""
function perform_detection_nms(batchout, overlap_thresh, batchsize::Int; kind::Symbol=:greedynms, beta::Float32=0.6f0)
    output = similar(batchout)
    i = 1  # index for writing into `output`

    for b in 1:batchsize
        b_cols = findall(==(b), @view(batchout[end, :]))
        if isempty(b_cols)
            continue
        end

        page = @view batchout[:, b_cols]
        class_ids = @view page[end-1, :]

        seen_classes = Set{eltype(class_ids)}()

        for (local_col_idx, cls) in enumerate(class_ids)
            if cls in seen_classes
                continue
            end
            push!(seen_classes, cls)

            c_idxs = findall(==(cls), class_ids)
            if isempty(c_idxs)
                continue
            end

            dets = @view page[:, c_idxs]

            scores = @view dets[end-2, :]
            sorted_idx = sortperm(scores, rev=true)
            # nms takes views of sorted_dets and copying here results in lower allocs and faster nms
            sorted_dets = dets[:, sorted_idx]

            keep = nms(sorted_dets, overlap_thresh; kind, beta)

            @inbounds for k in keep
                output[:, i] = sorted_dets[:, k]
                i += 1
            end
        end
    end
    return output[:, 1:i-1]
end
