"""
    bboxiou!(out, box1, box2)

Compute the Intersection over Union (IoU) between a single bounding box `box1`
and multiple boxes `box2`, writing results into `out`.

- `box1`: A 4-element vector `[x1, y1, x2, y2]`
- `box2`: A 4×N matrix, where each column is a bounding box.
- `out`: A preallocated vector of length N to hold IoU results.

This version avoids allocations by reusing `out`.
"""
function bboxiou!(out, box1, box2)
    b1x1, b1y1, b1x2, b1y2 = box1
    b1area = (b1x2 - b1x1) * (b1y2 - b1y1)

    @inbounds for i in eachindex(out)
        b2x1 = box2[1, i]
        b2y1 = box2[2, i]
        b2x2 = box2[3, i]
        b2y2 = box2[4, i]

        rectx1 = max(b1x1, b2x1)
        recty1 = max(b1y1, b2y1)
        rectx2 = min(b1x2, b2x2)
        recty2 = min(b1y2, b2y2)

        w = max(0f0, rectx2 - rectx1)
        h = max(0f0, recty2 - recty1)
        inter = w * h
        b2area = (b2x2 - b2x1) * (b2y2 - b2y1)
        out[i] = inter / (b1area + b2area - inter)
    end
    return out
end

"""
    nms(dets, iou_thresh)

Perform a simple Non-Maximum Suppression (NMS) on the detections `dets`.
`dets` is a 2D array of shape (≥5, N), assumed to be sorted in descending
order by the end-2 column (i.e. class confidence score). `iou_thresh` is
the overlap threshold above which boxes are considered duplicates.

Returns an array of indexes `keep` of the columns in `dets` you want to keep.
"""
function nms(dets::AbstractArray{T}, iou_thresh) where T
    N = size(dets, 2)
    idxs = similar(dets, Int, N)
    @inbounds for j in 1:N
        idxs[j] = j
    end

    keep = Vector{Int}()
    ious = similar(dets, T, N) # large enough to reuse

    idx_len = N
    while idx_len > 0
        i = idxs[1]
        push!(keep, i)

        if idx_len == 1
            break
        end

        # Compute IoUs between the top candidate and the rest
        b2_len = idx_len - 1
        b1 = @view dets[1:4, i]
        b2s = @view dets[1:4, idxs[2:idx_len]]
        # we must take a view into ious because bboxiou! writes into it
        bboxiou!(view(ious, 1:b2_len), b1, b2s)

        # Build new candidate list, excluding boxes with IoU >= threshold
        write_idx = 0
        for j in 1:b2_len
            if ious[j] < iou_thresh
                write_idx += 1
                idxs[write_idx] = idxs[j+1]
            end
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
function perform_detection_nms(batchout, overlap_thresh, batchsize::Int)
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

            keep = nms(sorted_dets, overlap_thresh)

            # @info "Batch $b, class $cls: input=$(size(sorted_dets, 2)), kept=$(length(keep))"

            @inbounds for k in keep
                output[:, i] = sorted_dets[:, k]
                i += 1
            end
        end
    end
    return output[:, 1:i-1]
end
