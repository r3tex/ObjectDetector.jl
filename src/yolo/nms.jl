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
    b2x1, b2y1, b2x2, b2y2 = view(box2, 1, :), view(box2, 2, :), view(box2, 3, :), view(box2, 4, :)

    b1area = (b1x2 - b1x1) * (b1y2 - b1y1)

    @inbounds for i in eachindex(out)
        rectx1 = max(b1x1, b2x1[i])
        recty1 = max(b1y1, b2y1[i])
        rectx2 = min(b1x2, b2x2[i])
        recty2 = min(b1y2, b2y2[i])
        w = max(0f0, rectx2 - rectx1)
        h = max(0f0, recty2 - recty1)
        inter = w * h
        b2area = (b2x2[i] - b2x1[i]) * (b2y2[i] - b2y1[i])
        out[i] = inter / (b1area + b2area - inter)
    end
end

"""
    nms(dets, iou_thresh)

Perform a simple Non-Maximum Suppression (NMS) on the detections `dets`.
`dets` is a 2D array of shape (≥5, N), assumed to be sorted in descending
order by the 5th column (i.e., confidence or score). `iou_thresh` is
the overlap threshold above which boxes are considered duplicates.

Returns an array of indexes `keep` of the columns in `dets` you want to keep.
"""
function nms(dets::AbstractArray, iou_thresh)
    N = size(dets, 2)
    idxs = similar(dets, Int, N)
    @inbounds for j in 1:N
        idxs[j] = j
    end

    keep = Vector{Int}()
    ious = similar(dets, N) # large enough to reuse

    idx_len = N
    while idx_len > 0
        i = idxs[1]
        push!(keep, i)

        if idx_len == 1
            break
        end

        # Compute IoUs between i and rest
        b2_len = idx_len - 1
        bboxiou!(ious[1:b2_len], view(dets, 1:4, i), view(dets, 1:4, idxs[2:idx_len]))

        # Compress idxs in-place: keep only boxes with IoU < threshold
        write_idx = 1  # we'll overwrite idxs[2:end] starting at idxs[1]
        for j in 1:b2_len
            if ious[j] < iou_thresh
                write_idx += 1
                idxs[write_idx] = idxs[j + 1]
            end
        end
        idx_len = write_idx - 1
    end

    return keep
end

"""
    perform_detection_nms(batchout, overlapThresh, batchsize)

For each batch `b` in `1:batchsize`, extract the detections from `batchout`,
group them by class, sort each group by the 5th column (score) descending, and
run NMS to remove duplicates using bboxiou and overlapThresh.

Returns a Vector of detection matrices, each of size (num_fields, kept_boxes).

The input `batchout` is a 2D array of shape (num_fields, N), where `N` is the
total number of detections across all batches.

batchout rows:
- 1-4: the bounding box coordinates x1, y1, x2, y2
- 5: the confidence/score.
- scores for each class
- second-to-last row (end-1) has the class index.
- The last row is the batch index
"""
function perform_detection_nms(batchout, overlapThresh, batchsize::Int)
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

        @inbounds for (local_col_idx, cls) in enumerate(class_ids)
            if cls in seen_classes
                continue
            end
            push!(seen_classes, cls)

            c_idxs = findall(==(cls), class_ids)
            if isempty(c_idxs)
                continue
            end

            dets = @view page[:, c_idxs]

            scores = @view dets[5, :]
            sorted_idx = sortperm(scores, rev=true)
            sorted_dets = dets[:, sorted_idx]  # copy, not @view!

            keep = nms(sorted_dets, overlapThresh)

            # @info "Batch $b, class $cls: input=$(size(sorted_dets, 2)), kept=$(length(keep))"

            @inbounds for k in keep
                output[:, i] = sorted_dets[:, k]
                i += 1
            end
        end
    end
    return output[:, 1:i-1]
end
