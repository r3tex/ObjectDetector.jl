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
    # The bounding box coords are in dets[1:4, :].
    # The columns are sorted by score already (descending).
    idxs = collect(1:size(dets, 2))        # candidate column indexes
    keep = Int[]                            # final picks

    while !isempty(idxs)
        # Pick the top-scoring box (first in the sorted list)
        i = first(idxs)
        push!(keep, i)

        # If there's only one left, no need to compute IoU
        if length(idxs) == 1
            break
        end

        # Compute IoU of the chosen box with the rest
        # - bboxiou should accept two bounding boxes or a box vs many boxes
        #   so it returns a vector of IoUs in this usage.
        iou = similar(dets, length(idxs) - 1)
        bboxiou!(iou, dets[1:4, i], dets[1:4, idxs[2:end]])

        # Find which have IoU >= threshold
        to_remove = findall(≥(iou_thresh), iou)

        # Those indexes in `to_remove` are offset by +1 in the `idxs` array
        remove_idxs = idxs[to_remove .+ 1]

        # Remove them all from `idxs`
        filter!(x -> x ∉ remove_idxs, idxs)

        # Also remove the “picked” box (we already kept i)
        filter!(x -> x != i, idxs)
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
function perform_detection_nms(
    batchout,
    overlapThresh,
    batchsize::Int
)
    output = []  # array of matrices

    @views for b in 1:batchsize
        # Get columns that belong to batch b
        b_idxs = findall(x -> x == b, batchout[end, :])
        if isempty(b_idxs)
            continue
        end
        page = batchout[:, b_idxs]

        # For each class present in this batch
        present_classes = unique(page[end-1, :])
        for c in present_classes
            # Gather all detections that match class c
            c_idxs = findall(x -> x == c, page[end-1, :])
            if isempty(c_idxs)
                continue
            end

            # Extract and sort by confidence (the 5th row),
            # descending:
            dets = sortslices(page[:, c_idxs], dims=2, by = x -> x[5], rev = true)

            # Run NMS to get the indexes of the columns to keep
            keep = nms(dets, overlapThresh)

            # Save the filtered detections
            push!(output, dets[:, keep])
        end
    end

    return output
end
