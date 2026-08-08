########################################################
##### TRAINING DATA HANDLING ###########################
########################################################

"""
    TrainSample(image, boxes)

One training sample: an image and its ground-truth boxes.

`image` can be an in-memory image (a `Colorant` matrix or a numeric array) or
a file path (`String`), in which case an `image_loader` function must be
passed to `train!` (e.g. `FileIO.load`).

`boxes` is a `5×N Matrix{Float32}` with one column per object:
`[class, cx, cy, w, h]`, where `class` is the 1-based class index and the
box center/size are normalized (0-1) relative to the image width and height
(the darknet label convention, except 1-based class indices).
"""
struct TrainSample{I}
    image::I
    boxes::Matrix{Float32}
    function TrainSample(image::I, boxes::AbstractMatrix) where {I}
        size(boxes, 1) == 5 || throw(ArgumentError("boxes must be 5×N ([class, cx, cy, w, h] columns), got $(size(boxes, 1))×$(size(boxes, 2))"))
        return new{I}(image, Matrix{Float32}(boxes))
    end
end

"""
    load_darknet_labels(path)

Read a darknet-format label file (`class cx cy w h` per line, 0-based class,
normalized coordinates) and return a 5×N `Matrix{Float32}` with 1-based
class indices, suitable for `TrainSample`.
"""
function load_darknet_labels(path::AbstractString)
    rows = [parse.(Float32, split(line)) for line in eachline(path) if !isempty(strip(line))]
    boxes = zeros(Float32, 5, length(rows))
    for (i, r) in enumerate(rows)
        length(r) == 5 || error("Malformed label line $i in $path: expected 5 values, got $(length(r))")
        boxes[1, i] = r[1] + 1 # 0-based class -> 1-based
        boxes[2:5, i] .= r[2:5]
    end
    return boxes
end

"""
    load_darknet_dataset(imagedir, labeldir=imagedir)

Build a vector of `TrainSample`s from a darknet-style dataset: image files in
`imagedir` paired with same-basename `.txt` label files in `labeldir`.
Images without a label file are skipped. The samples hold file paths, so an
`image_loader` (e.g. `FileIO.load`) must be passed to `train!`.
"""
function load_darknet_dataset(imagedir::AbstractString, labeldir::AbstractString=imagedir)
    samples = TrainSample{String}[]
    for f in sort(readdir(imagedir))
        lowercase(splitext(f)[2]) in (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif") || continue
        lbl = joinpath(labeldir, splitext(f)[1] * ".txt")
        isfile(lbl) || continue
        push!(samples, TrainSample(joinpath(imagedir, f), load_darknet_labels(lbl)))
    end
    isempty(samples) && @warn "No image/label pairs found" imagedir labeldir
    return samples
end

"""
    letterbox_boxes(boxes, scaled_padding)

Transform normalized image-relative boxes (5×N, `[class, cx, cy, w, h]`
columns) to model-input-relative coordinates, applying the same aspect-
preserving letterbox transform that `prepare_image` applied to the image.
`scaled_padding` is the padding vector returned by `prepare_image`.
"""
function letterbox_boxes(boxes::AbstractMatrix, scaled_padding)
    out = Matrix{Float32}(boxes)
    sx = Float32(1 - scaled_padding[1] - scaled_padding[3])
    sy = Float32(1 - scaled_padding[2] - scaled_padding[4])
    out[2, :] .= Float32(scaled_padding[1]) .+ out[2, :] .* sx
    out[3, :] .= Float32(scaled_padding[2]) .+ out[3, :] .* sy
    out[4, :] .*= sx
    out[5, :] .*= sy
    return out
end

function _prepare_train_image(img::AbstractArray{<:ImageCore.Colorant}, sz::Tuple)
    src_size = size(img)[[2, 1]]
    kern = resizekern(src_size, sizethatfits(src_size, sz))
    return prepare_image(img, sz, kern; use_gpu=false)
end
_prepare_train_image(img::AbstractArray{<:Real}, sz::Tuple) = prepare_image(Float32.(img), sz; use_gpu=false)

"""
    make_training_batch(model, samples; image_loader=nothing)

Letterbox `samples` (a vector of `TrainSample`) into a `(W, H, C, N)`
`Float32` input batch for `model`, and transform each sample's boxes to
model-input coordinates. Returns `(x, boxes_batch)` where `boxes_batch` is a
vector of 5×N box matrices, one per sample.
"""
function make_training_batch(model::AbstractModel, samples; image_loader=nothing)
    W, H, C, _ = get_input_size(model)
    n = length(samples)
    x = zeros(Float32, W, H, C, n)
    boxes_batch = Vector{Matrix{Float32}}(undef, n)
    for (i, s) in enumerate(samples)
        img = if s.image isa AbstractString
            image_loader === nothing && throw(ArgumentError("sample images are file paths; pass an `image_loader` function (e.g. `using FileIO` and `image_loader=FileIO.load`)"))
            image_loader(s.image)
        else
            s.image
        end
        arr, pad = _prepare_train_image(img, (W, H, C))
        x[:, :, :, i] .= arr
        boxes_batch[i] = letterbox_boxes(s.boxes, pad)
    end
    return x, boxes_batch
end
