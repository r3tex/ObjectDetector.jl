"""
    emptybatch(model::T) where {T<:AbstractModel}

Create an empty batched input array on the GPU if available.
"""
function emptybatch(model::T) where {T<:AbstractModel}
    modelInputSize = get_input_size(model)
    batch = zeros(Float32, modelInputSize...)
    if uses_gpu(model)
        gpu(batch)
    else
        batch
    end
end

# training entry points reach through the allocator wrapper to the raw model,
# whose conv weight arrays are shared with the wrapped copy
train!(wm::AllocWrappedModel, data; kw...) = train!(wm.model, data; kw...)
save_weights(wm::AllocWrappedModel, path::AbstractString) = save_weights(wm.model, path)
make_training_batch(wm::AllocWrappedModel, samples; kw...) = make_training_batch(wm.model, samples; kw...)

function gen_class_colors(model::YOLO.Yolo)
    classes = get_cfg(model)[:output][1][:classes]
    seed = [RGB{N0f8}(0,0,0), RGB{N0f8}(1,1,1)]
    return Colors.distinguishable_colors(classes, seed; dropseed=true)
end

function _promote_to_n0f8(img)
    if eltype(img) <: Union{RGB{N0f8}, RGBA{N0f8}}
        return img
    end
    if eltype(img) <: RGBA
        return RGBA{N0f8}.(img) # e.g. RGBA{Float32} → RGBA{N0f8}
    elseif eltype(img) <: RGB
        return RGB{N0f8}.(img) # e.g. RGB{Float32} → RGB{N0f8}
    elseif eltype(img) <: Gray
        return RGB{N0f8}.(img) # Gray{T} → RGB{N0f8}
    end

    if ndims(img) != 3
        error("Unsupported image array: expected 3D channel array or colourant matrix.")
    end

    nchan = size(img, 3)
    if nchan == 4
        # (h,w,4) → colour dimension first → colour view
        return colorview(RGBA{N0f8}, permutedims(img, (3,1,2)))
    elseif nchan == 3
        return colorview(RGB{N0f8},  permutedims(img, (3,1,2)))
    else
        error("Unsupported number of channels ($nchan). Expecting 3 (RGB) or 4 (RGBA).")
    end
end

# Map normalized model-space bbox coordinates to image pixel scale, accounting
# for the transpose between julia (column-major) and darknet (row-major) layouts
function _box_geometry(img, model, transpose)
    imgratio = size(img,2) / size(img,1)
    if transpose
        modelratio = get_cfg(model)[:width] / get_cfg(model)[:height]
        idxs = (1, 2, 3, 4)
    else
        modelratio = get_cfg(model)[:height] / get_cfg(model)[:width]
        idxs = (2, 1, 4, 3)
    end
    if modelratio > imgratio
        h, w = size(img,1) .* (1, modelratio)
    else
        h, w = size(img,2) ./ (modelratio, 1)
    end
    return w, h, idxs
end

"""
    draw_boxes(img::Array, model::YOLO.Yolo, padding::Array, results)
    draw_boxes!(img::Array, model::YOLO.Yolo, padding::Array, results)

Draw class-colored boxes with conf labels on image for each BBOX result.
With `draw_boxes!` if `img` is not a color image only boxes are drawn in black.
"""
function draw_boxes(img::AbstractArray, model::YOLO.Yolo, padding::AbstractArray, results; kwargs...)
    imgc = _promote_to_n0f8(img)
    return draw_boxes!(copy(imgc), model, padding, results; kwargs...)
end
function draw_boxes(img::Union{Matrix{RGBA{N0f8}}, Matrix{RGB{N0f8}}}, model::YOLO.Yolo, padding::AbstractArray, results; kwargs...)
    return draw_boxes!(copy(img), model, padding, results; kwargs...)
end
function draw_boxes!(img::Union{Matrix{RGBA{N0f8}},Matrix{RGB{N0f8}}}, model::YOLO.Yolo, padding::AbstractArray, results;
    transpose=true, fontsize=12, opacity=0.8, label_colors = nothing, kwargs...)

    if label_colors === nothing
        label_colors = gen_class_colors(model)
    end

    w, h, (x1i, y1i, x2i, y2i) = _box_geometry(img, model, transpose)
    length(results) == 0 && return img

    img_rgb24 = similar(img, RGB24)
    img_rgb24 .= RGB24.(img)
    surf = CairoImageSurface(img_rgb24)
    ctx  = CairoContext(surf)
    Cairo.set_matrix(ctx, Cairo.CairoMatrix(0, 1, 1, 0, 0, 0))

    for i in axes(results,2)
        # extract and scale bbox
        bbox = results[1:4, i] .- padding
        cls  = Int(results[end-1, i])
        color = label_colors[cls]
        conf = results[end-2, i]

        x1 = round(Int, bbox[x1i]*w) + 1
        y1 = round(Int, bbox[y1i]*h) + 1
        x2 = round(Int, bbox[x2i]*w)
        y2 = round(Int, bbox[y2i]*h)

        # draw the rectangle
        set_line_width(ctx, 1.0)
        rectangle(ctx, x1, y1, x2 - x1, y2 - y1)
        set_source_rgba(ctx, red(color), green(color), blue(color), opacity)
        stroke(ctx)

        # prepare label and position
        label = "$(round(Int, conf*100))"
        tx, ty = x1 + 1, y1 - 3.5  # 4px above the top‑left

        # draw the text
        set_font_face(ctx, "sans-serif $(fontsize)px")
        move_to(ctx, tx, ty)
        set_source_rgba(ctx, 0, 0, 0, opacity)
        x_bearing, y_bearing, width, height, x_advance, y_advance = text_extents(ctx, label)
        pad = 2.0
        bx = tx + x_bearing - pad
        by = ty + y_bearing - pad
        bw = width + 2*pad
        bh = height + 2*pad
        save(ctx)
        set_source_rgba(ctx, 1, 1, 1, opacity)
        rectangle(ctx, bx, by, bw, bh)
        fill(ctx)
        restore(ctx)
        set_source_rgba(ctx, 0, 0, 0, opacity)
        move_to(ctx, tx, ty)
        show_text(ctx, label)
    end

    # flush back into img and return
    finish(surf)
    img .= RGBA{N0f8}.(img_rgb24)
    return img
end

# keep this for users that want to keep drawing boxes directly into non-color type images
function draw_boxes!(img::AbstractArray, model::YOLO.Yolo, padding::AbstractArray, results; transpose=true, kwargs...)
    w, h, (x1i, y1i, x2i, y2i) = _box_geometry(img, model, transpose)
    length(results) == 0 && return img
    for i in 1:size(results,2)
        bbox = results[1:4, i] .- padding
        class = results[end-1, i]
        conf = results[end-2,i]
        p = Point(round(Int, bbox[x1i]*w)+1, round(Int, bbox[y1i]*h)+1)
        q = Point(round(Int, bbox[x2i]*w), round(Int, bbox[y1i]*h)+1)
        r = Point(round(Int, bbox[x1i]*w)+1, round(Int, bbox[y2i]*h))
        s = Point(round(Int, bbox[x2i]*w), round(Int, bbox[y2i]*h))
        pol = Polygon([p,q,s,r])
        draw!(img, pol, zero(eltype(img)))
    end
    return img
end
