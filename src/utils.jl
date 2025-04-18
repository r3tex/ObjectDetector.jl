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

"""
    flipdict(dict::Dict)

Flip the key=>value pair for each entry in a dict.
"""
flipdict(dict::Dict) = Dict(map(x->(dict[x],x),collect(keys(dict))))

"""
    createcountdict(dict::Dict)

Create a dict copy of namesdict, for counting the occurances of each named object.
"""
createcountdict(dict::Dict) = Dict(map(x->(x,0),collect(keys(dict))))

function gen_class_colors(model::YOLO.Yolo)
    classes = get_cfg(model)[:output][1][:classes]
    seed = [RGB{N0f8}(0,0,0), RGB{N0f8}(1,1,1)]
    return Colors.distinguishable_colors(classes, seed; dropseed=true)
end
"""
    draw_boxes(img::Array, model::YOLO.Yolo, padding::Array, results)
    draw_boxes!(img::Array, model::YOLO.Yolo, padding::Array, results)

Draw class-colored boxes with conf labels on image for each BBOX result.
With `draw_boxes!` if `img` is not a color image only boxes are drawn in black.
"""
function draw_boxes(img::AbstractArray, model::YOLO.Yolo, padding::AbstractArray, results; kwargs...)
    if size(img, 3) == 4
        imgc = colorview(RGBA{N0f8}, img[:, :, 1:3])
    elseif eltype(img) <: Gray{N0f8}
        imgc = RGB{N0f8}.(img)
    else
        imgc = colorview(RGBA{N0f8}, img)
    end
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

    imgratio = size(img,2) / size(img,1)
    if transpose
        modelratio = get_cfg(model)[:width]  / get_cfg(model)[:height]
        x1i,y1i,x2i,y2i = 1,2,3,4
    else
        modelratio = get_cfg(model)[:height] / get_cfg(model)[:width]
        x1i,y1i,x2i,y2i = 2,1,4,3
    end
    if modelratio > imgratio
        h, w = size(img,1) .* (1, modelratio)
    else
        h, w = size(img,2) ./ (modelratio, 1)
    end
    length(results) == 0 && return img

    img_rgb24 = similar(img, RGB24)
    img_rgb24 .= RGB24.(img)
    surf = CairoImageSurface(img_rgb24)
    ctx  = CairoContext(surf)
    Cairo.set_matrix(ctx, Cairo.CairoMatrix(0, 1, 1, 0, 0, 0))

    for i in axes(results,2)
        # extract and scale bbox
        bbox = results[1:4, i] .- padding
        cls  = Int(results[end-1, i]) + 1 # zero-based
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
    imgratio = size(img,2) / size(img,1)
    if transpose
        modelratio = get_cfg(model)[:width] / get_cfg(model)[:height]
        x1i, y1i, x2i, y2i = [1, 2, 3, 4]
    else
        modelratio = get_cfg(model)[:height] / get_cfg(model)[:width]
        x1i, y1i, x2i, y2i = [2, 1, 4, 3]
    end
    if modelratio > imgratio
        h, w = size(img,1) .* (1, modelratio)
    else
        h, w = size(img,2) ./ (modelratio, 1)
    end

    # p1 = Point(1, 1)
    # p2 = Point(round(Int, w-((padding[x1i]+padding[x2i])*w)), round(Int, h-((padding[y1i]+padding[y2i])*h)))
    # draw!(img, LineSegment(p1, p2), zero(eltype(img)))
    length(results) == 0 && return img
    for i in 1:size(results,2)
        bbox = results[1:4, i] .- padding
        class = results[end-1, i]
        conf = results[5,i]
        p = Point(round(Int, bbox[x1i]*w)+1, round(Int, bbox[y1i]*h)+1)
        q = Point(round(Int, bbox[x2i]*w), round(Int, bbox[y1i]*h)+1)
        r = Point(round(Int, bbox[x1i]*w)+1, round(Int, bbox[y2i]*h))
        s = Point(round(Int, bbox[x2i]*w), round(Int, bbox[y2i]*h))
        pol = Polygon([p,q,s,r])
        draw!(img, pol, zero(eltype(img)))
    end
    return img
end

"""
    benchmark(;select = [1,2,6], reverseAfter:Bool=false)

Convenient benchmarking
"""
function benchmark(;select = [1,2,6], reverseAfter::Bool = false, img = rand(RGB,416,416), verbose=true, kw...)
    pretrained_list = [
                        YOLO.v2_tiny_416_COCO,
                        YOLO.v3_tiny_416_COCO,
                        YOLO.v4_tiny_416_COCO,
                        YOLO.v7_tiny_416_COCO,
                        YOLO.v2_416_COCO, # broken weights?
                        YOLO.v3_416_COCO,
                        YOLO.v3_spp_416_COCO,
                        YOLO.v4_416_COCO,
                        YOLO.v7_416_COCO,
                        ][select]
    reverseAfter && (pretrained_list = vcat(pretrained_list, reverse(pretrained_list)))


    header = ["Model", "loaded?", "load time (s)", "#results", "run time (s)", "run time (fps)", "allocations"]
    table = Array{Any}(undef, length(pretrained_list), 7)
    for (i, pretrained) in pairs(pretrained_list)
        modelname = string(pretrained)
        verbose && @info "Loading and running $modelname"
        table[i,:] = [modelname false "-" "-" "-" "-" "-"]

        loaded = true
        t_load = @elapsed begin
            mod = try
                pretrained(;silent=true, kw...)
            catch ex
                loaded = false
                @warn "Failed to load $modelname: $ex"
            end
        end
        table[i, 2] = loaded
        loaded || continue

        table[i, 3] = round(t_load, digits=3)

        batch = emptybatch(mod)
        batch[:,:,:,1], padding = prepare_image(img, mod)

        res = mod(batch; detect_thresh=0.0, overlap_thresh=1.0) #run once
        t_run = @belapsed $mod($batch; detect_thresh=0.0, overlap_thresh=1.0);
        t_allocs = @allocated mod(batch; detect_thresh=0.0, overlap_thresh=1.0)
        table[i, 4] = size(res, 2)
        table[i, 5] = round(t_run, digits=4)
        table[i, 6] = round(1/t_run, digits=1)
        table[i, 7] = Base.format_bytes(t_allocs)

        mod = nothing
        batch = nothing
        GC.gc()
    end
    pretty_table(table, header = header)
end
