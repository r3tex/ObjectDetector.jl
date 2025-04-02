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

"""
    draw_boxes(img::Array, model::YOLO.Yolo, padding::Array, results)
    draw_boxes!(img::Array, model::YOLO.Yolo, padding::Array, results)

Draw boxes on image for each BBOX result.
"""
draw_boxes(img::AbstractArray, model::YOLO.Yolo, padding::AbstractArray, results; transpose=true) = draw_boxes!(copy(img), model, padding, results, transpose=transpose)
function draw_boxes!(img::AbstractArray, model::YOLO.Yolo, padding::AbstractArray, results; transpose=true)
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
    benchmark(;select = [1,3,4,5,6], reverseAfter:Bool=false)

Convenient benchmarking
"""
function benchmark(;select = [1,3,4,5,6], reverseAfter::Bool = false, img = rand(RGB,416,416), verbose=true, kw...)
    pretrained_list = [
                        YOLO.v2_tiny_416_COCO,
                        YOLO.v2_608_COCO,
                        YOLO.v3_tiny_416_COCO,
                        YOLO.v3_320_COCO,
                        YOLO.v3_416_COCO,
                        YOLO.v3_608_COCO,
                        YOLO.v3_spp_608_COCO
                        ][select]
    reverseAfter && (pretrained_list = vcat(pretrained_list, reverse(pretrained_list)))


    header = ["Model", "loaded?", "load time (s)", "#results", "run time (s)", "run time (fps)", "allocations"]
    table = Array{Any}(undef, length(pretrained_list), 7)
    for (i, pretrained) in pairs(pretrained_list)
        modelname = string(pretrained)
        verbose && @info "Loading and running $modelname"
        table[i,:] = [modelname false "-" "-" "-" "-" "-"]

        t_load = @elapsed begin
            mod = pretrained(;silent=true, kw...)
        end
        table[i, 2] = true
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
