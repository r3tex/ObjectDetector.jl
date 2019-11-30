"""
    emptybatch(model::T) where {T<:Model}

Create an empty batched input array on the GPU if available.
"""
function emptybatch(model::T) where {T<:Model}
    modelInputSize = getModelInputSize(model)
    gpu(zeros(Float32, modelInputSize...))
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
    drawBoxes(img::Array, model::YOLO.yolo, padding::Array, results)
    drawBoxes!(img::Array, model::YOLO.yolo, padding::Array, results)

Draw boxes on image for each BBOX result.
"""
drawBoxes(img::Array, model::YOLO.yolo, padding::Array, results; transpose=true) = drawBoxes!(copy(img), model, padding, results, transpose=transpose)
function drawBoxes!(img::Array, model::YOLO.yolo, padding::Array, results; transpose=true)
    imgratio = size(img,2) / size(img,1)
    if transpose
        modelratio = model.cfg[:width] / model.cfg[:height]
        x1i, y1i, x2i, y2i = [1, 2, 3, 4]
        if modelratio > imgratio
            h, w = size(img,1) .* (1, modelratio)
        else
            h, w = size(img,2) ./ (modelratio, 1)
        end
    else
        modelratio = model.cfg[:height] / model.cfg[:width]
        x1i, y1i, x2i, y2i = [2, 1, 4, 3]
        if modelratio < imgratio
            h, w = size(img,1) ./ (modelratio, 1)
        else
            h, w = size(img,2) .* (1, modelratio)
        end
    end
    @show h, w
    @show padding
    @show padding[[x1i, y1i, x2i, y2i]]

    p1 = Point(0, 0)
    p2 = Point(round(Int, w-((padding[x1i]+padding[x2i])*w)), round(Int, h-((padding[y1i]+padding[y2i])*h)))
    draw!(img, LineSegment(p1, p2), zero(eltype(img)))
    length(results) == 0 && return img
    for i in 1:size(results,2)
        bbox = results[1:4, i] .- padding
        class = results[end-1, i]
        conf = results[5,i]
        p = Point(round(Int, bbox[x1i]*w), round(Int, bbox[y1i]*h))
        q = Point(round(Int, bbox[x2i]*w), round(Int, bbox[y1i]*h))
        r = Point(round(Int, bbox[x1i]*w), round(Int, bbox[y2i]*h))
        s = Point(round(Int, bbox[x2i]*w), round(Int, bbox[y2i]*h))
        draw!(img, Polygon([p,q,s,r]), zero(eltype(img)))
    end
    return img
end


"""
    benchmark(;select = [1,3,4,5,6], reverseAfter:Bool=false)

Convenient benchmarking
"""
function benchmark(;select = [1,3,4,5,6], reverseAfter::Bool = false)
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
    IMG = rand(RGB,416,416)

    header = ["Model" "loaded?" "load time (s)" "ran?" "run time (s)" "run time (fps)"]
    table = Array{Any}(undef, length(pretrained_list), 6)
    for (i, pretrained) in pairs(pretrained_list)
        modelname = string(pretrained)
        @info "Loading and running $modelname"
        table[i,:] = [modelname false "-" "-" "-" "-"]

        t_load = @elapsed begin
            mod = pretrained(silent=true)
        end
        table[i, 2] = true
        table[i, 3] = round(t_load, digits=3)

        batch = emptybatch(mod)
        batch[:,:,:,1] .= gpu(resizePadImage(IMG, mod))

        res = mod(batch) #run once
        t_run = @belapsed $mod($batch);
        table[i, 4] = true
        table[i, 5] = round(t_run, digits=4)
        table[i, 6] = round(1/t_run, digits=1)

        mod = nothing
        batch = nothing
        GC.gc()
    end
    pretty_table(table, header)
end
