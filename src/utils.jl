"""
    emptybatch(model::T) where {T<:Model}

Create an empty batched input array on the GPU if available.
"""
function emptybatch(model::T) where {T<:Model}
    modelInputSize = getModelInputSize(model)
    gpu(Array{Float32}(undef, modelInputSize...))
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
    drawBoxes(img::Array, results)

Draw boxes on image for each BBOX result.
"""
function drawBoxes(img::Array, results)
    w = size(img,1)
    h = size(img,2)
    imgCopy = copy(img)
    size(results) == (0,) && return imgCopy
    for i in 1:size(results,2)
        bbox = results[1:4,i]
        class = results[end-1,i]
        conf = results[end-2,i]
        p = Point(round(Int, bbox[1]*w), round(Int, bbox[2]*h))
        q = Point(round(Int, bbox[1]*w), round(Int, bbox[4]*h))
        r = Point(round(Int, bbox[3]*w), round(Int, bbox[2]*h))
        s = Point(round(Int, bbox[3]*w), round(Int, bbox[4]*h))
        draw!(imgCopy, Polygon([p,q,s,r]), zero(eltype(imgCopy)))
    end
    return imgCopy
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
                        YOLO.v3_608_spp_COCO
                        ][select]
    reverseAfter && (pretrained_list = vcat(pretrained_list, reverse(pretrained_list)))
    IMG = rand(RGB,416,416)

    header = ["Model" "loaded?" "load time (s)" "ran?" "run time (s)" "run time (fps)" "objects detected"]
    table = Array{Any}(undef, length(pretrained_list), 7)
    for (i, pretrained) in pairs(pretrained_list)
        modelname = string(pretrained)
        @info "Loading and running $modelname"
        table[i,:] = [modelname false "-" "-" "-" "-" "-"]

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
        table[i, 7] = size(res,2)

        mod = nothing
        batch = nothing
        GC.gc()
    end
    pretty_table(table, header)
end
