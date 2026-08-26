# Read COCO's instances JSON into the package's TrainSample convention.
#
# COCO stores boxes as pixel `[x, y, w, h]` from the top-left corner, with
# category ids running 1-90 with gaps. ObjectDetector wants normalized
# `[class, cx, cy, w, h]` with 1-based contiguous classes, so both have to be
# converted.

using JSON3
using ObjectDetector: TrainSample

"""
    coco_classes(annotationfile) -> (ids, names)

The dataset's category ids in ascending order, and their COCO names.

Sorted id order is darknet's class order, which is what `coco.names` and the
pretrained weights use, so mapping by sorted id keeps a model trained here
compatible with them. Mapping by name would not: darknet renames several
classes ("motorbike" for COCO's "motorcycle", "aeroplane" for "airplane",
"sofa" for "couch", "tvmonitor" for "tv").
"""
function coco_classes(annotationfile::AbstractString)
    cats = JSON3.read(read(annotationfile), Dict{String, Any})["categories"]
    order = sortperm([c["id"] for c in cats])
    return [cats[i]["id"] for i in order], [String(cats[i]["name"]) for i in order]
end

"""
    coco_samples(imagedir, annotationfile; classes = nothing, limit = 0)

`TrainSample`s holding image paths, one per image that has at least one usable
box. Pass them to `train!` with `image_loader = FileIO.load`.

`classes` restricts the dataset to a subset of COCO names, remapped to
`1:length(classes)` in the order given, which is what a model built with fewer
`classes` in its cfg expects. `limit` caps the number of images kept, for a
quicker run.

Crowd regions are dropped: they mark an unspecified number of objects with a
single box, so they are neither one object nor background.
"""
function coco_samples(
        imagedir::AbstractString, annotationfile::AbstractString;
        classes::Union{Nothing, AbstractVector{<:AbstractString}} = nothing,
        limit::Int = 0,
    )
    ann = JSON3.read(read(annotationfile), Dict{String, Any})
    ids, names = coco_classes(annotationfile)

    if classes === nothing
        label = Dict(id => i for (i, id) in enumerate(ids))
    else
        wanted = Dict(String(n) => i for (i, n) in enumerate(classes))
        for n in classes
            n in names || error("$n is not a COCO class; see coco_classes(annotationfile)")
        end
        label = Dict(id => wanted[n] for (id, n) in zip(ids, names) if haskey(wanted, n))
    end

    # image id => (path, width, height)
    info = Dict(im["id"] => (joinpath(imagedir, im["file_name"]), im["width"], im["height"])
        for im in ann["images"])

    boxes = Dict{Any, Vector{NTuple{5, Float32}}}()
    for a in ann["annotations"]
        get(a, "iscrowd", 0) == 1 && continue
        haskey(label, a["category_id"]) || continue
        x, y, w, h = a["bbox"]
        (w > 0 && h > 0) || continue
        _, iw, ih = info[a["image_id"]]
        push!(get!(boxes, a["image_id"], NTuple{5, Float32}[]),
            (label[a["category_id"]], (x + w / 2) / iw, (y + h / 2) / ih, w / iw, h / ih))
    end

    samples = TrainSample{String}[]
    for imageid in sort!(collect(keys(boxes)))
        bs = boxes[imageid]
        m = Matrix{Float32}(undef, 5, length(bs))
        for (i, b) in enumerate(bs)
            m[:, i] .= b
        end
        push!(samples, TrainSample(first(info[imageid]), m))
        limit > 0 && length(samples) >= limit && break
    end
    return samples
end

nboxes(samples) = sum(s -> size(s.boxes, 2), samples; init = 0)
