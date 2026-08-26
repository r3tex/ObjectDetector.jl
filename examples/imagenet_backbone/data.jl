# Dataset plumbing around ImageNetDataset.jl.

using Random
using ImageNetDataset
using ImageNetDataset: ImageNet, CenterCropNormalize, RandomCropNormalize
using Flux: onehotbatch

"""
    imagenet(root, split; transform)

Full ILSVRC-2012, laid out the way ImageNetDataset.jl's installation docs
describe: `root/train/<wnid>/*.JPEG`, `root/val/<wnid>/*.JPEG` and
`root/devkit/data/meta.mat`.
"""
function imagenet(root::AbstractString, split::Symbol; transform = CenterCropNormalize())
    return ImageNet(split; dir = root, transform)
end

# Imagenette's ten WNIDs. Only used to print readable class names.
const IMAGENETTE_CLASSES = Dict{String, Vector{String}}(
    "n01440764" => ["tench", "Tinca tinca"],
    "n02102040" => ["English springer", "English springer spaniel"],
    "n02979186" => ["cassette player"],
    "n03000684" => ["chain saw", "chainsaw"],
    "n03028079" => ["church", "church building"],
    "n03394916" => ["French horn", "horn"],
    "n03417042" => ["garbage truck", "dustcart"],
    "n03425413" => ["gas pump", "petrol pump"],
    "n03445777" => ["golf ball"],
    "n03888257" => ["parachute", "chute"],
)

"""
    imagenet_subset(root, split; transform, classnames)

Same directory layout, but for a subset such as Imagenette that has fewer than
1000 classes and ships no devkit.

`ImageNet(split; dir)` reads the devkit `meta.mat` and asserts the exact
ILSVRC-2012 file count, so a subset has to build the struct directly. Labels are
assigned by sorted WNID, which is what ImageNetDataset does for the full set too,
so the two paths stay consistent. Everything downstream - indexing, the
transforms, `convert2image`, `class` - is unchanged.
"""
function imagenet_subset(
        root::AbstractString, split::Symbol;
        transform = CenterCropNormalize(),
        classnames::AbstractDict = IMAGENETTE_CLASSES,
    )
    dir = joinpath(root, String(split))
    paths = ImageNetDataset.get_file_paths(dir)
    isempty(paths) && error("no .JPEG files under $dir")
    imagewnids = ImageNetDataset.path_to_wnid.(paths)
    wnids = sort!(unique(imagewnids))
    wnid_to_label = Dict(wnids .=> eachindex(wnids))
    metadata = Dict{String, Any}(
        "class_WNIDs" => wnids,
        "class_names" => [get(classnames, w, [w]) for w in wnids],
        "class_description" => [join(get(classnames, w, [w]), ", ") for w in wnids],
        "wnid_to_label" => wnid_to_label,
    )
    targets = [wnid_to_label[w] for w in imagewnids]
    return ImageNet(split, transform, paths, targets, metadata)
end

nclasses(ds::ImageNet) = length(ds.metadata["class_WNIDs"])

"""
    makebatch(ds, idxs, ncls) -> (X, Y)

Assemble one batch. JPEG decode plus the crop/normalize transform dominates
loading, so the observations of a batch are decoded across all threads.
"""
function makebatch(ds::ImageNet, idxs::AbstractVector{Int}, ncls::Int)
    # The first observation is decoded up front to size the batch, then kept.
    features = ds[first(idxs)].features
    w, h, c = size(features)
    X = Array{Float32, 4}(undef, w, h, c, length(idxs))
    X[:, :, :, 1] .= features
    @sync for k in 2:lastindex(idxs)
        Threads.@spawn X[:, :, :, k] .= ds[idxs[k]].features
    end
    return X, onehotbatch(ds.targets[idxs], 1:ncls)
end

"""
    batches(ds, ncls; batchsize, shuffle, rng, prefetch)

Channel of `(X, Y)` batches. The channel runs its producer on its own task, so
decoding the next batches overlaps the current training step.
"""
function batches(
        ds::ImageNet, ncls::Int = nclasses(ds);
        batchsize::Int, shuffle::Bool = true,
        rng::AbstractRNG = Random.default_rng(), prefetch::Int = 2,
    )
    idxs = collect(1:length(ds))
    shuffle && Random.shuffle!(rng, idxs)
    return Channel{Any}(prefetch; spawn = true) do ch
        for chunk in Iterators.partition(idxs, batchsize)
            put!(ch, makebatch(ds, collect(chunk), ncls))
        end
    end
end

nbatches(ds::ImageNet, batchsize::Int) = cld(length(ds), batchsize)
