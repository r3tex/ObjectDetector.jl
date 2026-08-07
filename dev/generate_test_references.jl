# Regenerate the detection references (test/resrefs.jl) and reference images
# (test/results/) that the test suite compares against.
#
# Run from the repo root with the test environment:
#
#     julia --project=test dev/generate_test_references.jl [modelname...]
#
# With no arguments every model in YOLO_MODELS is regenerated; otherwise only
# the named models are, with all other entries preserved. Darknet parity is
# asserted for every model/image combination before its references are
# written, so blessed references are known-correct rather than just "whatever
# the first run produced".
#
# Models are compared at the same sizes the test suite uses (see
# model_test_sizes in test/maintests.jl).

using ObjectDetector, FileIO, Darknet, Suppressor, ReferenceTests, Test
using OrderedCollections: OrderedDict
const cpu = ObjectDetector.YOLO.cpu

repo = dirname(@__DIR__)
include(joinpath(repo, "test", "resrefs.jl"))
RES = OrderedDict{String, Matrix{Float32}}(RES_REFS)

# Keep in sync with model_test_sizes in test/maintests.jl
test_sizes = Dict(
    "v4_csp_x_swish_COCO" => 512,
    "v4x_mish_COCO"       => 512,
    "v4_p5_COCO"          => 512,
    "v4_p6_COCO"          => 448,
)

function sized_cfgfile(cfgfile, size)
    lines = readlines(cfgfile)
    lines = map(l -> startswith(strip(l), "width") ? "width=$size" :
                     startswith(strip(l), "height") ? "height=$size" : l, lines)
    tmp = joinpath(mktempdir(), basename(cfgfile))
    write(tmp, join(lines, "\n"))
    return tmp
end

names = collect(eachline(joinpath(ObjectDetector.YOLO.models_dir(), "coco.names")))
Darknet.download_defaults()
meta = Darknet.get_metadata(joinpath(pkgdir(Darknet), "data", "coco.data"))
testimages = ["dog-cycle-car", "dog-cycle-car_nonsquare"]

models = isempty(ARGS) ? sort(collect(keys(ObjectDetector.YOLO.YOLO_MODELS))) : ARGS

for modelname in models
    cfgfile, weightsfile = ObjectDetector.YOLO.YOLO_MODELS[modelname]()
    test_size = get(test_sizes, modelname, nothing)
    darknet_cfg = test_size === nothing ? cfgfile : sized_cfgfile(cfgfile, test_size)
    net = @suppress Darknet.load_network(darknet_cfg, weightsfile, 1)
    yolomod = ObjectDetector.YOLO.yolo_model(modelname; silent=true, w=test_size, h=test_size)
    for imagename in testimages
        img = load(joinpath(repo, "test", "images", "$imagename.png"))
        resultsdir = joinpath(repo, "test", "results", imagename)
        mkpath(resultsdir)
        # remove stale reference images so @test_reference recreates them
        for suffix in ("in_padded", "out_od", "out_darknet")
            rm(joinpath(resultsdir, "$(modelname)_$(suffix).png"); force=true)
        end
        batch = emptybatch(yolomod)
        img_padded, padding = prepare_image(img, yolomod)
        batch[:,:,:,1] .= img_padded
        @test_reference joinpath(resultsdir, "$(modelname)_in_padded.png") cpu(img_padded) by=psnr_equality(35.0)

        img_d = Darknet.array_to_image(collect(PermutedDimsArray(img_padded, (3, 2, 1))))
        darkres = Darknet.detect(net, meta, img_d, thresh=0.5, nms=0.5)
        juliares = yolomod(batch, detect_thresh=0.5, overlap_thresh=0.5)

        @test_reference joinpath(resultsdir, "$(modelname)_out_od.png") draw_boxes(img, yolomod, padding, juliares) by=psnr_equality(35.0)

        darkres_xyxy = zeros(Float32, 89, length(darkres))
        img_h, img_w = size(img_padded, 1), size(img_padded, 2)
        for i in eachindex(darkres)
            d_x, d_y, d_w, d_h = darkres[i][3]
            darkres_xyxy[1:4, i] = Float32[(d_x - d_w/2) / img_w, (d_y - d_h/2) / img_h, (d_x + d_w/2) / img_w, (d_y + d_h/2) / img_h]
            darkres_xyxy[end-2, i] = Float32(darkres[i][2])
            darkres_xyxy[end-1, i] = findfirst(==(darkres[i][1]), names)
        end
        @test_reference joinpath(resultsdir, "$(modelname)_out_darknet.png") draw_boxes(img, yolomod, padding, darkres_xyxy) by=psnr_equality(35.0)

        @assert size(darkres_xyxy) == size(juliares) "size mismatch $modelname/$imagename: $(size(darkres_xyxy)) vs $(size(juliares))"
        dark_sorted = sortslices(darkres_xyxy, dims=2, by = x -> x[1])
        julia_sorted = sortslices(juliares, dims=2, by = x -> x[1])
        @assert isapprox(dark_sorted[1:4, :], julia_sorted[1:4, :]; atol=0.05) "bbox mismatch $modelname/$imagename"
        @assert isapprox(dark_sorted[end-2, :], julia_sorted[end-2, :]; atol=0.05) "conf mismatch $modelname/$imagename"
        @assert dark_sorted[end-1, :] == julia_sorted[end-1, :] "class mismatch $modelname/$imagename"
        RES["dn_$(modelname)_$(imagename)"] = dark_sorted
        RES["od_$(modelname)_$(imagename)"] = julia_sorted
        println("$modelname / $imagename: parity OK, $(size(juliares, 2)) detections"); flush(stdout)
    end
    GC.gc()
end

sort!(RES)
open(joinpath(repo, "test", "resrefs.jl"), "w") do io
    println(io, "const RES_REFS = OrderedDict{String, Matrix{Float32}}(")
    for (k, v) in RES
        if size(v, 2) < 2
            # a row per line would parse as a Vector for single-column matrices
            println(io, "    ", repr(k), " => ", repr(v), ",")
        else
            println(io, "    ", repr(k), " => Float32[")
            for r in eachrow(v)
                println(io, "        ", join(r, " "))
            end
            println(io, "    ],")
        end
    end
    println(io, ")")
end
println("resrefs.jl rewritten with $(length(RES)) entries")
