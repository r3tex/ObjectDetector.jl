const YOLO_MODELS = Dict{String, Any}(
    "v2_COCO"         => () -> (joinpath(models_dir(), "yolov2.cfg"),       joinpath(artifact"yolov2-COCO",        "yolov2-COCO.weights")),
    "v2_tiny_COCO"    => () -> (joinpath(models_dir(), "yolov2-tiny.cfg"),  joinpath(artifact"yolov2-tiny-COCO",   "yolov2-tiny-COCO.weights")),
    "v3_COCO"         => () -> (joinpath(models_dir(), "yolov3.cfg"),       joinpath(artifact"yolov3-COCO",        "yolov3-COCO.weights")),
    "v3_spp_COCO"     => () -> (joinpath(models_dir(), "yolov3-spp.cfg"),   joinpath(artifact"yolov3-spp-COCO",    "yolov3-spp-COCO.weights")),
    "v3_tiny_COCO"    => () -> (joinpath(models_dir(), "yolov3-tiny.cfg"),  joinpath(artifact"yolov3-tiny-COCO",   "yolov3-tiny-COCO.weights")),
    "v4_COCO"         => () -> (joinpath(models_dir(), "yolov4.cfg"),       joinpath(artifact"yolov4-COCO",        "yolov4-COCO.weights")),
    "v4_tiny_COCO"    => () -> (joinpath(models_dir(), "yolov4-tiny.cfg"),  joinpath(artifact"yolov4-tiny-COCO",   "yolov4-tiny-COCO.weights")),
    "v7_COCO"         => () -> (joinpath(models_dir(), "yolov7.cfg"),       joinpath(artifact"yolov7-COCO",        "yolov7-COCO.weights")),
    "v7_tiny_COCO"    => () -> (joinpath(models_dir(), "yolov7-tiny.cfg"),  joinpath(artifact"yolov7-tiny-COCO",   "yolov7-tiny-COCO.weights")),
)

function yolo_model(modelkey::String; batch=1, silent=false, w=416, h=416, dummy::Bool=false, cfgchanges=nothing, kwargs...)
    cfgfile, weightsfile = YOLO_MODELS[modelkey]()
    weightsfile = dummy ? nothing : weightsfile
    cfgchanges === nothing && (cfgchanges = [(:net, 1, :width, w), (:net, 1, :height, h)])
    Yolo(cfgfile, weightsfile, batch; silent, cfgchanges, kwargs...)
end

const sizes = (320, 416, 608)

for model_name in keys(YOLO_MODELS)
    @eval $(Symbol(model_name))(; kwargs...) = yolo_model($model_name; kwargs...)
    version = split(model_name, "_COCO")[1]
    for sz in sizes
        func_name = Symbol("$(version)_$(sz)_COCO")
        @eval $func_name(; kwargs...) = yolo_model($model_name; w=$sz, h=$sz, kwargs...)
    end
end
