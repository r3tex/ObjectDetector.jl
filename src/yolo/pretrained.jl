const YOLO_MODELS = Dict{String, Any}(
    "v2_COCO"             => () -> (joinpath(models_dir(), "yolov2.cfg"),             joinpath(artifact"yolov2-COCO",             "yolov2-COCO.weights")),
    "v2_tiny_COCO"        => () -> (joinpath(models_dir(), "yolov2-tiny.cfg"),        joinpath(artifact"yolov2-tiny-COCO",        "yolov2-tiny-COCO.weights")),
    "v3_COCO"             => () -> (joinpath(models_dir(), "yolov3.cfg"),             joinpath(artifact"yolov3-COCO",             "yolov3-COCO.weights")),
    "v3_spp_COCO"         => () -> (joinpath(models_dir(), "yolov3-spp.cfg"),         joinpath(artifact"yolov3-spp-COCO",         "yolov3-spp-COCO.weights")),
    "v3_tiny_COCO"        => () -> (joinpath(models_dir(), "yolov3-tiny.cfg"),        joinpath(artifact"yolov3-tiny-COCO",        "yolov3-tiny-COCO.weights")),
    "v4_COCO"             => () -> (joinpath(models_dir(), "yolov4.cfg"),             joinpath(artifact"yolov4-COCO",             "yolov4-COCO.weights")),
    "v4_tiny_COCO"        => () -> (joinpath(models_dir(), "yolov4-tiny.cfg"),        joinpath(artifact"yolov4-tiny-COCO",        "yolov4-tiny-COCO.weights")),
    "v4_csp_COCO"         => () -> (joinpath(models_dir(), "yolov4-csp.cfg"),         joinpath(artifact"yolov4-csp-COCO",         "yolov4-csp-COCO.weights")),
    "v4_csp_x_swish_COCO" => () -> (joinpath(models_dir(), "yolov4-csp-x-swish.cfg"), joinpath(artifact"yolov4-csp-x-swish-COCO", "yolov4-csp-x-swish-COCO.weights")),
    "v4x_mish_COCO"       => () -> (joinpath(models_dir(), "yolov4x-mish.cfg"),       joinpath(artifact"yolov4x-mish-COCO",       "yolov4x-mish-COCO.weights")),
    "v4_p5_COCO"          => () -> (joinpath(models_dir(), "yolov4-p5.cfg"),          joinpath(artifact"yolov4-p5-COCO",          "yolov4-p5-COCO.weights")),
    "v4_p6_COCO"          => () -> (joinpath(models_dir(), "yolov4-p6.cfg"),          joinpath(artifact"yolov4-p6-COCO",          "yolov4-p6-COCO.weights")),
    "v7_COCO"             => () -> (joinpath(models_dir(), "yolov7.cfg"),             joinpath(artifact"yolov7-COCO",             "yolov7-COCO.weights")),
    "v7_tiny_COCO"        => () -> (joinpath(models_dir(), "yolov7-tiny.cfg"),        joinpath(artifact"yolov7-tiny-COCO",        "yolov7-tiny-COCO.weights")),
    "v7x_COCO"            => () -> (joinpath(models_dir(), "yolov7x.cfg"),            joinpath(artifact"yolov7x-COCO",            "yolov7x-COCO.weights")),
)

# Native (training) input size, used as the default when no size is given.
# Models not listed default to 416. Note that all sizes must be an integer
# multiple of the model's largest stride: 64 for v4_p6, otherwise 32.
const MODEL_DEFAULT_SIZE = Dict{String, Int}(
    "v4_csp_COCO"         => 512,
    "v4_csp_x_swish_COCO" => 640,
    "v4x_mish_COCO"       => 640,
    "v4_p5_COCO"          => 896,
    "v4_p6_COCO"          => 1280,
    "v7x_COCO"            => 640,
)

function yolo_model(modelkey::String; batch=1, silent=false, w=nothing, h=nothing, dummy::Bool=false, cfgchanges=nothing, kwargs...)
    cfgfile, weightsfile = YOLO_MODELS[modelkey]()
    weightsfile = dummy ? nothing : weightsfile
    default_size = get(MODEL_DEFAULT_SIZE, modelkey, 416)
    w = @something w default_size
    h = @something h default_size
    cfgchanges === nothing && (cfgchanges = [(:net, 1, :width, w), (:net, 1, :height, h)])
    Yolo(cfgfile, weightsfile, batch; silent, cfgchanges, kwargs...)
end

const sizes = (320, 416, 608)

# Sizes for the generated fixed-size convenience constructors, e.g. v3_416_COCO.
# Models not listed get the classic (320, 416, 608).
const MODEL_CONVENIENCE_SIZES = Dict{String, Tuple{Vararg{Int}}}(
    "v4_csp_COCO"         => (512, 640),
    "v4_csp_x_swish_COCO" => (512, 640),
    "v4x_mish_COCO"       => (512, 640),
    "v4_p5_COCO"          => (896,),
    "v4_p6_COCO"          => (1280,),
    "v7x_COCO"            => (416, 640),
)

for model_name in keys(YOLO_MODELS)
    @eval $(Symbol(model_name))(; kwargs...) = yolo_model($model_name; kwargs...)
    version = split(model_name, "_COCO")[1]
    for sz in get(MODEL_CONVENIENCE_SIZES, model_name, sizes)
        func_name = Symbol("$(version)_$(sz)_COCO")
        @eval $func_name(; kwargs...) = yolo_model($model_name; w=$sz, h=$sz, kwargs...)
    end
end
