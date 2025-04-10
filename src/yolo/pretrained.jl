const YOLO_MODELS = Dict{String, Any}(
    "v2_COCO"         => () -> (joinpath(models_dir(), "yolov2-608.cfg"),   joinpath(artifact"yolov2-COCO",        "yolov2-COCO.weights")),
    "v2_tiny_COCO"    => () -> (joinpath(models_dir(), "yolov2-tiny.cfg"),  joinpath(artifact"yolov2-tiny-COCO",   "yolov2-tiny-COCO.weights")),
    "v3_COCO"         => () -> (joinpath(models_dir(), "yolov3-416.cfg"),   joinpath(artifact"yolov3-COCO",        "yolov3-COCO.weights")),
    "v3_SPP_COCO"     => () -> (joinpath(models_dir(), "yolov3-spp.cfg"),   joinpath(artifact"yolov3-spp-COCO",    "yolov3-spp-COCO.weights")),
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

# -- YOLOv2
v2_COCO(;kwargs...)           = yolo_model("v2_COCO";        w=608, h=608, kwargs...)
v2_608_COCO(;kwargs...)       = v2_COCO(;                    w=608, h=608, kwargs...)

v2_tiny_COCO(;kwargs...)      = yolo_model("v2_tiny_COCO";   w=416, h=416, kwargs...)
v2_tiny_416_COCO(;kwargs...)  = v2_tiny_COCO(;               w=416, h=416, kwargs...)

# -- YOLOv3
v3_COCO(;kwargs...)           = yolo_model("v3_COCO";                     kwargs...)
v3_320_COCO(;kwargs...)       = v3_COCO(;                    w=320, h=320, kwargs...)
v3_416_COCO(;kwargs...)       = v3_COCO(;                    w=416, h=416, kwargs...)
v3_608_COCO(;kwargs...)       = v3_COCO(;                    w=608, h=608, kwargs...)

v3_SPP_COCO(;kwargs...)       = yolo_model("v3_SPP_COCO";    w=608, h=608, kwargs...)
v3_spp_608_COCO(;kwargs...)   = v3_SPP_COCO(;                w=608, h=608, kwargs...)

v3_tiny_COCO(;kwargs...)      = yolo_model("v3_tiny_COCO";   w=416, h=416, kwargs...)
v3_tiny_416_COCO(;kwargs...)  = v3_tiny_COCO(;               w=416, h=416, kwargs...)

# -- YOLOv4
v4_COCO(;kwargs...)           = yolo_model("v4_COCO";                     kwargs...)
v4_320_COCO(;kwargs...)       = v4_COCO(;                    w=320, h=320, kwargs...)
v4_416_COCO(;kwargs...)       = v4_COCO(;                    w=416, h=416, kwargs...)
v4_608_COCO(;kwargs...)       = v4_COCO(;                    w=608, h=608, kwargs...)

v4_tiny_COCO(;kwargs...)      = yolo_model("v4_tiny_COCO";                kwargs...)
v4_tiny_416_COCO(;kwargs...)  = v4_tiny_COCO(;               w=416, h=416, kwargs...)
v4_tiny_608_COCO(;kwargs...)  = v4_tiny_COCO(;               w=608, h=608, kwargs...)

# -- YOLOv7
v7_COCO(;kwargs...)           = yolo_model("v7_COCO";                     kwargs...)
v7_320_COCO(;kwargs...)       = v7_COCO(;                    w=320, h=320, kwargs...)
v7_416_COCO(;kwargs...)       = v7_COCO(;                    w=416, h=416, kwargs...)
v7_608_COCO(;kwargs...)       = v7_COCO(;                    w=608, h=608, kwargs...)

v7_tiny_COCO(;kwargs...)      = yolo_model("v7_tiny_COCO";                kwargs...)
v7_tiny_416_COCO(;kwargs...)  = v7_tiny_COCO(;               w=416, h=416, kwargs...)
v7_tiny_608_COCO(;kwargs...)  = v7_tiny_COCO(;               w=608, h=608, kwargs...)
