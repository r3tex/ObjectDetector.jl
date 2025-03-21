## YOLOV2
function v2_COCO(;batch=1, silent=false, w=608, h=608, cfgchanges=[(:net, 1, :width, w), (:net, 1, :height, h)], kwargs...)
    Yolo(joinpath(models_dir,"yolov2-608.cfg"), joinpath(artifact"yolov2-COCO", "yolov2-COCO.weights"), batch; silent, cfgchanges, kwargs...)
end
v2_608_COCO(;kwargs...) = v2_COCO(;w=608, h=608, kwargs...)

## YOLOV2-tiny
function v2_tiny_COCO(;batch=1, silent=false, w=416, h=416, cfgchanges=[(:net, 1, :width, w), (:net, 1, :height, h)], kwargs...)
    Yolo(joinpath(models_dir,"yolov2-tiny.cfg"), joinpath(artifact"yolov2-tiny-COCO", "yolov2-tiny-COCO.weights"), batch; silent, cfgchanges, kwargs...)
end
v2_tiny_416_COCO(;kwargs...) = v2_tiny_COCO(;w=416, h=416, kwargs...)

## YOLOV3
function v3_COCO(;batch=1, silent=false, w=416, h=416, cfgchanges=[(:net, 1, :width, w), (:net, 1, :height, h)], dummy::Bool=false, kwargs...)
    weightsfile = dummy ? nothing : joinpath(artifact"yolov3-COCO", "yolov3-COCO.weights")
    Yolo(joinpath(models_dir,"yolov3-416.cfg"), weightsfile, batch; silent, cfgchanges, kwargs...)
end
v3_320_COCO(;kwargs...) = v3_COCO(;w=320, h=320, kwargs...)
v3_416_COCO(;kwargs...) = v3_COCO(;w=416, h=416, kwargs...)
v3_608_COCO(;kwargs...) = v3_COCO(;w=608, h=608, kwargs...)

## YOLOV3 SPP
function v3_SPP_COCO(;batch=1, silent=false, w=608, h=608, cfgchanges=[(:net, 1, :width, w), (:net, 1, :height, h)], kwargs...)
    Yolo(joinpath(models_dir,"yolov3-spp.cfg"), joinpath(artifact"yolov3-spp-COCO", "yolov3-spp-COCO.weights"), batch; silent, cfgchanges, kwargs...)
end
v3_spp_608_COCO(;kwargs...) = v3_SPP_COCO(;w=608, h=608, kwargs...)

## YOLOV3-tiny
function v3_tiny_COCO(;batch=1, silent=false, w=416, h=416, cfgchanges=[(:net, 1, :width, w), (:net, 1, :height, h)], kwargs...)
    Yolo(joinpath(models_dir,"yolov3-tiny.cfg"), joinpath(artifact"yolov3-tiny-COCO", "yolov3-tiny-COCO.weights"), batch; silent, cfgchanges, kwargs...)
end
v3_tiny_416_COCO(;kwargs...) = v3_tiny_COCO(;w=416, h=416, kwargs...)
