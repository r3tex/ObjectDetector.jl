## YOLOV2
function v2_COCO(;batch=1, silent=false, w=608, h=608, cfgchanges=[(:net, 1, :width, w), (:net, 1, :height, h)])
    yolo(joinpath(models_dir,"yolov2-608.cfg"), getArtifact("yolov2-COCO"), batch, silent=silent, cfgchanges=cfgchanges)
end
v2_608_COCO(;batch=1, silent=false, cfgchanges=nothing) = v2_COCO(w=608, h=608, batch=batch, silent=silent, cfgchanges=cfgchanges)

## YOLOV2-tiny
function v2_tiny_COCO(;batch=1, silent=false, cfgchanges=[(:net, 1, :width, w), (:net, 1, :height, h)], w=416, h=416)
    yolo(joinpath(models_dir,"yolov2-tiny.cfg"), getArtifact("yolov2-tiny-COCO"), batch, silent=silent, cfgchanges=cfgchanges)
end
v2_tiny_416_COCO(;batch=1, silent=false, cfgchanges=nothing) = v2_tiny_COCO(w=416, h=416, batch=batch, silent=silent, cfgchanges=cfgchanges)

## YOLOV3
function v3_COCO(;batch=1, silent=false, w=416, h=416, cfgchanges=[(:net, 1, :width, w), (:net, 1, :height, h)])
    yolo(joinpath(models_dir,"yolov3-416.cfg"), getArtifact("yolov3-COCO"), batch, silent=silent, cfgchanges=cfgchanges)
end
v3_320_COCO(;batch=1, silent=false, cfgchanges=nothing) = v3_COCO(w=320, h=320, batch=batch, silent=silent, cfgchanges=cfgchanges)
v3_416_COCO(;batch=1, silent=false, cfgchanges=nothing) = v3_COCO(w=416, h=416, batch=batch, silent=silent, cfgchanges=cfgchanges)
v3_608_COCO(;batch=1, silent=false, cfgchanges=nothing) = v3_COCO(w=608, h=608, batch=batch, silent=silent, cfgchanges=cfgchanges)

## YOLOV3 SPP
function v3_SPP_COCO(;batch=1, silent=false, w=608, h=608, cfgchanges=[(:net, 1, :width, w), (:net, 1, :height, h)])
    yolo(joinpath(models_dir,"yolov3-spp.cfg"), getArtifact("yolov3-spp-COCO"), batch, silent=silent, cfgchanges=cfgchanges)
end
v3_spp_608_COCO(;batch=1, silent=false, cfgchanges=nothing) = v3_SPP_COCO(w=608, h=608, batch=batch, silent=silent, cfgchanges=cfgchanges)

## YOLOV3-tiny
function v3_tiny_COCO(;batch=1, silent=false, w=416, h=416, cfgchanges=[(:net, 1, :width, w), (:net, 1, :height, h)])
    yolo(joinpath(models_dir,"yolov3-tiny.cfg"), getArtifact("yolov3-tiny-COCO"), batch, silent=silent, cfgchanges=cfgchanges)
end
v3_tiny_416_COCO(;batch=1, silent=false, cfgchanges=nothing) = v3_tiny_COCO(w=416, h=416, batch=batch, silent=silent, cfgchanges=cfgchanges)
