## YOLOV2
v2_608_COCO(;batch=1, silent=false) =
    yolo(joinpath(models_dir,"yolov2-608.cfg"), getArtifact("yolov2-COCO"), batch, silent=silent)

## YOLOV2-tiny
v2_tiny_416_COCO(;batch=1, silent=false) =
    yolo(joinpath(models_dir,"yolov2-tiny.cfg"), getArtifact("yolov2-tiny-COCO"), batch, silent=silent)

## YOLOV3
v3_320_COCO(;batch=1, silent=false) =
    yolo(joinpath(models_dir,"yolov3-320.cfg"), getArtifact("yolov3-COCO"), batch, silent=silent)

v3_416_COCO(;batch=1, silent=false) =
    yolo(joinpath(models_dir,"yolov3-416.cfg"), getArtifact("yolov3-COCO"), batch, silent=silent)

v3_608_COCO(;batch=1, silent=false) =
    yolo(joinpath(models_dir,"yolov3-608.cfg"), getArtifact("yolov3-COCO"), batch, silent=silent)

v3_608_spp_COCO(;batch=1, silent=false) =
    yolo(joinpath(models_dir,"yolov3-spp.cfg"), getArtifact("yolov3-spp-COCO"), batch, silent=silent)

## YOLOV3-tiny
v3_tiny_416_COCO(;batch=1, silent=false) =
    yolo(joinpath(models_dir,"yolov3-tiny.cfg"), getArtifact("yolov3-tiny-COCO"), batch, silent=silent)
