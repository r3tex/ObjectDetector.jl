using ObjectDetector, FileIO

#yolomod = YOLO.v2_tiny_416_COCO(silent=true)
# yolomod = YOLO.v3_tiny_416_COCO(silent=true)
yolomod = YOLO.v3_608_COCO(silent=false, cfgchanges=[(:net,1,:width,512),(:net,1,:height,384)])
#yolomod = YOLO.v3_spp_608_COCO(silent=true)

batch = emptybatch(yolomod)
img = load(joinpath(dirname(dirname(pathof(ObjectDetector))),"test","images","dog-cycle-car.png"))

batch[:,:,:,1] .= gpu(resizePadImage(img, yolomod)) # Send resized image to the batch
res = yolomod(batch) # Run the model on the length-1 batch

target_img_size, padding = ObjectDetector.calcSizeAndPadding(size(img), size(batch))

imgBoxes = drawBoxes(img, yolomod, padding, res)

save(joinpath(@__DIR__,"result.png"), imgBoxes)

ObjectDetector.benchmark()
