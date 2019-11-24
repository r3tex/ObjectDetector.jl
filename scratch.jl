using ObjectDetector, FileIO
using Profile

Profile.init(n=10000000, delay=0.01)
yolomod = YOLO.v3_416_COCO()


batch = emptybatch(yolomod)
img = load(joinpath(dirname(dirname(pathof(ObjectDetector))),"test","images","dog-cycle-car.png"))

batch[:,:,:,1] .= gpu(resizePadImage(img, yolomod)) # Send resized image to the batch
res = yolomod(batch) # Run the model on the length-1 batch

imgBoxes = drawBoxes(img, res)

save(joinpath(@__DIR__,"test.png"), imgBoxes)

ObjectDetector.benchmark()


save(joinpath(@__DIR__,"test.png"), resizePadImage(img, yolomod))
