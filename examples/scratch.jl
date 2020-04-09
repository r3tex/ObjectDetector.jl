using ObjectDetector, FileIO
using BenchmarkTools

yolomod = YOLO.v3_COCO(silent=false, w=416, h=416)


batch = emptybatch(yolomod)
img = load(joinpath(dirname(dirname(pathof(ObjectDetector))),"test","images","dog-cycle-car_nonsquare.png"))
img = load(joinpath(dirname(dirname(pathof(ObjectDetector))),"test","images","dog-cycle-car.png"))
batch[:,:,:,1], padding = prepareImage(img, yolomod)
save(joinpath(@__DIR__,"result.png"), cpu(batch[:,:,:,1]))

@time res = yolomod(batch, detectThresh=0.2, overlapThresh=0.8) # Run the model on the length-1 batch

imgBoxes = drawBoxes(img, yolomod, padding, res)
save(joinpath(@__DIR__,"result.png"), imgBoxes)

imgBoxes = drawBoxes(collect(img'), yolomod, padding, res, transpose=false)
save(joinpath(@__DIR__,"result_transposed.png"), imgBoxes)

ObjectDetector.benchmark()

using ImageCore, ObjectDetector, FileIO
img = ones(Gray, 200, 100)

yolomod = YOLO.v3_tiny_COCO(w=416, h=416, silent=true)
batch = emptybatch(yolomod)
batch[:,:,:,1], padding = prepareImage(img, yolomod)

res = collect([ padding[1] padding[2] 1.0-padding[3] 1.0-padding[4] 0.0 0.0;]') #note the transpose!

imgboxes = drawBoxes(img, yolomod, padding, res)
save(joinpath(@__DIR__, "test.png"), imgboxes)
all(imgboxes[1,:] .== Gray(0))
all(imgboxes[:,1] .== Gray(0))
all(imgboxes[end,:] .== Gray(0))
all(imgboxes[:,end] .== Gray(0))
imgboxes
