using Flux
path = "/Users/ian/Documents/GitHub/ObjectDetector.jl"
include(joinpath(path,"src","ObjectDetector.jl"))
weights_file =  joinpath(path,"yolov3.weights")
cfg_file = joinpath(path,"yolov3.cfg")
yolo = ObjectDetector.Yolo(cfg_file, weights_file, 1)
IMG = gpu(reshape(permutedims(channelview(load(joinpath(path,"dog-cycle-car.png")))[1:3,:,:], [3,2,1]), 416, 416, 3, 1));
CuArrays.@time yolo(IMG)
