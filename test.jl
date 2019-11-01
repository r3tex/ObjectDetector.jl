using CuArrays
using FileIO, ImageCore

path = "/home/ian/Documents/GitHub/ObjectDetector.jl"
include(joinpath(path,"src","ObjectDetector.jl"))

weights_file =  joinpath(path,"data","yolov3-416.weights")
cfg_file = joinpath(path,"data","yolov3-416.cfg")
yolo = ObjectDetector.Yolo(cfg_file, weights_file, 1)
IMG = cu(reshape(permutedims(channelview(load(joinpath(path,"data","dog-cycle-car.png")))[1:3,:,:], [3,2,1]), 416, 416, 3, 1));
CuArrays.@time yolo(IMG)
