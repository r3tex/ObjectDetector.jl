using FileIO, ImageCore, ImageTransformations, Flux

prepareimage(img, w, h) = gpu(reshape(permutedims(Float32.(channelview(imresize(img, w, h))[1:3,:,:]), [3,2,1]), w, h, 3, 1))

pkgdir = @__DIR__
includet(joinpath(pkgdir,"src","ObjectDetector.jl"))

models = ["yolov2-tiny", "yolov2-608", "yolov3-tiny", "yolov3-320", "yolov3-416", "yolov3-608", "yolov3-spp", ]
imgsizes = [(416,416), (608,608), (416,416), (320,320), (416,416), (608,608), (608,608)]

n = 3 #select model

cfg_file = joinpath(pkgdir,"data","$(models[n]).cfg")
weights_file = joinpath(pkgdir,"data","$(models[n]).weights")

@time yolo = ObjectDetector.Yolo(cfg_file, weights_file, 1, silent=false)

IMG = load(joinpath(pkgdir,"data","dog-cycle-car.png"))
IMG_for_model = prepareimage(IMG,imgsizes[n][1],imgsizes[n][1])


yolo(IMG_for_model)

using BenchmarkTools
@btime yolo(IMG_for_model)

using Profile
Profile.init(n = 10^7, delay = 0.001)

# Visualize using Juno's built-in flame graph
@profiler yolo(IMG_for_model)


# using Profile
# #Profile.init() # returns the current settings
# Profile.init(n = 10^7, delay = 0.01)
# Profile.clear()
# @profile ObjectDetector.Yolo(cfg_file, weights_file, 1, silent=false)
#
# using PProf
# pprof()


# using StatProfilerHTML
# statprofilehtml()
