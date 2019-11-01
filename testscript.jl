using CuArrays
using FileIO, ImageCore, ImageTransformations
using DataFrames

prepareimage(img, w, h) = cu(reshape(permutedims(channelview(imresize(img, w, h))[1:3,:,:], [3,2,1]), w, h, 3, 1))

pkgdir = "/home/ian/Documents/GitHub/ObjectDetector.jl"
include(joinpath(pkgdir,"src","ObjectDetector.jl"))

models = ["yolov2-608", "yolov3-320", "yolov3-416", "yolov3-608", "yolov3-spp", "yolov3-tiny"]
imgsizes = [(608,608), (320,320), (416,416), (608,608), (608,608), (416,416)]

IMG = load(joinpath(pkgdir,"data","dog-cycle-car.png"))

df = DataFrame(model=String[], load=Bool[], load_time=Float64[], run=Bool[], forwardpass_time=Float64[])

for (i, model) in pairs(models[1:end-1])
    new_df = DataFrame(model=model, load=false, load_time=0.0, run=false, forwardpass_time=0.0)
    @info "Testing: $model ====================================================="
    cfg_file = joinpath(pkgdir,"data","$(model).cfg")
    weights_file = joinpath(pkgdir,"data","$(model).weights")
    IMG_for_model = prepareimage(IMG,imgsizes[i][1],imgsizes[i][1])
    try
        t = @elapsed begin
            yolo = ObjectDetector.Yolo(cfg_file, weights_file, 1)
        end
        @info "Model successfully loaded in $(round(t, digits=2)) seconds"
        new_df[:load] = true
        new_df[:load_time] = t
        try
            yolo(IMG_for_model);
            @info "Model ran succesfully"
            t = @elapsed yolo(IMG_for_model);
            @info "Single forward pass time: $(round(t, digits=4)) seconds"
            new_df[:run] = true
            new_df[:forwardpass_time] = t
        catch e2
            @error "Error running model"
            @show e2
        end
    catch e
        @error "Error loading model"
        @show e
    end
    append!(df,new_df)
end

display(df)
