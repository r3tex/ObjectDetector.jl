#using CuArrays
using FileIO, ImageCore, ImageTransformations
using DataFrames
using Flux

prepareimage(img, w, h) = gpu(reshape(permutedims(Float32.(channelview(imresize(img, w, h))[1:3,:,:]), [3,2,1]), w, h, 3, 1))

pkgdir = "/Users/ian/Documents/GitHub/ObjectDetector.jl"
include(joinpath(pkgdir,"src","ObjectDetector.jl"))

models = ["yolov2-608", "yolov2-tiny", "yolov3-320", "yolov3-416", "yolov3-608", "yolov3-spp", "yolov3-tiny"]
imgsizes = [(608,608), (416,416), (320,320), (416,416), (608,608), (608,608), (416,416)]

IMG = load(joinpath(pkgdir,"data","dog-cycle-car.png"))

df = DataFrame(model=String[], load=Bool[], load_time=Float64[], run=Bool[], forwardpass_time=Float64[])

for (i, model) in pairs(models[1:end-1])
    new_df = DataFrame(model=model, load=false, load_time=0.0, run=false, forwardpass_time=0.0)
    @info "Testing: $model ====================================================="
    cfg_file = joinpath(pkgdir,"data","$(model).cfg")
    weights_file = joinpath(pkgdir,"data","$(model).weights")
    IMG_for_model = prepareimage(IMG,imgsizes[i][1],imgsizes[i][1])
    @show typeof(IMG_for_model)
    try
        t = @elapsed begin
            yolomod = ObjectDetector.Yolo(cfg_file, weights_file, 1, silent=true)
        end
        @info "Model successfully loaded in $(round(t, digits=2)) seconds"
        new_df[1, :load] = true
        new_df[1, :load_time] = t
        try
            yolomod(IMG_for_model);
            @info "Model ran succesfully"
            t = @elapsed yolomod(IMG_for_model);
            @info "Single forward pass time: $(round(t, digits=4)) seconds"
            new_df[1, :run] = true
            new_df[1, :forwardpass_time] = t
        catch e2
            bt = catch_backtrace()
            msg = sprint(showerror, e2, bt)
            @error "Error running model"
            println(msg)
        end
    catch e
        bt = catch_backtrace()
        msg = sprint(showerror, e, bt)
        @error "Error loading model"
        println(msg)
    end
    append!(df,new_df)
    GC.gc()
end

display(df)
