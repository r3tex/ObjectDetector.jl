using ObjectDetector
using Test, PrettyTables
using FileIO, ImageCore

dThresh = 0.5 #Detect Threshold (minimum acceptable confidence)
oThresh = 0.5 #Overlap Threshold (maximum acceptable IoU)
@info "Testing all models with detectThresh = $dThresh, overlapThresh = $oThresh"

testimages = ["dog-cycle-car","dog-cycle-car_nonsquare"]
pretrained_list = [
                    YOLO.v2_tiny_416_COCO,
                    # YOLO.v2_608_COCO,
                    YOLO.v3_tiny_416_COCO,
                    YOLO.v3_320_COCO,
                    YOLO.v3_416_COCO,
                    YOLO.v3_608_COCO,
                    # YOLO.v3_spp_608_COCO
                    ]

header = ["Model" "loaded?" "load time (s)" "ran?" "run time (s)" "objects detected"]
table = Array{Any}(undef, length(pretrained_list) * length(testimages), 6)
for (j, imagename) in pairs(testimages)
    global table
    @info """Testing image "$imagename" """
    IMG = load(joinpath(@__DIR__,"images","$imagename.png"))
    resultsdir = joinpath(@__DIR__,"results",imagename)
    !isdir(resultsdir) && mkdir(resultsdir)
    for (k, pretrained) in pairs(pretrained_list)
        i = ((j-1)*length(pretrained_list)) + k
        modelname = string(pretrained)
        table[i,:] .= [modelname, false, "-", "-", "-", "-"]
        @testset "Pretrained Model: $modelname" begin
            global table

            t_load = @elapsed begin
                yolomod = pretrained(silent=true)
            end
            table[i, 2] = true
            table[i, 3] = round(t_load, digits=3)
            @info "$modelname: Loaded in $(round(t_load, digits=2)) seconds."

            batch = emptybatch(yolomod)
            batch[:,:,:,1], padding = prepareImage(IMG, yolomod)

            res = yolomod(batch, detectThresh=dThresh, overlapThresh=oThresh) #run once
            @test size(res,2) > 0

            val, t_run, bytes, gctime, m = @timed yolomod(batch, detectThresh=dThresh, overlapThresh=oThresh);
            table[i, 4] = true
            table[i, 5] = round(t_run, digits=4)
            table[i, 6] = size(res, 2)
            @info "$modelname: Ran in $(round(t_run, digits=2)) seconds. (bytes $bytes, gctime $gctime)"

            imgBoxes = drawBoxes(IMG, yolomod, padding, res)
            resfile = joinpath(resultsdir,"$(modelname).jpg")
            save(resfile, imgBoxes)
            @info "$modelname: View result: $resfile"

        end
        GC.gc()
    end
end
pretty_table(table, header)
@info "Times approximate. For more accurate benchmarking run ObjectDetector.benchmark()"


@testset "Custom cfg's" begin
    @testset "Valid non-square dimensions (512x384)" begin
        IMG = load(joinpath(@__DIR__,"images","dog-cycle-car.png"))
        yolomod = YOLO.v3_COCO(silent=true, cfgchanges=[(:net, 1, :width, 512), (:net, 1, :height, 384)])
        batch = emptybatch(yolomod)
        batch[:,:,:,1], padding = prepareImage(IMG, yolomod)
        res = yolomod(batch, detectThresh=dThresh, overlapThresh=oThresh) #run once
        @test size(res,2) > 0
    end
    @testset "Invalid non-square dimensions" begin
        IMG = load(joinpath(@__DIR__,"images","dog-cycle-car.png"))
        # invalid height
        @test_throws AssertionError YOLO.v3_COCO(silent=false, w=512, h=383)
        # invalid width
        @test_throws AssertionError YOLO.v3_COCO(silent=false, w=511, h=384)
    end
end

@testset "Drawing boxes" begin
    # Testing that boxes are drawn as expected

    ### square model
    yolomod = YOLO.v3_tiny_COCO(w=416, h=416, silent=true)
    #Nonsquare low aspect ratio image
    img = ones(Gray, 200, 100)
    batch = emptybatch(yolomod)
    batch[:,:,:,1], padding = prepareImage(img, yolomod)
    res = collect([ padding[1] padding[2] 1.0-padding[3] 1.0-padding[4] 0.0 0.0;]') #note the transpose!
    imgboxes = drawBoxes(img, yolomod, padding, res)
    @test all(imgboxes[1,:] .== Gray(0))
    @test all(imgboxes[:,1] .== Gray(0))
    @test all(imgboxes[end,:] .== Gray(0))
    @test all(imgboxes[:,end] .== Gray(0))

    #Nonsquare high aspect ratio image
    img = ones(Gray, 100, 200)
    batch = emptybatch(yolomod)
    batch[:,:,:,1], padding = prepareImage(img, yolomod)
    res = collect([ padding[1] padding[2] 1.0-padding[3] 1.0-padding[4] 0.0 0.0;]') #note the transpose!
    imgboxes = drawBoxes(img, yolomod, padding, res)
    @test all(imgboxes[1,:] .== Gray(0))
    @test all(imgboxes[:,1] .== Gray(0))
    @test all(imgboxes[end,:] .== Gray(0))
    @test all(imgboxes[:,end] .== Gray(0))

    #Square image, square model
    img = ones(Gray, 100, 100)
    batch = emptybatch(yolomod)
    batch[:,:,:,1], padding = prepareImage(img, yolomod)
    res = collect([ padding[1] padding[2] 1.0-padding[3] 1.0-padding[4] 0.0 0.0;]') #note the transpose!
    imgboxes = drawBoxes(img, yolomod, padding, res)
    @test all(imgboxes[1,:] .== Gray(0))
    @test all(imgboxes[:,1] .== Gray(0))
    @test all(imgboxes[end,:] .== Gray(0))
    @test all(imgboxes[:,end] .== Gray(0))

    ### nonsquare low aspect ratio model
    yolomod = YOLO.v3_tiny_COCO(w=512, h=416, silent=true)
    #Square image
    img = ones(Gray, 100, 100)
    batch = emptybatch(yolomod)
    batch[:,:,:,1], padding = prepareImage(img, yolomod)
    res = collect([ padding[1] padding[2] 1.0-padding[3] 1.0-padding[4] 0.0 0.0;]') #note the transpose!
    imgboxes = drawBoxes(img, yolomod, padding, res)
    @test all(imgboxes[1,:] .== Gray(0))
    @test all(imgboxes[:,1] .== Gray(0))
    @test all(imgboxes[end,:] .== Gray(0))
    @test all(imgboxes[:,end] .== Gray(0))

    #nonsquare low aspect ratio image
    img = ones(Gray, 200, 100)
    batch = emptybatch(yolomod)
    batch[:,:,:,1], padding = prepareImage(img, yolomod)
    res = collect([ padding[1] padding[2] 1.0-padding[3] 1.0-padding[4] 0.0 0.0;]') #note the transpose!
    imgboxes = drawBoxes(img, yolomod, padding, res)
    @test all(imgboxes[1,:] .== Gray(0))
    @test all(imgboxes[:,1] .== Gray(0))
    @test all(imgboxes[end,:] .== Gray(0))
    @test all(imgboxes[:,end] .== Gray(0))

    ### nonsquare high aspect ratio model
    yolomod = YOLO.v3_tiny_COCO(w=416, h=512, silent=true)
    #Square image
    img = ones(Gray, 100, 100)
    batch = emptybatch(yolomod)
    batch[:,:,:,1], padding = prepareImage(img, yolomod)
    res = collect([ padding[1] padding[2] 1.0-padding[3] 1.0-padding[4] 0.0 0.0;]') #note the transpose!
    imgboxes = drawBoxes(img, yolomod, padding, res)
    @test all(imgboxes[1,:] .== Gray(0))
    @test all(imgboxes[:,1] .== Gray(0))
    @test all(imgboxes[end,:] .== Gray(0))
    @test all(imgboxes[:,end] .== Gray(0))

    #nonsquare low aspect ratio image
    img = ones(Gray, 100, 200)
    batch = emptybatch(yolomod)
    batch[:,:,:,1], padding = prepareImage(img, yolomod)
    res = collect([ padding[1] padding[2] 1.0-padding[3] 1.0-padding[4] 0.0 0.0;]') #note the transpose!
    imgboxes = drawBoxes(img, yolomod, padding, res)
    @test all(imgboxes[1,:] .== Gray(0))
    @test all(imgboxes[:,1] .== Gray(0))
    @test all(imgboxes[end,:] .== Gray(0))
    @test all(imgboxes[:,end] .== Gray(0))

    ## Non-transposed
    imgboxes = drawBoxes(collect(img'), yolomod, padding, res, transpose=false)
    @test all(imgboxes[1,:] .== Gray(0))
    @test all(imgboxes[:,1] .== Gray(0))
    @test all(imgboxes[end,:] .== Gray(0))
    @test all(imgboxes[:,end] .== Gray(0))

end
