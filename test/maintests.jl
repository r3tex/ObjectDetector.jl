dThresh = 0.5 #Detect Threshold (minimum acceptable confidence)
oThresh = 0.5 #Overlap Threshold (maximum acceptable IoU)
@info "Testing all models with detectThresh = $dThresh, overlapThresh = $oThresh"

@testset "Download all artifacts" begin
    @info artifact"yolov2-COCO"
    @info artifact"yolov2-tiny-COCO"
    @info artifact"yolov3-COCO"
    @info artifact"yolov3-spp-COCO"
    @info artifact"yolov3-tiny-COCO"
    @info "All artifacts downloaded"
end

testimages = ["dog-cycle-car_nonsquare","dog-cycle-car"]
pretrained_list = [
                    YOLO.v2_tiny_416_COCO,
                    # YOLO.v2_608_COCO,
                    YOLO.v3_tiny_416_COCO,
                    YOLO.v3_320_COCO,
                    YOLO.v3_416_COCO,
                    YOLO.v3_608_COCO,
                    # YOLO.v3_spp_608_COCO
                    ]

@testset "batch sizes" begin
    IMG = load(joinpath(@__DIR__, "images", "dog-cycle-car.png"))
    @testset "Batch size $batch_size" for batch_size in [1, 3]
        yolomod = YOLO.v3_tiny_416_COCO(batch = batch_size, silent=true)
        batch = emptybatch(yolomod)
        @test size(batch) == (416, 416, 3, batch_size)
        for b in 1:batch_size
            batch[:, :, :, b], padding = prepareImage(IMG, yolomod)
        end
        res = yolomod(batch, detectThresh  = dThresh, overlapThresh = oThresh);
        @test size(res) == (89, 4 * batch_size)
        for b in 2:batch_size
            @test res[1:end-1, res[end,:] .== 1] == res[1:end-1, res[end,:] .== b]
        end
    end
end

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

header = ["Model" "loaded?" "load time (s)" "ran?" "run time (s)" "objects detected"]
table = Array{Any}(undef, length(pretrained_list), 6)
for (k, pretrained) in pairs(pretrained_list)
    global table
    modelname = string(pretrained)
    table[k,:] .= [modelname, false, "-", "-", "-", "-"]
    @testset "Pretrained Model: $modelname" begin
        global table

        t_load = @elapsed begin
            yolomod = pretrained(silent=true)
        end
        table[k, 2] = true
        table[k, 3] = round(t_load, digits=3)
        @info "$modelname: Loaded in $(round(t_load, digits=2)) seconds."

        batch = emptybatch(yolomod)
        for (j, imagename) in pairs(testimages)

            @info """Testing image "$imagename" """
            IMG = load(joinpath(@__DIR__,"images","$imagename.png"))
            resultsdir = joinpath(@__DIR__,"results",imagename)
            !isdir(resultsdir) && mkdir(resultsdir)
            batch[:,:,:,1], padding = prepareImage(IMG, yolomod)

            val, t_run, bytes, gctime, m = @timed res = yolomod(batch, detectThresh=dThresh, overlapThresh=oThresh);
            @test size(res,2) > 0
            table[k, 4] = true
            table[k, 5] = round(t_run, digits=4)
            table[k, 6] = size(res, 2)
            @info "$modelname: Ran in $(round(t_run, digits=2)) seconds. (bytes $bytes, gctime $gctime)"

            imgBoxes = drawBoxes(IMG, yolomod, padding, res)
            resfile = joinpath(resultsdir,"$(modelname).png")
            save(resfile, imgBoxes)
            @info "$modelname: View result: $resfile"

        end
    end
    GC.gc()
end
pretty_table(table, header)
@info "Times approximate. For more accurate benchmarking run ObjectDetector.benchmark()"
