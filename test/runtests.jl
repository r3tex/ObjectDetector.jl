using ObjectDetector
using Test, PrettyTables
using FileIO, ImageCore, ImageTransformations

dThresh = 0.5 #Detect Threshold (minimum acceptable confidence)
oThresh = 0.5 #Overlap Threshold (maximum acceptable IoU)
@info "Testing all models with detectThresh = $dThresh, overlapThresh = $oThresh"
pretrained_list = [
                    YOLO.v2_tiny_416_COCO,
                    # YOLO.v2_608_COCO,
                    YOLO.v3_tiny_416_COCO,
                    YOLO.v3_320_COCO,
                    YOLO.v3_416_COCO,
                    YOLO.v3_608_COCO,
                    # YOLO.v3_spp_608_COCO
                    ]

IMG = load(joinpath(@__DIR__,"images","dog-cycle-car.png"))

resultsdir = joinpath(@__DIR__,"results")
!isdir(resultsdir) && mkdir(resultsdir)

header = ["Model" "loaded?" "load time (s)" "ran?" "run time (s)" "objects detected"]
table = Array{Any}(undef, length(pretrained_list), 6)
for (i, pretrained) in pairs(pretrained_list)
    modelname = string(pretrained)
    table[i,:] = [modelname false "-" "-" "-" "-"]
    @testset "Pretrained Model: $modelname" begin
        global table

        t_load = @elapsed begin
            yolomod = pretrained(silent=true)
        end
        table[i, 2] = true
        table[i, 3] = round(t_load, digits=3)
        @info "$modelname: Loaded in $(round(t_load, digits=2)) seconds."

        batch = emptybatch(yolomod)
        batch[:,:,:,1] .= gpu(resizePadImage(IMG, yolomod))

        res = yolomod(batch, detectThresh=dThresh, overlapThresh=oThresh) #run once
        @test size(res,2) > 0

        val, t_run, bytes, gctime, m = @timed yolomod(batch, detectThresh=dThresh, overlapThresh=oThresh);
        table[i, 4] = true
        table[i, 5] = round(t_run, digits=4)
        table[i, 6] = size(res, 2)
        @info "$modelname: Ran in $(round(t_run, digits=2)) seconds. (bytes $bytes, gctime $gctime)"

        imgBoxes = drawBoxes(IMG, res)
        resfile = joinpath(resultsdir,"$(modelname)_dog-cycle-car.jpg")
        save(resfile, imgBoxes)
        @info "$modelname: View result: $resfile"

    end
    GC.gc()
end
pretty_table(table, header)
@info "Times approximate. For more accurate benchmarking run ObjectDetector.benchmark()"
