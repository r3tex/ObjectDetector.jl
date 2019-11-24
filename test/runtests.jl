using ObjectDetector
using Test, PrettyTables
using FileIO, ImageCore, ImageTransformations

pretrained_list = [
                    YOLO.v2_tiny_416_COCO,
                    # YOLO.v2_608_COCO,
                    YOLO.v3_tiny_416_COCO,
                    YOLO.v3_320_COCO,
                    YOLO.v3_416_COCO,
                    # YOLO.v3_608_COCO,
                    # YOLO.v3_608_spp_COCO
                    ]

IMG = load(joinpath(@__DIR__,"images","dog-cycle-car.png"))

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

        res = yolomod(batch) #run once
        val, t_run, bytes, gctime, memallocs = @timed yolomod(batch);
        table[i, 4] = true
        table[i, 5] = round(t_run, digits=4)
        table[i, 6] = size(res, 2)

        imgBoxes = drawBoxes(IMG, res)
        save(joinpath(@__DIR__,"results","$(modelname)_dog-cycle-car.png"), imgBoxes)

        @test size(res,2) > 0

        @info "$modelname: Ran in $(round(t_run, digits=2)) seconds. (bytes $bytes, gctime $gctime)"
    end
    GC.gc()
end
pretty_table(table, header)
@info "Times approximate. For more accurate benchmarking run ObjectDetector.benchmark()"
