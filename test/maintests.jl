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

expected_result = Float32[0.9061836 0.90347874 0.86342007 0.25070268; 0.06912026 0.09315592 0.048579663 0.31197208; 0.9592236 0.98370415 0.9952955 0.63642; 0.20277503 0.17728066 0.22791806 0.8776546; 0.7579072 0.75653523 0.67497426 0.7199427; 2.4333147f-5 3.6622354f-5 9.410795f-6 5.131908f-5; 5.0670973f-8 9.748572f-7 2.287108f-6 0.0024132896; 0.79304016 0.8786167 0.7855292 0.00013889415; 1.2882784f-6 2.6557157f-6 1.9974797f-5 0.00018639595; 3.9483078f-7 1.2455074f-6 2.3119653f-6 8.458286f-7; 0.00037109293 0.0005130561 0.00041094978 5.6840445f-8; 1.1522116f-6 2.5936124f-6 1.590554f-5 1.222089f-7; 0.33549955 0.18681282 0.28465307 3.0787999f-6; 7.2389037f-7 1.8015919f-6 1.1849861f-5 1.7277285f-5; 4.3378504f-6 4.3711675f-6 4.2347906f-6 1.836845f-7; 1.4324411f-6 6.138337f-7 2.718315f-5 5.601748f-6; 1.392484f-7 1.1233042f-6 5.9373897f-6 7.905464f-8; 6.4429264f-6 4.013792f-5 2.907526f-5 1.7171907f-7; 2.7439762f-7 3.0629474f-6 1.807655f-6 9.4817384f-5; 3.9595194f-9 4.5407308f-8 1.8337872f-7 2.7969329f-5; 1.2165353f-7 5.1604434f-7 2.4497655f-7 0.034334633; 1.7659727f-6 1.5089969f-5 4.227207f-7 0.99178845; 6.662733f-8 6.9862006f-7 4.1930545f-7 3.1253494f-5; 2.8718f-8 1.4523202f-7 1.631737f-6 1.9862251f-5; 7.0118045f-7 5.848172f-7 7.3624815f-6 0.000101643396; 5.0485033f-7 1.4580353f-6 1.6093588f-6 8.989906f-7; 1.3348279f-7 5.669308f-7 4.7351546f-6 1.5622498f-5; 4.156687f-6 3.0817698f-6 0.00012369463 1.2472734f-7; 3.6990633f-7 1.7211746f-6 2.6087644f-6 1.0391945f-5; 7.790469f-6 1.6066344f-5 5.7466114f-6 0.0002043011; 1.8530956f-7 7.185996f-7 3.017728f-6 7.531207f-6; 4.722248f-6 4.0077717f-5 3.8897047f-6 1.5041164f-5; 2.2348969f-8 5.197169f-7 4.6443347f-7 4.599802f-6; 1.6876313f-6 7.4748223f-6 1.6548032f-5 6.859822f-5; 6.174115f-8 2.829325f-7 7.9907574f-7 2.454162f-6; 5.8917756f-8 2.2122501f-7 2.3779474f-7 1.3675987f-6; 5.5239376f-8 7.374804f-7 3.6861707f-7 2.953091f-7; 8.632187f-7 2.1692122f-6 2.3702729f-5 8.8118406f-7; 1.06504295f-7 5.049423f-7 6.972289f-7 3.5403177f-6; 8.413168f-9 6.702054f-8 2.361977f-7 2.6265886f-6; 1.7689601f-7 1.6554736f-8 1.724474f-5 1.2485948f-6; 7.645712f-8 3.781616f-7 1.4209143f-6 3.5144794f-6; 6.44674f-8 7.5890193f-7 5.904003f-7 7.231128f-7; 3.0203108f-8 4.220166f-7 2.2357215f-6 1.1988283f-5; 1.0549161f-7 2.789158f-7 9.371499f-7 2.4820658f-6; 2.0175905f-8 2.6493023f-8 4.6797786f-7 3.1925345f-5; 7.430172f-7 6.373934f-6 8.028852f-7 5.11811f-6; 1.0259796f-8 8.331094f-8 8.7897416f-7 5.1635396f-8; 3.547776f-8 2.438084f-7 5.4184585f-7 1.7775882f-7; 1.3723443f-8 1.6381033f-7 4.49018f-7 2.011293f-6; 1.939725f-7 1.2005167f-6 3.2685443f-6 1.0440632f-6; 7.855397f-8 6.1915216f-7 4.934043f-6 3.152985f-7; 3.448777f-8 9.511359f-8 3.1010482f-6 2.5547075f-7; 1.2292561f-8 1.3019639f-8 7.901263f-7 1.0518296f-6; 2.6007424f-8 9.727651f-8 9.079601f-6 2.0572217f-7; 3.4971258f-8 1.5633313f-8 6.41827f-7 1.3471895f-7; 5.5591027f-8 2.2676471f-7 5.607804f-6 1.5474944f-7; 3.0677143f-9 4.738475f-8 2.6793822f-7 4.039292f-7; 4.278945f-8 1.9126506f-7 4.1550757f-7 4.505628f-6; 2.7700478f-6 2.445585f-6 5.2841406f-5 2.3247149f-7; 7.272614f-6 4.13044f-6 4.2157476f-6 3.9828362f-7; 3.7216847f-7 3.1774462f-6 6.6079924f-6 0.0008551486; 4.1927137f-6 6.9609223f-6 4.004444f-5 0.00021517805; 4.1760504f-7 3.7205385f-7 6.9318603f-6 4.434092f-6; 1.9660123f-7 6.5879925f-7 3.4948869f-6 0.00015892963; 1.3218524f-6 1.712884f-6 7.915165f-8 9.461342f-6; 9.876588f-7 1.017804f-6 2.785479f-5 0.00010794935; 1.8075085f-6 1.7974286f-6 2.8762186f-6 1.0620079f-6; 3.293534f-7 2.6075902f-6 8.40003f-7 1.1317136f-5; 8.557883f-7 4.88165f-7 1.34047805f-5 1.8960001f-6; 8.094455f-7 1.9571617f-6 3.1407487f-6 1.7301534f-6; 1.3119935f-6 1.2231271f-6 1.0616321f-6 3.197129f-5; 3.945902f-6 1.0470602f-5 3.2400567f-5 5.1340487f-7; 7.692556f-7 2.07866f-6 3.758752f-6 8.9871804f-7; 4.984112f-7 8.5427594f-8 6.0065354f-6 1.222916f-6; 3.9040387f-6 4.5831284f-6 6.625041f-5 2.4796336f-6; 1.2494155f-6 7.956592f-6 8.879633f-6 0.00012881805; 4.7369657f-8 2.8674796f-7 1.0128146f-5 8.6169376f-7; 2.3363873f-8 7.9769455f-7 2.675557f-7 1.9117492f-6; 2.201083f-6 3.5499397f-6 1.3272576f-5 1.2196614f-7; 9.863193f-9 2.318904f-8 1.0746955f-6 1.8162882f-6; 2.6425296f-8 8.1972715f-8 5.4809334f-7 1.6376657f-6; 1.9434648f-8 1.0865069f-7 1.3450891f-7 3.065744f-6; 2.6215886f-7 2.2955615f-7 8.044372f-6 6.7533238f-6; 6.849613f-8 5.9965316f-7 1.1289311f-7 9.600693f-7; 0.0 0.0 0.0 0.0; 0.79304016 0.8786167 0.7855292 0.99178845; 3.0 3.0 3.0 17.0]
@testset "batch sizes" begin
    IMG = load(joinpath(@__DIR__, "images", "dog-cycle-car.png"))
    @testset "disallow_bumper :$disallow_bumper" for disallow_bumper in (true, false)
        @testset "Batch size $batch_size" for batch_size in (1, 3)
            yolomod = YOLO.v3_tiny_416_COCO(;batch = batch_size, silent=true, disallow_bumper)
            batch = emptybatch(yolomod)
            @test size(batch) == (416, 416, 3, batch_size)
            @test ObjectDetector.getModelInputSize(yolomod) == size(batch)
            @test typeof(batch) in (Array{Float32, 4}, CuArray{Float32, 4, CUDA.DeviceMemory})
            for b in 1:batch_size
                batch[:, :, :, b], padding = prepareImage(IMG, yolomod)
            end
            res = yolomod(batch, detectThresh  = dThresh, overlapThresh = oThresh);
            expected_objects = 4
            @test size(res) == (89, expected_objects * batch_size)
            @test typeof(res) == Array{Float32, 2}

            # test that all batch results are the same
            for b in 1:batch_size
                @test res[1:end-1, res[end,:] .== b] == expected_result
            end
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

header = ["Model", "loaded?", "load time (s)", "ran?", "run time (s)", "objects detected"]
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
            mkpath(resultsdir)
            batch[:,:,:,1], padding = prepareImage(IMG, yolomod)

            val, t_run, bytes, gctime, m = @timed res = yolomod(batch, detectThresh=dThresh, overlapThresh=oThresh);
            @test size(res,2) > 0
            table[k, 4] = true
            table[k, 5] = round(t_run, digits=4)
            table[k, 6] = size(res, 2)
            @info "$modelname: Ran in $(round(t_run, digits=2)) seconds. (bytes $bytes, gctime $gctime)"

            resfile = joinpath(resultsdir,"$(modelname).png")
            @test_reference resfile drawBoxes(IMG, yolomod, padding, res)
            @info "$modelname: View result: $resfile"

        end
    end
    GC.gc()
end
pretty_table(table, header = header)
@info "Times approximate. For more accurate benchmarking run ObjectDetector.benchmark()"
