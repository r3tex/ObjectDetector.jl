dThresh = 0.5 #Detect Threshold (minimum acceptable confidence)
oThresh = 0.5 #Overlap Threshold (maximum acceptable IoU)
@info "Testing all models with detect_thresh = $dThresh, overlap_thresh = $oThresh"

@testset "Download all artifacts" begin
    @info artifact"yolov2-COCO"
    @info artifact"yolov2-tiny-COCO"
    @info artifact"yolov3-COCO"
    @info artifact"yolov3-spp-COCO"
    @info artifact"yolov3-tiny-COCO"
    @info artifact"yolov4-COCO"
    @info artifact"yolov4-tiny-COCO"
    @info artifact"yolov7-COCO"
    @info artifact"yolov7-tiny-COCO"
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
                    # YOLO.v4_tiny_416_COCO,
                    # YOLO.v4_416_COCO,
                    # YOLO.v7_tiny_416_COCO,
                    # YOLO.v7_416_COCO,
                    ]

Darknet.download_defaults()
datadir = joinpath(pkgdir(Darknet), "data")
metafile = joinpath(datadir, "coco.data")
namesfile = joinpath(ObjectDetector.YOLO.models_dir(), "coco.names")
names = collect(eachline(namesfile))
meta = Darknet.get_metadata(metafile)
img = load(joinpath(@__DIR__, "images", "dog-cycle-car.png"))
img_d = Darknet.array_to_image(convert(Array{Float32}, channelview(img)))
buffer_output = !Sys.iswindows() # windows seems to oom or something and we don't get the output
@testset "Check all models work in Darknet" begin
    @testset "$k" for (k, v) in YOLO.YOLO_MODELS
        cfgfile, weightsfile = v()
        @info "Testing Darknet.jl with $k. cfgfile=$(basename(cfgfile)), weightsfile=$(basename(weightsfile))"
        mktemp() do path, io
            io = buffer_output ? io : stdout
            results = try
                redirect_stdio(;stderr=io, stdout=io) do
                    net = Darknet.load_network(cfgfile, weightsfile, 1)
                    Darknet.detect(net, meta, img_d, thresh=dThresh, nms=oThresh)
                end
            catch
                @error "Failed to load $k" cfgfile weightsfile
                if buffer_output
                    close(io)
                    println("darknet output ========")
                    println(read(path, String))
                    println("end darknet output ====")
                end
                rethrow()
            end
            @test size(results, 1) >= 2 # minimum, Most detect 3.
        end
    end
end

expected_result = Float32[0.9061836 0.90347874 0.86342007 0.25070268; 0.06912026 0.09315592 0.048579663 0.31197208; 0.9592236 0.98370415 0.9952955 0.63642; 0.20277503 0.17728066 0.22791806 0.8776546; 0.7579072 0.75653523 0.67497426 0.7199427; 1.8442268f-5 2.77061f-5 6.3520447f-6 3.6946796f-5; 3.8403897f-8 7.3751386f-7 1.543739f-6 0.0017374302; 0.60105085 0.6647045 0.530212 9.999582f-5; 9.763955f-7 2.0091425f-6 1.3482474f-5 0.0001341944; 2.992451f-7 9.422702f-7 1.560517f-6 6.089481f-7; 0.000281254 0.00038814504 0.00027738052 4.092186f-8; 8.7326947f-7 1.962159f-6 1.073583f-5 8.798341f-8; 0.25427753 0.14133048 0.1921335 2.2165596f-6; 5.486417f-7 1.3629677f-6 7.998351f-6 1.2438655f-5; 3.287688f-6 3.3069423f-6 2.8583747f-6 1.3224232f-7; 1.0856575f-6 4.6438683f-7 1.8347928f-5 4.0329373f-6; 1.0553737f-7 8.498192f-7 4.007585f-6 5.6914807f-8; 4.8831403f-6 3.0365749f-5 1.9625051f-5 1.2362788f-7; 2.0796793f-7 2.3172277f-6 1.2201207f-6 6.8263085f-5; 3.0009484f-9 3.435223f-8 1.2377592f-7 2.0136313f-5; 9.2202086f-8 3.9040572f-7 1.6535286f-7 0.024718968; 1.3384434f-6 1.1416093f-5 2.853256f-7 0.71403086; 5.0497334f-8 5.285307f-7 2.830204f-7 2.2500724f-5; 2.176558f-8 1.0987314f-7 1.1013805f-6 1.4299682f-5; 5.3142975f-7 4.424348f-7 4.9694854f-6 7.317742f-5; 3.826297f-7 1.1030551f-6 1.0862758f-6 6.472217f-7; 1.01167565f-7 4.2890312f-7 3.1961074f-6 1.12473035f-5; 3.150383f-6 2.3314674f-6 8.349069f-5 8.9796536f-8; 2.8035467f-7 1.3021292f-6 1.7608488f-6 7.4816044f-6; 5.9044523f-6 1.21547555f-5 3.878815f-6 0.00014708508; 1.4044745f-7 5.436459f-7 2.0368886f-6 5.4220372f-6; 3.5790258f-6 3.0320205f-5 2.6254506f-6 1.0828776f-5; 1.6938445f-8 3.9318414f-7 3.1348063f-7 3.311594f-6; 1.279068f-6 5.6549666f-6 1.1169496f-5 4.938679f-5; 4.679406f-8 2.140484f-7 5.3935554f-7 1.766856f-6; 4.4654193f-8 1.6736452f-7 1.6050532f-7 9.845927f-7; 4.1866322f-8 5.5792987f-7 2.4880703f-7 2.1260563f-7; 6.5423967f-7 1.6410854f-6 1.5998732f-5 6.34402f-7; 8.0720376f-8 3.8200665f-7 4.7061158f-7 2.548826f-6; 6.3764007f-9 5.07034f-8 1.5942737f-7 1.8909933f-6; 1.3407076f-7 1.2524241f-8 1.1639756f-5 8.989167f-7; 5.7947403f-8 2.8609256f-7 9.590806f-7 2.5302238f-6; 4.8860308f-8 5.7413604f-7 3.9850502f-7 5.2059977f-7; 2.2891154f-8 3.192704f-7 1.5090544f-6 8.630877f-6; 7.995285f-8 2.1100962f-7 6.3255203f-7 1.7869452f-6; 1.5291464f-8 2.0042906f-8 3.15873f-7 2.2984419f-5; 5.631381f-7 4.8221054f-6 5.4192685f-7 3.684746f-6; 7.775973f-9 6.3027656f-8 5.9328494f-7 3.7174527f-8; 2.688885f-8 1.8444963f-7 3.65732f-7 1.2797616f-7; 1.0401096f-8 1.2392829f-7 3.0307558f-7 1.4480156f-6; 1.4701315f-7 9.0823323f-7 2.2061834f-6 7.5166565f-7; 5.9536617f-8 4.6841043f-7 3.330352f-6 2.2699686f-7; 2.613853f-8 7.1956784f-8 2.0931277f-6 1.839243f-7; 9.31662f-9 9.849815f-9 5.333149f-7 7.5725706f-7; 1.9711214f-8 7.359311f-8 6.1284973f-6 1.4810817f-7; 2.6504969f-8 1.1827153f-8 4.332167f-7 9.698992f-8; 4.213284f-8 1.715555f-7 3.7851232f-6 1.1141073f-7; 2.3250428f-9 3.584823f-8 1.808514f-7 2.9080587f-7; 3.243043f-8 1.4469876f-7 2.8045693f-7 3.243794f-6; 2.0994391f-6 1.8501713f-6 3.5666588f-5 1.6736615f-7; 5.5119663f-6 3.1248235f-6 2.845521f-6 2.8674137f-7; 2.8206918f-7 2.40385f-6 4.460225f-6 0.000615658; 3.177688f-6 5.266183f-6 2.7028967f-5 0.00015491586; 3.1650586f-7 2.8147184f-7 4.6788273f-6 3.192292f-6; 1.4900549f-7 4.9840486f-7 2.3589587f-6 0.000114420225; 1.0018415f-6 1.2958571f-6 5.3425325f-8 6.811624f-6; 7.4855376f-7 7.700046f-7 1.8801267f-5 7.771735f-5; 1.3699238f-6 1.359818f-6 1.9413735f-6 7.645848f-7; 2.4961932f-7 1.9727338f-6 5.669804f-7 8.147689f-6; 6.4860814f-7 3.6931405f-7 9.047882f-6 1.3650114f-6; 6.134846f-7 1.4806617f-6 2.1199246f-6 1.2456113f-6; 9.943693f-7 9.253388f-7 7.165744f-7 2.3017496f-5; 2.9906275f-6 7.92138f-6 2.186955f-5 3.696221f-7; 5.8302436f-7 1.5725794f-6 2.537061f-6 6.470255f-7; 3.7774947f-7 6.4628985f-8 4.054257f-6 8.8042947f-7; 2.9588991f-6 3.467298f-6 4.4717322f-5 1.7851941f-6; 9.4694104f-7 6.019442f-6 5.9935237f-6 9.274161f-5; 3.5901806f-8 2.1693494f-7 6.836238f-6 6.2037014f-7; 1.7707647f-8 6.03484f-7 1.805932f-7 1.3763499f-6; 1.6682166f-6 2.6856544f-6 8.958647f-6 8.780863f-8; 7.475385f-9 1.7543325f-8 7.253918f-7 1.3076234f-6; 2.0027922f-8 6.2015246f-8 3.699489f-7 1.1790255f-6; 1.4729659f-8 8.219807f-8 9.079005f-8 2.20716f-6; 1.9869209f-7 1.7366732f-7 5.4297443f-6 4.862006f-6; 5.191371f-8 4.5365874f-7 7.6199946f-8 6.911949f-7; 0.0 0.0 0.0 0.0; 0.60105085 0.6647045 0.530212 0.71403086; 3.0 3.0 3.0 17.0]
@testset "Comparison to Darknet" begin

    cfgfile = joinpath(ObjectDetector.YOLO.models_dir(), "yolov3-tiny.cfg")
    weightsfile = joinpath(artifact"yolov3-tiny-COCO", "yolov3-tiny-COCO.weights")

    net = Darknet.load_network(cfgfile, weightsfile, 1)
    results = Darknet.detect(net, meta, img_d, thresh=dThresh, nms=oThresh)
    @test length(results) == size(expected_result, 2)

    img_h, img_w = size(img, 1), size(img, 2)

    results_xyxy = Matrix{Float32}(undef, size(expected_result, 1), length(results))
    for i in eachindex(results)
        # Darknet.jl returns pixel coords in x, y, w, h format, where x & y are center coords
        d_w = results[i][3][3]
        d_x1 = results[i][3][1] - d_w / 2
        d_x2 = results[i][3][1] + d_w / 2
        d_h = results[i][3][4]
        d_y1 = results[i][3][2] - d_h / 2
        d_y2 = results[i][3][2] + d_h / 2
        darknet_bbox = Float32[d_x1 / img_w, d_y1 / img_h, d_x2 / img_w, d_y2 / img_h]
        results_xyxy[1:4, i] = darknet_bbox
        class_id = findfirst(==(results[i][1]), names)
        class_id === nothing && @error "Class not found in names file" results[i][1] names
        results_xyxy[end, i] = class_id
        results_xyxy[end-1, i] = results[i][2]
    end

    darknet_results = sortslices(results_xyxy, dims=2, by = x -> x[1])
    expected_results = sortslices(expected_result, dims=2, by = x -> x[1])

    @testset "result $i" for i in axes(expected_results, 2)
        @testset "bbox" begin
            @test darknet_results[1:4, i] ≈ expected_results[1:4, i] # bbox
        end
        @testset "conf" begin
            @test darknet_results[end-1, i] ≈ expected_results[end-1, i] # conf
        end
        @testset "class id" begin
            @test darknet_results[end, i] == expected_results[end, i] # class id
        end
    end
end

@testset "batch sizes" begin

    @testset "disallow_bumper :$disallow_bumper" for disallow_bumper in (true, false)
        @testset "Batch size $batch_size" for batch_size in (1, 3)
            yolomod = YOLO.v3_tiny_416_COCO(;batch = batch_size, silent=true, disallow_bumper)
            batch = emptybatch(yolomod)
            @test size(batch) == (416, 416, 3, batch_size)
            @test ObjectDetector.get_input_size(yolomod) == size(batch)
            @test typeof(batch) in (Array{Float32, 4}, CuArray{Float32, 4, CUDA.DeviceMemory})
            for b in 1:batch_size
                batch[:, :, :, b], padding = prepare_image(img, yolomod)
            end
            res = yolomod(batch, detect_thresh  = dThresh, overlap_thresh = oThresh);
            expected_objects = 4
            @test size(res) == (89, expected_objects * batch_size)
            @test typeof(res) == Array{Float32, 2}

            # test that all batch results are the same
            for b in 1:batch_size
                @test res[1:end-1, res[end,:] .== b] ≈ expected_result
            end
        end
    end
end

@testset "Custom cfg's" begin
    @testset "Valid non-square dimensions (512x384)" begin
        img = load(joinpath(@__DIR__,"images","dog-cycle-car.png"))
        yolomod = YOLO.v3_COCO(silent=true, cfgchanges=[(:net, 1, :width, 512), (:net, 1, :height, 384)])
        batch = emptybatch(yolomod)
        batch[:,:,:,1], padding = prepare_image(img, yolomod)
        res = yolomod(batch, detect_thresh=dThresh, overlap_thresh=oThresh) #run once
        @test size(res,2) > 0
    end
    @testset "Invalid non-square dimensions" begin
        img = load(joinpath(@__DIR__,"images","dog-cycle-car.png"))
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
            img = load(joinpath(@__DIR__,"images","$imagename.png"))
            resultsdir = joinpath(@__DIR__,"results",imagename)
            mkpath(resultsdir)
            batch[:,:,:,1], padding = prepare_image(img, yolomod)

            val, t_run, bytes, gctime, m = @timed res = yolomod(batch, detect_thresh=dThresh, overlap_thresh=oThresh);
            @test size(res,2) > 0
            table[k, 4] = true
            table[k, 5] = round(t_run, digits=4)
            table[k, 6] = size(res, 2)
            @info "$modelname: Ran in $(round(t_run, digits=2)) seconds. (bytes $bytes, gctime $gctime)"

            resfile = joinpath(resultsdir,"$(modelname).png")
            @test_reference resfile draw_boxes(img, yolomod, padding, res)
            @info "$modelname: View result: $resfile"

        end
    end
    GC.gc()
end
pretty_table(table, header = header)
@info "Times approximate. For more accurate benchmarking run ObjectDetector.benchmark()"
