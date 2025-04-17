dThresh = 0.5 #Detect Threshold (minimum acceptable confidence)
oThresh = 0.5 #Overlap Threshold (maximum acceptable IoU)
@info "Testing all models with detect_thresh = $dThresh, overlap_thresh = $oThresh"

psnr_thresh = 35.0

@testset "Download all artifacts" begin
    # artifact"yolov2-COCO" # broken, see below
    artifact"yolov2-tiny-COCO"
    artifact"yolov3-COCO"
    artifact"yolov3-spp-COCO"
    artifact"yolov3-tiny-COCO"
    artifact"yolov4-COCO"
    artifact"yolov4-tiny-COCO"
    artifact"yolov7-COCO"
    artifact"yolov7-tiny-COCO"
end

Darknet.download_defaults()

const skip_models = (
    "v2_COCO", # Not all weights are read during load: Read 196856372 bytes. Filesize 203934260 bytes
    # "v2_tiny_COCO",
    # "v3_COCO",
    # "v3_tiny_COCO",
    # "v3_spp_COCO",
    # "v4_COCO",
    # "v4_tiny_COCO",
    # "v7_COCO",
    # "v7_tiny_COCO",
)

const testimages = ["dog-cycle-car", "dog-cycle-car_nonsquare"]
const namesfile = joinpath(ObjectDetector.YOLO.models_dir(), "coco.names")
const names = collect(eachline(namesfile))
const datadir = joinpath(pkgdir(Darknet), "data")
const metafile = joinpath(datadir, "coco.data")
const meta = Darknet.get_metadata(metafile)

include("resrefs.jl")

@testset "Darknet vs ObjectDetector" begin
    @testset "$modelname" for (modelname, files) in sort(collect(YOLO.YOLO_MODELS), by = first)
        if modelname in skip_models
            @test_skip nothing
            continue
        end
        cfgfile, weightsfile = files()
        @info "Testing model $modelname"

        yolomod, net = nothing, nothing

        @testset "Load in Darknet.jl" begin
            net = @suppress Darknet.load_network(cfgfile, weightsfile, 1)
        end
        @testset "Load in ObjectDetector.jl" begin
            yolomod = YOLO.yolo_model(modelname; silent=true)
        end

        (yolomod === nothing || net === nothing) && continue

        @testset "$imagename" for imagename in testimages
            @info "  Testing image $imagename"
            img = load(joinpath(@__DIR__, "images", "$imagename.png"))
            resultsdir = joinpath(@__DIR__,"results", imagename)
            mkpath(resultsdir)
            batch = emptybatch(yolomod)
            img_padded, padding = prepare_image(img, yolomod)
            batch[:,:,:,1] .= img_padded
            @test_reference joinpath(resultsdir,"$(modelname)_in_padded.png") cpu(img_padded) by=psnr_equality(psnr_thresh)

            # test darknet
            img_padded_darknet = collect(PermutedDimsArray(img_padded, (3, 2, 1)))
            img_d = Darknet.array_to_image(img_padded_darknet)
            darkres = Darknet.detect(net, meta, img_d, thresh=dThresh, nms=oThresh)

            # test ObjectDetector
            juliares = yolomod(batch, detect_thresh=dThresh, overlap_thresh=oThresh)

            od_resimg = joinpath(resultsdir,"$(modelname)_out_od.png")
            if juliares !== nothing
                @test_reference od_resimg draw_boxes(img, yolomod, padding, juliares) by=psnr_equality(psnr_thresh)
            end

            darkres_xyxy = zeros(Float32, 89, length(darkres))
            # Note: These might be the wrong way around, but our padded input is
            # square so it doesn't matter here currently
            img_h, img_w = size(img_padded, 1), size(img_padded, 2)
            for i in eachindex(darkres)
                d_x, d_y, d_w, d_h = darkres[i][3]
                d_x1, d_y1 = d_x - d_w/2, d_y - d_h/2
                d_x2, d_y2 = d_x + d_w/2, d_y + d_h/2
                bbox = Float32[d_x1 / img_w, d_y1 / img_h, d_x2 / img_w, d_y2 / img_h]
                conf = Float32(darkres[i][2])
                class_id = findfirst(==(darkres[i][1]), names)
                darkres_xyxy[1:4, i] = bbox
                darkres_xyxy[end-2, i] = conf
                darkres_xyxy[end-1, i] = class_id
                # last row is batch id
            end
            darknet_resimg = joinpath(resultsdir,"$(modelname)_out_darknet.png")
            @test_reference darknet_resimg draw_boxes(img, yolomod, padding, darkres_xyxy) by=psnr_equality(psnr_thresh)

            @test ReferenceTests._psnr(load(od_resimg), load(darknet_resimg)) > 30.0
            @test size(darkres_xyxy, 2) > 0
            @test size(darkres_xyxy) == size(juliares)
            if size(darkres_xyxy) != size(juliares)
                # Don't test further because the tests below will show as errors not failures
                continue
            end
            dark_sorted = sortslices(darkres_xyxy, dims=2, by = x -> x[1])
            julia_sorted = sortslices(juliares, dims=2, by = x -> x[1])

            ref_dark_sorted =  get!(RES_REFS, "dn_$(modelname)_$(imagename)", dark_sorted)
            ref_julia_sorted = get!(RES_REFS, "od_$(modelname)_$(imagename)", julia_sorted)
            @test dark_sorted ≈ ref_dark_sorted atol=0.05
            @test julia_sorted ≈ ref_julia_sorted atol=0.05

            dark_bbox = dark_sorted[1:4, :]
            julia_bbox = julia_sorted[1:4, :]
            @test dark_bbox ≈ julia_bbox atol=0.05

            dark_conf = dark_sorted[end-2, :]
            julia_conf = julia_sorted[end-2, :]
            @test dark_conf ≈ julia_conf atol=0.05

            dark_classid = dark_sorted[end-1, :]
            julia_classid = julia_sorted[end-1, :]
            @test dark_classid == julia_classid
        end
        GC.gc()
    end
end

# println(repr(sort!(RES_REFS)))

@testset "batch sizes" begin
    img = load(joinpath(@__DIR__, "images", "$(testimages[1]).png"))
    expected_result = Float32[0.9061836 0.90347874 0.86342007 0.25070268; 0.06912026 0.09315592 0.048579663 0.31197208; 0.9592236 0.98370415 0.9952955 0.63642; 0.20277503 0.17728066 0.22791806 0.8776546; 0.7579072 0.75653523 0.67497426 0.7199427; 1.8442268f-5 2.77061f-5 6.3520447f-6 3.6946796f-5; 3.8403897f-8 7.3751386f-7 1.543739f-6 0.0017374302; 0.60105085 0.6647045 0.530212 9.999582f-5; 9.763955f-7 2.0091425f-6 1.3482474f-5 0.0001341944; 2.992451f-7 9.422702f-7 1.560517f-6 6.089481f-7; 0.000281254 0.00038814504 0.00027738052 4.092186f-8; 8.7326947f-7 1.962159f-6 1.073583f-5 8.798341f-8; 0.25427753 0.14133048 0.1921335 2.2165596f-6; 5.486417f-7 1.3629677f-6 7.998351f-6 1.2438655f-5; 3.287688f-6 3.3069423f-6 2.8583747f-6 1.3224232f-7; 1.0856575f-6 4.6438683f-7 1.8347928f-5 4.0329373f-6; 1.0553737f-7 8.498192f-7 4.007585f-6 5.6914807f-8; 4.8831403f-6 3.0365749f-5 1.9625051f-5 1.2362788f-7; 2.0796793f-7 2.3172277f-6 1.2201207f-6 6.8263085f-5; 3.0009484f-9 3.435223f-8 1.2377592f-7 2.0136313f-5; 9.2202086f-8 3.9040572f-7 1.6535286f-7 0.024718968; 1.3384434f-6 1.1416093f-5 2.853256f-7 0.71403086; 5.0497334f-8 5.285307f-7 2.830204f-7 2.2500724f-5; 2.176558f-8 1.0987314f-7 1.1013805f-6 1.4299682f-5; 5.3142975f-7 4.424348f-7 4.9694854f-6 7.317742f-5; 3.826297f-7 1.1030551f-6 1.0862758f-6 6.472217f-7; 1.01167565f-7 4.2890312f-7 3.1961074f-6 1.12473035f-5; 3.150383f-6 2.3314674f-6 8.349069f-5 8.9796536f-8; 2.8035467f-7 1.3021292f-6 1.7608488f-6 7.4816044f-6; 5.9044523f-6 1.21547555f-5 3.878815f-6 0.00014708508; 1.4044745f-7 5.436459f-7 2.0368886f-6 5.4220372f-6; 3.5790258f-6 3.0320205f-5 2.6254506f-6 1.0828776f-5; 1.6938445f-8 3.9318414f-7 3.1348063f-7 3.311594f-6; 1.279068f-6 5.6549666f-6 1.1169496f-5 4.938679f-5; 4.679406f-8 2.140484f-7 5.3935554f-7 1.766856f-6; 4.4654193f-8 1.6736452f-7 1.6050532f-7 9.845927f-7; 4.1866322f-8 5.5792987f-7 2.4880703f-7 2.1260563f-7; 6.5423967f-7 1.6410854f-6 1.5998732f-5 6.34402f-7; 8.0720376f-8 3.8200665f-7 4.7061158f-7 2.548826f-6; 6.3764007f-9 5.07034f-8 1.5942737f-7 1.8909933f-6; 1.3407076f-7 1.2524241f-8 1.1639756f-5 8.989167f-7; 5.7947403f-8 2.8609256f-7 9.590806f-7 2.5302238f-6; 4.8860308f-8 5.7413604f-7 3.9850502f-7 5.2059977f-7; 2.2891154f-8 3.192704f-7 1.5090544f-6 8.630877f-6; 7.995285f-8 2.1100962f-7 6.3255203f-7 1.7869452f-6; 1.5291464f-8 2.0042906f-8 3.15873f-7 2.2984419f-5; 5.631381f-7 4.8221054f-6 5.4192685f-7 3.684746f-6; 7.775973f-9 6.3027656f-8 5.9328494f-7 3.7174527f-8; 2.688885f-8 1.8444963f-7 3.65732f-7 1.2797616f-7; 1.0401096f-8 1.2392829f-7 3.0307558f-7 1.4480156f-6; 1.4701315f-7 9.0823323f-7 2.2061834f-6 7.5166565f-7; 5.9536617f-8 4.6841043f-7 3.330352f-6 2.2699686f-7; 2.613853f-8 7.1956784f-8 2.0931277f-6 1.839243f-7; 9.31662f-9 9.849815f-9 5.333149f-7 7.5725706f-7; 1.9711214f-8 7.359311f-8 6.1284973f-6 1.4810817f-7; 2.6504969f-8 1.1827153f-8 4.332167f-7 9.698992f-8; 4.213284f-8 1.715555f-7 3.7851232f-6 1.1141073f-7; 2.3250428f-9 3.584823f-8 1.808514f-7 2.9080587f-7; 3.243043f-8 1.4469876f-7 2.8045693f-7 3.243794f-6; 2.0994391f-6 1.8501713f-6 3.5666588f-5 1.6736615f-7; 5.5119663f-6 3.1248235f-6 2.845521f-6 2.8674137f-7; 2.8206918f-7 2.40385f-6 4.460225f-6 0.000615658; 3.177688f-6 5.266183f-6 2.7028967f-5 0.00015491586; 3.1650586f-7 2.8147184f-7 4.6788273f-6 3.192292f-6; 1.4900549f-7 4.9840486f-7 2.3589587f-6 0.000114420225; 1.0018415f-6 1.2958571f-6 5.3425325f-8 6.811624f-6; 7.4855376f-7 7.700046f-7 1.8801267f-5 7.771735f-5; 1.3699238f-6 1.359818f-6 1.9413735f-6 7.645848f-7; 2.4961932f-7 1.9727338f-6 5.669804f-7 8.147689f-6; 6.4860814f-7 3.6931405f-7 9.047882f-6 1.3650114f-6; 6.134846f-7 1.4806617f-6 2.1199246f-6 1.2456113f-6; 9.943693f-7 9.253388f-7 7.165744f-7 2.3017496f-5; 2.9906275f-6 7.92138f-6 2.186955f-5 3.696221f-7; 5.8302436f-7 1.5725794f-6 2.537061f-6 6.470255f-7; 3.7774947f-7 6.4628985f-8 4.054257f-6 8.8042947f-7; 2.9588991f-6 3.467298f-6 4.4717322f-5 1.7851941f-6; 9.4694104f-7 6.019442f-6 5.9935237f-6 9.274161f-5; 3.5901806f-8 2.1693494f-7 6.836238f-6 6.2037014f-7; 1.7707647f-8 6.03484f-7 1.805932f-7 1.3763499f-6; 1.6682166f-6 2.6856544f-6 8.958647f-6 8.780863f-8; 7.475385f-9 1.7543325f-8 7.253918f-7 1.3076234f-6; 2.0027922f-8 6.2015246f-8 3.699489f-7 1.1790255f-6; 1.4729659f-8 8.219807f-8 9.079005f-8 2.20716f-6; 1.9869209f-7 1.7366732f-7 5.4297443f-6 4.862006f-6; 5.191371f-8 4.5365874f-7 7.6199946f-8 6.911949f-7; 0.0 0.0 0.0 0.0; 0.60105085 0.6647045 0.530212 0.71403086; 3.0 3.0 3.0 17.0]
    expected_result = sortslices(expected_result, dims=2, by = x -> x[end-1])

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
                batch_result = sortslices(res[1:end-1, res[end,:] .== b], dims=2, by = x -> x[end-1])
                @test batch_result ≈ expected_result
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
