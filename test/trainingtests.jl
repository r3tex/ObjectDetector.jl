using ObjectDetector: YOLO, TrainSample, train!, save_weights
import Flux
using Random: MersenneTwister

@testset "Training" begin
    cfgfile = joinpath(YOLO.models_dir(), "yolov3-tiny.cfg")
    cfgchanges = [(:net, 1, :width, 160), (:net, 1, :height, 160)]
    # random-init model (weights_stop_layer=0 randomizes every layer)
    model = YOLO.Yolo(cfgfile, nothing, 1; silent=true, use_gpu=false, disallow_bumper=true,
                      weights_stop_layer=0, cfgchanges)

    # toy dataset: white rectangles on black backgrounds, with darknet-style
    # normalized (class, cx, cy, w, h) ground truth
    img1 = zeros(Float32, 120, 160, 3); img1[30:60, 40:80, :] .= 1f0
    img2 = zeros(Float32, 160, 160, 3); img2[100:140, 20:50, :] .= 1f0
    boxes1 = Float32[1; (40 + 80) / 2 / 160; (30 + 60) / 2 / 120; 40 / 160; 30 / 120;;]
    boxes2 = Float32[1; (20 + 50) / 2 / 160; (100 + 140) / 2 / 160; 30 / 160; 40 / 160;;]
    data = [TrainSample(img1, boxes1), TrainSample(img2, boxes2)]

    @testset "pure forward matches buffered forward" begin
        x = rand(MersenneTwister(0), Float32, 160, 160, 3, 1)
        heads = YOLO.train_heads(model)
        raws = YOLO.raw_head_outputs(model.chain, x, map(h -> h.chainidx, heads))
        model.W[0] = copy(x)
        for i in eachindex(model.chain)
            model.W[i] .= model.chain[i](model.W[i-1])
        end
        for (hi, h) in enumerate(heads)
            @test isapprox(raws[hi], model.W[h.chainidx]; rtol=1e-5)
        end
    end

    @testset "targets and loss" begin
        heads = YOLO.train_heads(model)
        x, boxes_batch = ObjectDetector.make_training_batch(model, data)
        @test size(x) == (160, 160, 3, 2)
        tgts_cpu = YOLO.build_targets(heads, boxes_batch)
        @test sum(t -> t.npos[], tgts_cpu) == 2 # one positive per image
        tgts = [(tobj=t.tobj, tx=t.tx, ty=t.ty, tw=t.tw, th=t.th, twr=t.twr, thr=t.thr, tcls=t.tcls, npos=t.npos[]) for t in tgts_cpu]
        l = YOLO.train_loss(model.chain, x, heads, tgts)
        @test isfinite(l)
        l2, gs = Flux.withgradient(m -> YOLO.train_loss(m, x, heads, tgts), model.chain)
        @test l2 == l
        g1 = gs[1].layers[1].layers[1].layers[1].weight
        @test g1 !== nothing
        @test any(!=(0), g1)
    end

    @testset "overfit toy dataset" begin
        res = train!(model, data; epochs=25, batchsize=2, lr=1e-3, silent=true,
                     rng=MersenneTwister(0))
        @test length(res.losses) == 25
        @test all(isfinite, res.losses)
        @test res.losses[end] < 0.5 * res.losses[1]
        # inference still works on the same model afterwards
        batch = emptybatch(model)
        @test model(batch; detect_thresh=0.5) isa AbstractArray
    end

    @testset "save_weights roundtrip" begin
        mktempdir() do dir
            path = save_weights(model, joinpath(dir, "trained.weights"))
            reloaded = YOLO.Yolo(cfgfile, path, 1; silent=true, use_gpu=false,
                                 disallow_bumper=true, cfgchanges)
            convs = YOLO.conv_layers(model.chain)
            convs2 = YOLO.conv_layers(reloaded.chain)
            @test length(convs) == length(convs2)
            for (c1, c2) in zip(convs, convs2)
                @test isapprox(c1.weight, c2.weight; rtol=1e-5)
                @test isapprox(c1.bias, c2.bias; rtol=1e-4)
            end
        end
    end

    @testset "checkpointing" begin
        mktempdir() do dir
            train!(model, data; epochs=2, batchsize=2, lr=1e-4, silent=true,
                   checkpoint_dir=dir, checkpoint_every=1)
            @test isfile(joinpath(dir, "checkpoint-epoch0001.weights"))
            @test isfile(joinpath(dir, "checkpoint-epoch0002.weights"))
        end
    end

    @testset "darknet label loading" begin
        mktempdir() do dir
            write(joinpath(dir, "img1.txt"), "0 0.5 0.5 0.25 0.25\n1 0.1 0.2 0.05 0.05\n")
            boxes = ObjectDetector.load_darknet_labels(joinpath(dir, "img1.txt"))
            @test size(boxes) == (5, 2)
            @test boxes[1, :] == [1f0, 2f0] # classes converted to 1-based
            @test boxes[2:5, 1] == Float32[0.5, 0.5, 0.25, 0.25]
        end
    end

    @testset "letterbox box transform" begin
        # a 2:1 wide image letterboxed into a square model input is padded
        # equally top and bottom; y coordinates compress by 2, x is unchanged
        img = zeros(Float32, 80, 160, 3)
        arr, pad = ObjectDetector._prepare_train_image(img, (160, 160, 3))
        boxes = ObjectDetector.letterbox_boxes(Float32[1; 0.5; 0.5; 0.5; 0.5;;], pad)
        @test boxes[2] ≈ 0.5
        @test boxes[3] ≈ 0.5
        @test boxes[4] ≈ 0.5
        @test boxes[5] ≈ 0.25
    end

    @testset "trainable_batchnorm matches folded inference" begin
        cfg, weights = YOLO.YOLO_MODELS["v3_tiny_COCO"]()
        mf = YOLO.Yolo(cfg, weights, 1; silent=true, use_gpu=false, disallow_bumper=true, cfgchanges)
        mb = YOLO.Yolo(cfg, weights, 1; silent=true, use_gpu=false, disallow_bumper=true, cfgchanges,
                       trainable_batchnorm=true)
        @test count(p -> last(p) !== nothing, YOLO.conv_bn_layers(mb.chain)) == 11
        @test count(p -> last(p) !== nothing, YOLO.conv_bn_layers(mf.chain)) == 0
        x = rand(MersenneTwister(1), Float32, 160, 160, 3, 1)
        hf = YOLO.train_heads(mf)
        rf = YOLO.raw_head_outputs(mf.chain, x, map(h -> h.chainidx, hf))
        rb = YOLO.raw_head_outputs(mb.chain, x, map(h -> h.chainidx, YOLO.train_heads(mb)))
        for i in eachindex(rf)
            @test maximum(abs.(rf[i] .- rb[i])) < 1e-3
        end
    end

    @testset "from-scratch training (live batchnorm)" begin
        ms = YOLO.Yolo(cfgfile, nothing, 1; silent=true, use_gpu=false, disallow_bumper=true,
                       weights_stop_layer=0, trainable_batchnorm=true, cfgchanges)
        res = train!(ms, data; epochs=25, batchsize=2, lr=1e-3, silent=true,
                     warmup_batches=5, flip_augment=true, rng=MersenneTwister(0))
        @test all(isfinite, res.losses)
        @test res.losses[end] < 0.5 * res.losses[1]
        # saving writes real bn params; reloading (folded) gives equivalent outputs
        mktempdir() do dir
            path = save_weights(ms, joinpath(dir, "scratch.weights"))
            mr = YOLO.Yolo(cfgfile, path, 1; silent=true, use_gpu=false, disallow_bumper=true, cfgchanges)
            x = rand(MersenneTwister(2), Float32, 160, 160, 3, 1)
            hs = YOLO.train_heads(ms)
            rs = YOLO.raw_head_outputs(ms.chain, x, map(h -> h.chainidx, hs))
            rr = YOLO.raw_head_outputs(mr.chain, x, map(h -> h.chainidx, YOLO.train_heads(mr)))
            for i in eachindex(rs)
                @test maximum(abs.(rs[i] .- rr[i])) < 1e-2
            end
        end
    end

    @testset "yolov4-tiny (classic decode, grouped routes) training" begin
        # note: the official yolov4-tiny cfg has new_coords commented out; it
        # is a classic-decode model with scale_x_y=1.05 and grouped routes
        m4 = YOLO.Yolo(joinpath(YOLO.models_dir(), "yolov4-tiny.cfg"), nothing, 1;
                       silent=true, use_gpu=false, disallow_bumper=true,
                       weights_stop_layer=0, trainable_batchnorm=true, cfgchanges)
        heads = YOLO.train_heads(m4)
        @test all(h -> !h.new_coords, heads)
        @test any(h -> h.sxy != 1f0, heads)
        res = train!(m4, data; epochs=25, batchsize=2, lr=1e-3, silent=true,
                     warmup_batches=5, rng=MersenneTwister(0))
        @test all(isfinite, res.losses)
        @test res.losses[end] < 0.5 * res.losses[1]
    end

    @testset "yolov7-tiny (new_coords) training" begin
        m7 = YOLO.Yolo(joinpath(YOLO.models_dir(), "yolov7-tiny.cfg"), nothing, 1;
                       silent=true, use_gpu=false, disallow_bumper=true,
                       weights_stop_layer=0, trainable_batchnorm=true, cfgchanges)
        heads = YOLO.train_heads(m7)
        @test all(h -> h.new_coords, heads)
        res = train!(m7, data; epochs=20, batchsize=2, lr=1e-3, silent=true,
                     warmup_batches=5, rng=MersenneTwister(0))
        @test all(isfinite, res.losses)
        @test res.losses[end] < 0.5 * res.losses[1]
    end

    @testset "yolov4 (classic, mish) gradient" begin
        m4f = YOLO.Yolo(joinpath(YOLO.models_dir(), "yolov4.cfg"), nothing, 1;
                        silent=true, use_gpu=false, disallow_bumper=true, weights_stop_layer=0,
                        cfgchanges=[(:net, 1, :width, 128), (:net, 1, :height, 128)])
        h4 = YOLO.train_heads(m4f)
        @test all(h -> !h.new_coords, h4)
        xb, bb = ObjectDetector.make_training_batch(m4f, data[1:1])
        tg = [(tobj=t.tobj, tx=t.tx, ty=t.ty, tw=t.tw, th=t.th, twr=t.twr, thr=t.thr, tcls=t.tcls, npos=t.npos[])
              for t in YOLO.build_targets(h4, bb)]
        l, gs = Flux.withgradient(m -> YOLO.train_loss(m, xb, h4, tg), m4f.chain)
        @test isfinite(l)
        g1 = gs[1].layers[1].layers[1].layers[1].weight
        @test g1 !== nothing
        @test any(!=(0), g1)
    end

    @testset "pretrained transfer with custom classes" begin
        cfg, weights = YOLO.YOLO_MODELS["v3_tiny_COCO"]()
        changes = [(:net, 1, :width, 160), (:net, 1, :height, 160),
                   (:yolo, 1, :classes, 2), (:yolo, 2, :classes, 2),
                   (:convolutional, 10, :filters, 21), (:convolutional, 13, :filters, 21)]
        # stop at block 15: the yolov3-tiny.conv.15 backbone split
        tmodel = YOLO.Yolo(cfg, weights, 1; silent=true, use_gpu=false, disallow_bumper=true,
                           weights_stop_layer=15, cfgchanges=changes)
        @test ObjectDetector.get_cfg(tmodel)[:output][1][:classes] == 2
        img = zeros(Float32, 160, 160, 3); img[60:100, 50:110, :] .= 1f0
        boxes = Float32[2; 0.5; 0.5; 60 / 160; 40 / 160;;]
        # single-image Adam fine-tuning oscillates and BLAS differences make the
        # exact trajectory platform-dependent, so compare smoothed endpoints at
        # a gentle LR rather than two single noisy epoch losses
        res = train!(tmodel, [TrainSample(img, boxes)]; epochs=15, batchsize=1, lr=5e-4,
                     warmup_batches=2, silent=true, rng=MersenneTwister(0))
        @test sum(res.losses[end-2:end]) / 3 < res.losses[1]
    end
end
