using ObjectDetector: YOLO, TrainSample, evaluate, DetectionMetrics
using ObjectDetector: average_precision, box_iou
import Flux

@testset "Evaluation" begin
    @testset "box_iou" begin
        a = (0.0f0, 0.0f0, 1.0f0, 1.0f0)
        @test box_iou(a, a) ≈ 1
        @test box_iou(a, (2.0f0, 2.0f0, 3.0f0, 3.0f0)) == 0    # disjoint
        @test box_iou(a, (1.0f0, 0.0f0, 2.0f0, 1.0f0)) == 0    # touching edges only
        # half-overlapping unit squares: intersection 1/2, union 3/2
        @test box_iou(a, (0.5f0, 0.0f0, 1.5f0, 1.0f0)) ≈ 1 / 3
        # contained box: intersection is the small one
        @test box_iou(a, (0.25f0, 0.25f0, 0.75f0, 0.75f0)) ≈ 0.25
    end

    @testset "average_precision" begin
        @test average_precision(trues(10), 10) ≈ 1          # perfect
        @test average_precision(falses(10), 10) == 0        # nothing correct
        @test isnan(average_precision(Bool[], 0))           # class absent from the data
        @test average_precision(Bool[], 5) == 0             # present but never detected
        # every detection correct but only half the boxes found: recall caps at 0.5,
        # and the 101-point grid puts 51 of its points at or below that
        @test average_precision(trues(5), 10) ≈ 51 / 101
        # ranking matters: the same detections score higher when the hits come first
        front = average_precision(Bool[true, true, true, false, false, false], 3)
        back = average_precision(Bool[false, false, false, true, true, true], 3)
        @test front ≈ 1
        @test back < front
    end

    @testset "evaluate" begin
        cfgfile = joinpath(YOLO.models_dir(), "yolov3-tiny.cfg")
        model = YOLO.Yolo(cfgfile, nothing, 2; silent=true, use_gpu=false, disallow_bumper=true,
                          weights_stop_layer=0,
                          cfgchanges=[(:net, 1, :width, 160), (:net, 1, :height, 160)])
        img = zeros(Float32, 160, 160, 3); img[40:90, 30:80, :] .= 1f0
        boxes = Float32[1; 0.34; 0.40; 0.31; 0.31;;]
        data = [TrainSample(img, boxes) for _ in 1:3]  # not a multiple of the batch size

        m = evaluate(model, data; silent=true)
        @test m isa DetectionMetrics
        @test sum(m.ngt) == 3
        @test size(m.ap) == (80, 10)
        @test m.iou_thresholds[1] ≈ 0.5
        @test count(!isnan, m.ap[:, 1]) == 1        # only class 1 has ground truth
        @test all(x -> isnan(x) || 0 <= x <= 1, m.ap)
        @test isnan(m.mAP) || 0 <= m.mAP <= 1
        @test occursin("mAP", sprint(show, m))

        @test_throws ArgumentError evaluate(model, TrainSample[]; silent=true)
        # path-backed samples need a loader, same as train!
        @test_throws ArgumentError evaluate(model, [TrainSample("nope.jpg", boxes)]; silent=true)
    end
end
