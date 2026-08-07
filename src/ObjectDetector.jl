module ObjectDetector
export YOLO
export prepare_image, prepare_image!, resizekern, sizethatfits, emptybatch, draw_boxes

import Flux.gpu

using ImageFiltering
using ImageTransformations
using ImageCore

using BenchmarkTools
using PrettyTables
using ImageDraw
using PrecompileTools
using TimerOutputs
using AllocArrays
using Adapt
using Cairo
using Colors

const to = TimerOutput()

abstract type AbstractModel end
function get_input_size end

include("prepareimage.jl")
include("allocators.jl")

function uses_gpu end
function get_cfg end

## YOLO models
include(joinpath(@__DIR__,"yolo","yolo.jl"))
import .YOLO

include("utils.jl")

@setup_workload begin
    @compile_workload begin
        # A dummy-weight model needs no downloads and exercises cfg parsing,
        # chain construction, and the full inference + NMS path
        model = YOLO.Yolo(joinpath(YOLO.models_dir(), "yolov3-tiny.cfg"), nothing, 1;
                          silent=true, cfgchanges=[(:net, 1, :width, 160), (:net, 1, :height, 160)])
        batch = emptybatch(model)
        model(batch; detect_thresh=0.0, overlap_thresh=0.5)
    end
end

end #module
