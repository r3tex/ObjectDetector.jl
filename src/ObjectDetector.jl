module ObjectDetector
export YOLO
export prepare_image, prepare_image!, resizekern, sizethatfits, emptybatch, draw_boxes
export train!, save_weights, TrainSample, load_darknet_dataset, load_darknet_labels

import Flux.gpu

using ImageFiltering
using ImageTransformations
using ImageCore

using ImageDraw
using PrecompileTools
using TimerOutputs
using AllocArrays
using Adapt
using Cairo
using Colors

const to = TimerOutput()

"""
    benchmark(; models = sort(collect(keys(YOLO.YOLO_MODELS))), kw...)

Benchmark the pretrained models. Requires BenchmarkTools and PrettyTables to
be loaded first: `using BenchmarkTools, PrettyTables`.
"""
function benchmark end

function __init__()
    Base.Experimental.register_error_hint(MethodError) do io, exc, argtypes, kwargs
        if exc.f === benchmark
            print(io, "\nObjectDetector.benchmark requires BenchmarkTools and PrettyTables: run `using BenchmarkTools, PrettyTables` first.")
        end
    end
end

abstract type AbstractModel end
function get_input_size end

include("prepareimage.jl")
include("allocators.jl")
include("training_data.jl")

function uses_gpu end
function get_cfg end
function train! end
function save_weights end
# Not exported: Metalhead.jl exports `backbone` for the same concept, and a
# clash would make every unqualified call an error in a plausible pairing.
function backbone end
function copy_backbone! end

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
