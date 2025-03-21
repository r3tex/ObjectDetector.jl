module ObjectDetector
export YOLO
export prepareImage, prepareImage!, resizekern, sizethatfits, emptybatch, drawBoxes

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

if !isdefined(Base, :get_extension)
    include("../ext/CUDAExt.jl")
end

@setup_workload begin
    @compile_workload begin
        # don't use GPU here because GPU compilation of Conv requires realistic weights not dummy weights
        yolomod = YOLO.v3_COCO(dummy=true, silent=true, use_gpu=false)
        batch = emptybatch(yolomod)
        res = yolomod(batch)
        res = nothing
        batch = nothing
        yolomod = nothing
    end
end

end #module
