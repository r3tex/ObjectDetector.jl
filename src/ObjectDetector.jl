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

if !isdefined(Base, :get_extension)
    include("../ext/CUDAExt.jl")
end

end #module
