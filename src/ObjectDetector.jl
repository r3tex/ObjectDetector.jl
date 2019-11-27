module ObjectDetector
export YOLO
export resizePadImage, resizePadImage!, resizekern, sizethatfits, emptybatch, drawBoxes

import Flux.gpu
import Flux.cpu
export gpu, cpu

using Pkg.Artifacts
getArtifact(name::String) = joinpath(@artifact_str(name), "$(name).weights")

using ImageFiltering
using ImageTransformations
using ImageCore

using BenchmarkTools
using PrettyTables
using ImageDraw

abstract type Model end
function getModelInputSize end

include("prepareimage.jl")

## YOLO models
include(joinpath(@__DIR__,"yolo","yolo.jl"))
import .YOLO

include("utils.jl")

end #module
