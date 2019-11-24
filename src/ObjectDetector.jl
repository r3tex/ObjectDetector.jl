module ObjectDetector
export YOLO
export resizePadImage, resizekern, sizethatfits, emptybatch

import Flux.gpu
export gpu

using Pkg.Artifacts
getArtifact(name::String) = joinpath(@artifact_str(name), "$(name).weights")

using ImageFiltering
using ImageTransformations
using ImageCore

using BenchmarkTools
using PrettyTables

abstract type Model end
function getModelInputSize end

include("prepareimage.jl")
const models_dir = joinpath(dirname(@__DIR__), "models")

## YOLO models
include(joinpath(@__DIR__,"yolo","yolo.jl"))
import .YOLO

include("utils.jl")

end #module
