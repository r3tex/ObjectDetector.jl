module ObjectDetector
export YOLO
export prepareImage, prepareImage!, resizekern, sizethatfits, emptybatch, drawBoxes

import Flux.gpu
import Flux.cpu
export gpu, cpu

using Pkg.Artifacts
#getArtifact(name::String) = joinpath(@artifact_str(name), "$(name).weights") #Broken in 1.3.1 https://github.com/JuliaLang/Pkg.jl/issues/1579

using ImageFiltering
using ImageTransformations
using ImageCore

using BenchmarkTools
using PrettyTables
using ImageDraw

abstract type AbstractModel end
function getModelInputSize end

include("prepareimage.jl")

## YOLO models
include(joinpath(@__DIR__,"yolo","yolo.jl"))
import .YOLO

include("utils.jl")

include("../deps/SnoopCompile/precompile/precompile_ObjectDetector.jl")
_precompile_()

end #module
