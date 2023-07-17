module ObjectDetector
export YOLO
export prepareImage, prepareImage!, resizekern, sizethatfits, emptybatch, drawBoxes

using CUDA
import cuDNN # not used but needed to load Flux CUDA Exts in Flux 0.14+
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
using PrecompileTools

abstract type AbstractModel end
function getModelInputSize end

include("prepareimage.jl")

## YOLO models
include(joinpath(@__DIR__,"yolo","yolo.jl"))
import .YOLO

include("utils.jl")

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
