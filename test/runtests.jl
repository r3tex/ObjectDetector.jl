using ObjectDetector
using Test, PrettyTables
using FileIO, ImageCore
using LazyArtifacts

include("prepareImage.jl")
include("maintests.jl")
include("drawBoxes.jl")
