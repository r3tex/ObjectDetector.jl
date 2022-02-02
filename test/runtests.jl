using ObjectDetector
using Test, PrettyTables
using FileIO, ImageCore
using LazyArtifacts
using ReferenceTests

include("prepareImage.jl")
include("maintests.jl")
include("drawBoxes.jl")
