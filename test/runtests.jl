using ObjectDetector
using Test, PrettyTables
using FileIO, ImageCore
using LazyArtifacts
using ReferenceTests
using CUDA
using cuDNN

include("prepareImage.jl")
include("maintests.jl")
include("drawBoxes.jl")
