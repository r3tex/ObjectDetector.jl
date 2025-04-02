using ObjectDetector
using Test, PrettyTables
using FileIO, ImageCore
using LazyArtifacts
using ReferenceTests
using CUDA
using cuDNN
using Darknet

include("prepare_image.jl")
include("maintests.jl")
include("draw_boxes.jl")
