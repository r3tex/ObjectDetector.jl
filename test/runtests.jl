using CUDA
using cuDNN
using Darknet
using FileIO
using Flux: cpu
using ImageCore
using LazyArtifacts
using OrderedCollections: OrderedDict
using PrettyTables
using ReferenceTests
using Suppressor
using Test

using ObjectDetector

@testset "ObjectDetector" verbose=true begin
    include("prepare_image.jl")
    include("maintests.jl")
end
