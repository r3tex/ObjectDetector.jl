## For generating a sysimage with ObjectDetector built-in, to make loading fast
# Make sure to have an un-modified sysimage. i.e. re-download julia

# Step 1: Activate the folder this file sits in

# Step 2: Make sure ObjectDetector loads
using ObjectDetector

# Step 3: Run
using PackageCompilerX
create_sysimage(:ObjectDetector, sysimage_path=joinpath(@__DIR__, "ObjectDetector.so"),
        precompile_statements_file = [joinpath(@__DIR__, "precompile.jl")])
