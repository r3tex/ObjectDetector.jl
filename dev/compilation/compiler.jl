## For generating a sysimage with ObjectDetector built-in, to make loading faster
# Make sure to have an un-modified sysimage. i.e. re-download julia

# Step 1: Add PackageCompilerX and activate the folder this file sits in
# Step 2: Make sure ObjectDetector loads
# Step 3: Run
# Note: You will likely see Zygote-related precompile failures

using PackageCompilerX
create_sysimage(:ObjectDetector, sysimage_path=joinpath(@__DIR__, "ObjectDetector.so"),
        precompile_execution_file = joinpath(@__DIR__, "precompile.jl"))

# Step 4: Launch julia pointing to this new sysimage
# i.e. run this in bash:
#==
cd /path/to/ObjectDetector.jl/dev/compilation
julia-1.3 -q -JObjectDetector.so
==#
#(~5 seconds to loaded REPL with ObjectDetector loaded)
#==
julia> @time ObjectDetector.YOLO.v3_608_COCO(silent=false)
4.096615 seconds (4.74 M allocations: 2.239 GiB, 12.35% gc time)
==#
