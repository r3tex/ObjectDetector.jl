function _precompile_()
    ccall(:jl_generating_output, Cint, ()) == 1 || return nothing
    isdefined(ObjectDetector.YOLO, Symbol("#7#19")) && precompile(Tuple{getfield(ObjectDetector.YOLO, Symbol("#7#19")),Tuple{Int64,Symbol}})
    isdefined(ObjectDetector.YOLO, Symbol("#9#21")) && precompile(Tuple{getfield(ObjectDetector.YOLO, Symbol("#9#21")),Function})
    isdefined(ObjectDetector.YOLO, Symbol("#9#21")) && precompile(Tuple{getfield(ObjectDetector.YOLO, Symbol("#9#21")),Nothing})
    isdefined(ObjectDetector.YOLO, Symbol("#9#21")) && precompile(Tuple{getfield(ObjectDetector.YOLO, Symbol("#9#21")),Tuple{Int64,Symbol}})
    precompile(Tuple{Type{Base.Broadcast.Broadcasted{Base.Broadcast.DefaultArrayStyle{4},Axes,F,Args} where Args<:Tuple where F where Axes},typeof(ObjectDetector.YOLO.leaky),Tuple{Array{Float32,4}}})
    precompile(Tuple{typeof(Base.Broadcast.copyto_nonleaf!),Array{Float64,1},Base.Broadcast.Broadcasted{Base.Broadcast.DefaultArrayStyle{1},Tuple{Base.OneTo{Int64}},typeof(ObjectDetector.YOLO.cfgparse),Tuple{Base.Broadcast.Extruded{Array{SubString{String},1},Tuple{Bool},Tuple{Int64}}}},Base.OneTo{Int64},Int64,Int64})
    precompile(Tuple{typeof(Base.Broadcast.copyto_nonleaf!),Array{Int64,1},Base.Broadcast.Broadcasted{Base.Broadcast.DefaultArrayStyle{1},Tuple{Base.OneTo{Int64}},typeof(ObjectDetector.YOLO.cfgparse),Tuple{Base.Broadcast.Extruded{Array{SubString{String},1},Tuple{Bool},Tuple{Int64}}}},Base.OneTo{Int64},Int64,Int64})
    precompile(Tuple{typeof(Base.Broadcast.instantiate),Base.Broadcast.Broadcasted{Base.Broadcast.DefaultArrayStyle{4},Nothing,typeof(ObjectDetector.YOLO.leaky),Tuple{Array{Float32,4}}}})
    precompile(Tuple{typeof(ObjectDetector.YOLO.upsample),Array{Float32,4},Int64})
    precompile(Tuple{typeof(ObjectDetector.YOLO.v3_320_COCO)})
    precompile(Tuple{typeof(copy),Base.Broadcast.Broadcasted{Base.Broadcast.DefaultArrayStyle{4},NTuple{4,Base.OneTo{Int64}},typeof(ObjectDetector.YOLO.leaky),Tuple{Array{Float32,4}}}})
end
