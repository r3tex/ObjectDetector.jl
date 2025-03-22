mutable struct AllocWrappedModel
    model::AbstractModel
    allocator
    allocator_size::Int
    T
    model_aa
end
function Base.show(io::IO, wm::AllocWrappedModel)
    println(io, "AllocWrappedModel")
    println(io, "Model:")
    println(io, wm.model)
    println(io, "Allocator:")
    println(io, wm.allocator)
end
function (wm::AllocWrappedModel)(args...; kw...)
    with_allocator(wm.allocator) do
        try
            inputs = Adapt.adapt(wm.T, args)
            ret = Array(wm.model_aa(inputs...; kw...))
            return ret
        finally
            alloc = allocated_bytes(wm.allocator)
            reset!(wm.allocator)
            if alloc !== nothing && alloc > wm.allocator_size
                healthy_alloc = trunc(Int, alloc * 1.1)
                @debug "Resizing allocator to: allocated ($(Base.format_bytes(alloc))) x 1.1 = $(Base.format_bytes(healthy_alloc))"
                wm.allocator = BumperAllocator(SlabBuffer{healthy_alloc}())
                wm.allocator_size = healthy_alloc
            end
        end
    end
end
emptybatch(wm::AllocWrappedModel) = emptybatch(wm.model)
prepare_image(img::AbstractArray, model::AllocWrappedModel) = prepare_image(img, model.model)
uses_gpu(wm::AllocWrappedModel) = uses_gpu(wm.model)
get_cfg(wm::AllocWrappedModel) = get_cfg(wm.model)
draw_boxes(img, wm::AllocWrappedModel, padding, results; kw...) = draw_boxes(img, wm.model, padding, results; kw...)
draw_boxes!(img::AbstractArray, wm::AllocWrappedModel, padding::AbstractArray, results; kw...) = draw_boxes!(img, wm.model, padding, results; kw...)
get_input_size(wm::AllocWrappedModel) = get_input_size(wm.model)

allocated_bytes(alloc::BumperAllocator) = allocated_bytes(alloc.bumper)
allocated_bytes(alloc::UncheckedBumperAllocator) = allocated_bytes(alloc.buf)
function allocated_bytes(buf::SlabBuffer{SlabSize}) where {SlabSize}
    current_slab_used = buf.current - buf.slabs[end]
    return ((length(buf.slabs) - 1) * SlabSize) + current_slab_used
end
allocated_bytes(::Any) = nothing

const DEFAULT_SLAB_SIZE = 2^29 # 512 MB, see https://github.com/r3tex/ObjectDetector.jl/pull/109

"""
    wrap_model(model; T=AllocArray, allocator=BumperAllocator(SlabBuffer{$DEFAULT_SLAB_SIZE}()))

Wraps a model to use a Bumper.jl allocator for temporary arrays. The type `T` can be set to `CheckedAllocArray` for testing purposes.
"""
function wrap_model(model; T=AllocArray, allocator=BumperAllocator(SlabBuffer{DEFAULT_SLAB_SIZE}()))
    model_aa = Adapt.adapt(T, model)
    return AllocWrappedModel(model, allocator, 0, T, model_aa)
end
