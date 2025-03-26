struct AllocWrappedModel
    model::AbstractModel
    allocator
    T
    model_aa
end
function Base.show(io::IO, wm::AllocWrappedModel)
    println(io, "AllocWrappedModel")
    println(io, "Model:")
    println(io, wm.model)
    println(io, "Allocator:")
    print(io, wm.allocator)
end
function (wm::AllocWrappedModel)(args...; kw...)
    with_allocator(wm.allocator) do
        try
            inputs = Adapt.adapt(wm.T, args)
            ret = Array(wm.model_aa(inputs...; kw...))
            return ret
        finally
            reset!(wm.allocator)
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


"""
    wrap_model(model; T=AllocArray, allocator=BumperAllocator())

Wraps a model to use a Bumper.jl allocator for temporary arrays. The type `T` can be set to `CheckedAllocArray` for testing purposes.
"""
function wrap_model(model; T=AllocArray, allocator=BumperAllocator())
    model_aa = Adapt.adapt(T, model)
    return AllocWrappedModel(model, allocator, T, model_aa)
end
