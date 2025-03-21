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
    println(io, wm.allocator)
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
get_input_size(wm::AllocWrappedModel) = get_input_size(wm.model)

"""
    wrap_model(model; n_bytes=2^33, T=AllocArray)

Wraps a model to use a bump allocator for temporary arrays. The `n_bytes` of memory will be preallocated upfront once when `wrap_model` is called, and re-used for every batch inferred over. The type `T` can be set to `CheckedAllocArray` for testing purposes.
"""
function wrap_model(model; T=AllocArray)
    b = BumperAllocator()
    model_aa = Adapt.adapt(T, model)
    return AllocWrappedModel(model, b, T, model_aa)
end
