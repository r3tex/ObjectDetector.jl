struct WrappedModel
    model::AbstractModel
    allocator
    T
    model_aa
end
function (wm::WrappedModel)(args...; kw...)
    with_allocator(wm.allocator) do
        inputs = Adapt.adapt(wm.T, args)
        ret = Array(wm.model_aa(inputs...; kw...))
        reset!(wm.allocator)
        return ret
    end
end
emptybatch(wm::WrappedModel) = emptybatch(wm.model)
prepareImage(img::AbstractArray, model::WrappedModel) = prepareImage(img, model.model)
uses_gpu(wm::WrappedModel) = uses_gpu(wm.model)
drawBoxes(img, wm::WrappedModel, padding, results; kw...) = drawBoxes(img, wm.model, padding, results; kw...)

"""
    wrap_model(model; n_bytes=2^33, T=AllocArray)

Wraps a model to use a bump allocator for temporary arrays. The `n_bytes` of memory will be preallocated upfront once when `wrap_model` is called, and re-used for every batch inferred over. The type `T` can be set to `CheckedAllocArray` for testing purposes.
"""
function wrap_model(model; n_bytes=2^33, T=AllocArray)
    b = BumperAllocator(n_bytes)
    model_aa = Adapt.adapt(T, model)
    return WrappedModel(model, b, T, model_aa)
end
